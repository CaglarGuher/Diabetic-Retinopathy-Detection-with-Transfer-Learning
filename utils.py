import pandas as pd
import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader,Dataset
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import roc_curve, auc
import logging
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.progress_reporter import CLIReporter
from ray import tune
from PIL import Image
def extract_bv(image):
    b,green_fundus,r = cv2.split(image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast_enhanced_green_fundus = clahe.apply(green_fundus)

    # applying alternate sequential filtering (3 times closing opening)
    r1 = cv2.morphologyEx(contrast_enhanced_green_fundus, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    R1 = cv2.morphologyEx(r1, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)), iterations = 1)
    r2 = cv2.morphologyEx(R1, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    R2 = cv2.morphologyEx(r2, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)), iterations = 1)
    r3 = cv2.morphologyEx(R2, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)
    R3 = cv2.morphologyEx(r3, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(23,23)), iterations = 1)	
    f4 = cv2.subtract(R3,contrast_enhanced_green_fundus)
    f5 = clahe.apply(f4)

    # removing very small contours through area parameter noise removal
    ret,f6 = cv2.threshold(f5,15,255,cv2.THRESH_BINARY)	
    mask = np.ones(f5.shape[:2], dtype="uint8") * 255	
    contours, hierarchy = cv2.findContours(f6.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        if cv2.contourArea(cnt) <= 200:
            cv2.drawContours(mask, [cnt], -1, 0, -1)
    im = cv2.bitwise_and(f5, f5, mask=mask)
    ret,fin = cv2.threshold(im,15,255,cv2.THRESH_BINARY_INV)
    newfin = cv2.erode(fin, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3)), iterations=1)	

    
    fundus_eroded = cv2.bitwise_not(newfin)	
    xmask = np.ones(fundus_eroded.shape[:2], dtype="uint8") * 255
    xcontours, xhierarchy = cv2.findContours(fundus_eroded.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)	
    for cnt in xcontours:
        shape = "unidentified"
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.04 * peri, False)   
        if len(approx) > 4 and cv2.contourArea(cnt) <= 3000 and cv2.contourArea(cnt) >= 100:
            shape = "circle"	
        else:
            shape = "veins"
        if(shape=="circle"):
            cv2.drawContours(xmask, [cnt], -1, 0, -1)

    finimage = cv2.bitwise_and(fundus_eroded,fundus_eroded,mask=xmask)	
    blood_vessels = cv2.bitwise_not(finimage)
    return blood_vessels

def blurry_or_not(laplacian_value):
    
    if laplacian_value <10000:
        return 0
    else:
        return 1
    
def preprocess_image(image):
  

    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5, tileGridSize=(10,10))
    l_normalized = clahe.apply(l)
    normalized_lab = cv2.merge((l_normalized, a, b))
    normalized_bgr = cv2.cvtColor(normalized_lab, cv2.COLOR_LAB2BGR)

    return normalized_bgr


def show_images_by_level(t_t_img_dir,dataset_total,filter=None):
    fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(20, 20))
    axs = axs.flatten()

    images_by_level = {level: [] for level in range(5)}
    for i in range(2000):
        i = random.randint(0,1000)
        
        level = dataset_total["level"][i]
        image_path = f'{t_t_img_dir }/{dataset_total["image"][i]}.jpg'
        image = cv2.imread(image_path)
        image = cv2.resize(image,(512,512))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if filter is not None:
            for j in range(len(filter)):
                image = filter[j](image)
        images_by_level[level].append(image)

    for i, level in enumerate(range(5)):
        images = images_by_level[level][:5]
        for j in range(len(images)):

            axs[5 * j + i].imshow((images[j]))
            axs[5 * j + i].set_title(f'Level: {level}')
            axs[5 * j + i].axis('off')
         
            axs[5 * j + i].text(0.5, -0.1, (cv2.Laplacian(extract_bv(images[j]), cv2.CV_64F).var()), transform=axs[5 * j + i].transAxes,
                                 fontsize=12, ha='center', va='bottom')
            
    plt.tight_layout()
    plt.show()




def crop_image(img,tol=7):
    """
    Crop out black borders
    https://www.kaggle.com/ratthachat/aptos-updated-preprocessing-ben-s-cropping
    """  
    
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mask = gray_img>tol        
        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img



def get_data(data_label, train_test_path, val_path, train_test_sample_size, batch_size, image_filter, model, validation=False):

    
    
    class data_adjust(Dataset): # Inherits from the Dataset class.

        def __init__(self,df,data_path,image_filter = None,image_transform=None,model=model): # Constructor.
            super(Dataset,self).__init__() #Calls the constructor of the Dataset class.
            self.df = df
            self.data_path = data_path
            self.image_transform = image_transform
            self.image_filter = image_filter
            self.model = model

            
        def __len__(self):
            return len(self.df) #Returns the number of samples in the dataset.
        
        def __getitem__(self,index):
            image_id = self.df['image'][index]
            image = cv2.imread(f'{self.data_path}/{image_id}.jpg') #Image.

            resize_224 = transforms.Compose([transforms.Resize([224,224])])
            resize_229 = transforms.Compose([transforms.Resize([229,229])])
            resize_256 = transforms.Compose([transforms.Resize([256,256])])
            resize_600 = transforms.Compose([transforms.Resize([600,600])])
            resize_528 = transforms.Compose([transforms.Resize([528,528])])
            resize_456 = transforms.Compose([transforms.Resize([456,456])])
            
            if self.image_filter:
                image = self.image_filter(image)

            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            if self.image_transform :


                if self.model in ["resnet152", "resnet101", "vgg19", "densenet161", "alexnet", "googlenet","wide_resnet101_2", 
                                  "mobilenet_v2", "shufflenet_v2_x1_0", "resnext50_32x4d", "wide_resnet50_2",]:
                    
                    image = resize_224(image)

                elif self.model == "inception_v3":
                    image = resize_229(image)

                elif self.model == "efficient-netb7":
                    image = resize_600(image)
                
                elif self.model == "efficient-netb6":
                    image = resize_528(image)

                elif self.model == "efficient-netb5":
                    image = resize_456(image)

                elif self.model in  ["resnext101_32x8d","resnext101_64x4d"]:
                    image = resize_256(image)
                    

                image = self.image_transform(image)
                
            
            label = self.df['level'][index] 
            return image,torch.tensor(label) 


    logging.info("Starting data preparation")
    
    df_test_train = (data_label[data_label["validation"] == 0].sample(n=train_test_sample_size))
    df_validation = data_label[data_label["validation"] == 1].sample(n = 250)
    df_validation = df_validation.reset_index()
    max_count = df_test_train["level"].value_counts().max()
    balanced_dfs = []

    logging.info("Balancing the train-test dataset")

    for label in df_test_train["level"].unique():
        subset = df_test_train[df_test_train["level"] == label]
        oversampled_subset = subset.sample(n=max_count, replace=True)
        balanced_dfs.append(oversampled_subset)

    balanced_train_test = pd.concat(balanced_dfs).reset_index()


    logging.info("Setting up data transforms")


    train_test_transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.GaussianBlur(kernel_size=3),
        transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.8, 1.2), shear=(-10, 10)),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(degrees=[-10, 10]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    logging.info("Preparing DataLoaders")

    if validation ==True:
        valid_data = data_adjust(df_validation, f'{val_path}', image_transform=valid_transform, image_filter=image_filter, model =model)
        valid_dataloader = DataLoader(valid_data, batch_size=1, shuffle=False)
        test_dataloader, train_dataloader = 0, 0

        logging.info("Validation DataLoader prepared")
    else:

        data_train_Test = data_adjust(balanced_train_test, f'{train_test_path}', image_transform=train_test_transform, image_filter=image_filter, model=model)
        train_set, valid_set = train_test_split(data_train_Test, test_size=0.2, random_state=42)

        train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        test_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        valid_dataloader = 0
        logging.info("Train and Test DataLoaders prepared")

    logging.info("Data preparation completed")

    return train_dataloader, test_dataloader, valid_dataloader

def optimize(model, model_name, train, test, device, data_label, path, path_for_val, tt_samp_size, batch_size, image_filter, lr, Epoch):

    logging.info(f"Starting optimization for model: {model_name}")

    
    model.to(device)
    train_loader, test_loader, valid_loader = get_data(data_label, path, path_for_val, tt_samp_size,
                                                       batch_size=batch_size, image_filter=image_filter, model=model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()

    logging.info("Training the model")
    for epoch in range(Epoch):
        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
        valid_loss, valid_acc = test(test_loader, model, loss_fn, device=device)
        wandb.log({"train_acc": train_acc, "train_loss": train_loss, "test_acc": valid_acc, "test_loss": valid_loss})
        logging.info(f"Epoch {epoch + 1}/{Epoch} - Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Test Loss: {valid_loss:.4f}, Test Acc: {valid_acc:.4f}")

    logging.info("Optimization completed")
    return {"loss": valid_loss}

def evaluate_model(model, Data_loaeder, device):
    model.eval()
    predictions = []

    with torch.no_grad():
        for images_batch in Data_loaeder:
            images = images_batch.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predictions.extend(predicted.cpu().numpy())

    return predictions



def plot_roc_curve(y_true, y_pred_prob, n_classes):
  
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], label=f'ROC curve of class {i} (area = {roc_auc[i]:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def evaluate_classification_metrics(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'precision': precision,
        'recall': recall,
        'confusion_matrix': conf_matrix
    }

def train(dataloader,model, loss_fn, optimizer, device):
        
        model.train()
        running_loss = 0
        correct = 0
        total = 0

        for batch, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            output = model(x)
            loss   = loss_fn(output, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        
        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        print('Train Loss: {:.4f} | Train Accuracy: {:.2f}%'.format(avg_loss, accuracy))

        return avg_loss, accuracy


def test(dataloader,model,loss_fn, device):

    model.eval() 
    running_loss = 0
    correct = 0
    total = 0

    with torch.no_grad(): 
        
        for (x, y) in dataloader:
            x, y = x.to(device), y.to(device) 
            
            output        = model(x)
            loss          = loss_fn(output, y).item()
            running_loss += loss
            
        
            _, predicted = torch.max(output.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()

    avg_loss = running_loss / len(dataloader)
    accuracy = 100 * correct / total

    print('Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'.format(avg_loss, accuracy))

    return avg_loss, accuracy


def get_predictions(model, data_loader,device):
    """
    Takes a trained PyTorch model and a DataLoader as input, and returns the model's predictions.

    Args:
        model (torch.nn.Module): A trained PyTorch model.
        data_loader (torch.utils.data.DataLoader): A DataLoader containing the input data.

    Returns:
        List[torch.Tensor]: A list of PyTorch tensors containing the model's predictions.
    """

    model.eval() 
    predictions = []  

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            inputs = batch[0]  
            inputs = inputs.to(device) 

        
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
       
            if (i + 1) % 10 == 0:
                logging.info(f'Processed {i + 1} batches out of {len(data_loader)}')

    return predictions


            