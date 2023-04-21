import pandas as pd
import torch
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import wandb
from sklearn.metrics import roc_curve, auc

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

    # removing blobs of unwanted bigger chunks taking in consideration they are not straight lines like blood
    #vessels and also in an interval of area
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
    # Ensure image is color image with 3 channels

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
            # Add label to the bottom of the image
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
def crop_image_only_outside(img,tol=0):
    # img is 2D or 3D image data
    # tol  is tolerance
    mask = img>tol
    if img.ndim==3:
        mask = mask.all(2)
    m,n = mask.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def crop_eye_image(img):
    # Load the image


    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Threshold the image to create a binary mask of the eye
    _, mask = cv2.threshold(gray, 8, 255, cv2.THRESH_BINARY)

    # Find the contours of the eye in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Find the largest contour (the outer boundary of the eye)
    largest_contour = max(contours, key=cv2.contourArea)

    # Find the bounding box of the largest contour
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Crop the image to the bounding box
    cropped_img = img[y:y+h, x:x+w]

    # Return the cropped image
    return cropped_img






def detect_symptoms(img):
    # Load the image
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply a median filter to remove noise
    gray = cv2.medianBlur(gray, 5)

    # Apply adaptive thresholding to segment the blood vessels and little dots
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # Perform morphological opening to remove small objects
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Detect microaneurysms
    circles = cv2.HoughCircles(opening, cv2.HOUGH_GRADIENT, 1, 200, param1=75, param2=25, minRadius=20, maxRadius=50)

    # Draw circles around the microaneurysms
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(img, (x, y), r, (0, 255, 0), 2)
    return img


def get_data(data_label,dataset,train_test_path,val_path,train_test_sample_size,batch_size,image_filter,model,validation = False):

    
    
    df_test_train = (data_label[data_label["validation"] == 0].sample(n = train_test_sample_size))
    df_validation = data_label[data_label["validation"] ==1]
    df_validation= df_validation.reset_index()
    max_count = df_test_train["level"].value_counts().max()
    balanced_dfs = []


    # For each unique class label
    for label in df_test_train["level"].unique():
        # Get the subset of samples with the current label
        subset = df_test_train[df_test_train["level"] == label]
        
        # Oversample the subset to match the maximum count
        oversampled_subset = subset.sample(n=max_count, replace=True)
        
        # Append the oversampled subset to the balanced dataset list
        balanced_dfs.append(oversampled_subset)
        
    # Concatenate the balanced datasets into a single dataframe
    balanced_train_test = pd.concat(balanced_dfs).reset_index()

   

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


    if validation == True:
        valid_data = dataset(df_validation,f'{val_path}',image_transform = valid_transform,image_filter=image_filter)
        valid_dataloader = DataLoader(valid_data,batch_size =8,shuffle=False) #validate model with 2500 eye images (500 for each class)
        test_dataloader,train_dataloader = 0,0
    else:
        train_set,valid_set = train_test_split( data_train_Test,test_size=0.2,random_state=42)
        data_train_Test = dataset(balanced_train_test,f'{train_test_path}',image_transform = train_test_transform,image_filter=image_filter,model = model)
        train_dataloader = DataLoader(train_set,batch_size=batch_size,shuffle=True) #DataLoader for train_set.

        test_dataloader = DataLoader(valid_set,batch_size=batch_size,shuffle=False) #DataLoader for test_set.
        valid_dataloader = 0

    return train_dataloader,test_dataloader,valid_dataloader


def optimize(model,model_name,train,test,device,data_label,path,path_for_val,tt_samp_size,batch_size,image_filter,lr,Epoch):

        model.to(device)
        train_loader,test_loader,valid_loader = get_data(data_label,path,path_for_val,tt_samp_size = 20,
                                                   batch_size=batch_size,image_filter=image_filter , model = model_name)  
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = torch.nn.CrossEntropyLoss() 


        for epoch in range(Epoch):
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
            valid_loss, valid_acc = test(test_loader, model, loss_fn, device=device)
            wandb.log({"train_acc": train_acc, "train_loss": train_loss,"test_acc":valid_acc,"test_loss":valid_loss})      
        
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
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curve for each class
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