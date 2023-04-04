import torch
import torch
import pandas as pd 
from Model import select_model
from get_data import get_data
from filters import preprocess_image
import numpy as np
from sklearn.metrics import confusion_matrix
import wandb





for model_name in ["resnet152","resnet101","vgg19","googlenet","efficient-netb5","efficient-netb6","efficient-netb7",
                   "wide_resnet50_2","resnext50_32x4d","shufflenet_v2_x1_0","mobilenet_v2","alexnet","densenet161"]:
    print(f"currently  {model_name} is working")
    
    model = select_model(model_name)
       

    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        name = model_name,
        # track hyperparameters and run metadata
        config={
        "learning_rate": 0.001,
        "architecture": "CNN",
        "dataset": "Eye Fundus",
        "epochs": 10,
        }
    )



    data_label = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
    path = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
    path_for_validation = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"


    torch.cuda.empty_cache()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_data,test_data,valid_data = get_data(data_label,path,path_for_validation,train_test_sample_size = 100,batch_size=16,image_filter=preprocess_image , model = model_name)
    


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


    def validate(dataloader,model,loss_fn, device):

        model.eval() 
        running_loss = 0
        correct = 0
        total = 0

        with torch.no_grad(): 
            
            for (x, y) in dataloader:
                x, y = x.to(device), y.to(device)  # Move data to GPU
                
                output        = model(x)
                loss          = loss_fn(output, y).item()
                running_loss += loss
                
                # Calculate accuracy
                _, predicted = torch.max(output.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * correct / total

        print('Validation Loss: {:.4f} | Validation Accuracy: {:.2f}%'.format(avg_loss, accuracy))

        return avg_loss, accuracy


    def optimize(config, train_loader, valid_loader,model):

        
        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(0.99, 0.999))
        loss_fn = torch.nn.CrossEntropyLoss() 


        for epoch in range(config["epoch"]):
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
            valid_loss, valid_acc = validate(valid_loader, model, loss_fn, device=device)
            wandb.log({"train_acc": train_acc, "train_loss": train_loss,"test_acc":valid_acc,"test_loss":valid_loss})      
        
        return {"loss": valid_loss}


    def train_test_loop(config,device):
        
        model.to(device)
        train_loader = train_data # instantiate your train data loader
        valid_loader = test_data # instantiate your validation data loader
        result = optimize(config, train_loader, valid_loader,model=model)
        return result
    config = { 
            "lr":1e-7,
            "beta1":0.99,
            "beta2":0.999,
            "epoch":20
        }


    train_test_loop(config=config,device=device)
    wandb.finish()

    # Set model to evaluation mode
    '''
    model.eval()

    # Create empty lists to store predictions and labels
    all_preds = []
    all_labels = []

    # Iterate over the validation set and make predictions

    with torch.no_grad():
        for images, labels in valid_data:
            images = images.to(device)
            labels = labels.to(device)
            preds = model(images)
            all_preds.extend(preds.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Create the confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Print the confusion matrix

    print(cm)

    accuracy = np.sum(np.diag(cm)) / np.sum(cm)

    print(f"accuracy of the model is  = {accuracy}")
    '''