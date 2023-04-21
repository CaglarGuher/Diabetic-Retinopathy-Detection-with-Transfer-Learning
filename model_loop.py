import torch
import torch
import pandas as pd 
from models import select_model
from get_data import get_data
from filters import preprocess_image
import wandb
from models import model_dict
from train_test_eval import train , test , eval

data_label = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
path = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
path_for_validation = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



for model_name in model_dict:

    model = select_model(model_name)
    wandb.init(
        # set the wandb project where this run will be logged
        project="my-awesome-project",
        name = model_name
        # track hyperparameters and run metadata
    )
    
    def optimize():

        model.to(device)
        train_loader,test_loader,valid_loader = get_data(data_label,path,path_for_validation,train_test_sample_size = 20,
                                                   batch_size=16,image_filter=preprocess_image , model = model_name)  
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = torch.nn.CrossEntropyLoss() 


        for epoch in range(15):
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
            valid_loss, valid_acc = test(test_loader, model, loss_fn, device=device)
            wandb.log({"train_acc": train_acc, "train_loss": train_loss,"test_acc":valid_acc,"test_loss":valid_loss})      
        
        return {"loss": valid_loss}


    optimize()
    torch.save(model.state_dict(), f"C:/Users/PC/Desktop/saved_models/{model_name}.pth")
    wandb.finish()

    # Set model to evaluation mode


