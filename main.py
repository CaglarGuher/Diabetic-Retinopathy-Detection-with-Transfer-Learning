import torch
import torch
import pandas as pd 
from models import select_model
from utils import optimize,preprocess_image,train,test
from paramtune import param_tuning
import wandb
from models import model_dict

import json
from collections import Counter




data_label = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
path = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
path_for_val = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Epoch = 15


param_tuning(data_label,path,path_for_val,device)

with open('best_results.json', 'r') as file:
    best_results = json.load(file)




for model_name in model_dict:

    model = select_model(model_name)
    wandb.init(
        # set the wandb project where this run will be logged
        project="Diabetic Retinopath Detection",
        name = model_name
        # track hyperparameters and run metadata
    )
    
    optimize(model,model_name,train,test,device,data_label,path,
             path_for_val,tt_samp_size = 100,batch_size=best_results[model_name]["config"]["batch_size"],
             image_filter = preprocess_image,lr=best_results[model_name]["config"]["lr"],Epoch = Epoch)
    torch.save(model.state_dict(), f"C:/Users/PC/Desktop/saved_models/{model_name}.pth")
    wandb.finish()





