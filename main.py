import torch
import torch
import pandas as pd 
from models import select_model
from utils import optimize,preprocess_image,train,test
from paramtune import param_tuning
import wandb
from models import model_dict
import argparse
import json
import logging



def main(args):
    
    data_label = pd.read_csv(args.data_label_path)
    torch.cuda.empty_cache()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    param_tuning(data_label, args.train_test_path, args.validation_path, device) 

    with open(args.best_results_path, 'r') as file:
        best_results = json.load(file)

    for model_name in model_dict:
        model = select_model(model_name)
        wandb.init(project="Diabetic Retinopath Detection", name=model_name)
        
        optimize(model, model_name, train, test, device, data_label, args.train_test_path,
                 args.validation_path, tt_samp_size=args.tt_samp_size, batch_size=best_results[model_name]["config"]["batch_size"],
                 image_filter=preprocess_image, lr=best_results[model_name]["config"]["lr"], Epoch=args.epochs)
        
        torch.save(model.state_dict(), f"{args.saved_models_path}/{model_name}.pth")

        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_label_path", type=str, default="C:\\Users\\PC\\Desktop\\diabetic retinopathy\\data\\labels\\label.csv")
    parser.add_argument("--train_test_path", type=str, default="C:\\Users\\PC\\Desktop\\diabetic retinopathy\\data\\test_train_images")
    parser.add_argument("--validation_path", type=str, default="C:\\Users\\PC\\Desktop\\diabetic retinopathy\\data\\validation_images")
    parser.add_argument("--best_results_path", type=str, default="C:\\Users\\PC\Desktop\\diabetic retinopathy\\data\\best results")
    parser.add_argument("--saved_models_path", type=str, default="C:\\Users\\PC\\Desktop\\diabetic retinopathy\\data\\saved models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tt_samp_size",type = int,default =10,help="how many train data you want to put ")
    args = parser.parse_args()
    main(args)



