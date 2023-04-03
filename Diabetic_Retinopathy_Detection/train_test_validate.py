import torch
import torch
from torch import nn
import pandas as pd 
from ray import tune
import os
from Models import get_model
from get_data import get_data
from filters import preprocess_image
from paramtune import run_search
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter()
path = "data/Images"
data_label = pd.read_csv("dataset label.csv")
path_for_validation = "data/validation_images"

torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = get_model("convnext_tiny")

train_data,test_data,valid_data = get_data(data_label,path,path_for_validation,train_test_sample_size = 50,batch_size=16,image_filter=preprocess_image)



def train(dataloader, loss_fn, optimizer, device):
    
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


def validate(dataloader,  loss_fn, device):

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

    
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], betas=(config["beta1"], config["beta2"]))
    loss_fn = torch.nn.CrossEntropyLoss() # instantiate your loss function


    for epoch in range(config["epoch"]):
        train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
        valid_loss, valid_acc = validate(valid_loader, model, loss_fn, device=device)
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', valid_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', valid_acc, epoch)
    writer.close()
    return {"loss": valid_loss}


def train_test_loop(config,device):
    
    model.to(device)
    train_loader = train_data # instantiate your train data loader
    valid_loader = test_data # instantiate your validation data loader
    result = optimize(config, train_loader, valid_loader,model=model)
    return result
config = { 
        "lr":1e-3,
        "beta1":0.99,
        "beta2":0.999,
        "epoch":20
    }


train_test_loop(config=config,device=device)
