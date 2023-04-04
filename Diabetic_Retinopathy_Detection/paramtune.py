from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import torch
import torch
import pandas as pd 
from Model import select_model
from get_data import get_data
from filters import preprocess_image
from ray import air


data_label = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
path = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
path_for_validation = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




for model_name in ["resnet152","resnet101","vgg19","googlenet","efficient-netb5","efficient-netb6","efficient-netb7",
                   "wide_resnet50_2","resnext50_32x4d","shufflenet_v2_x1_0","mobilenet_v2","alexnet","densenet161"]:
    
    print(f"currently  {model_name} is working")
    
    model = select_model(model_name)
    model.to(device)   

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

        return avg_loss, accuracy


    
    def train_test_loop(config):

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = torch.nn.CrossEntropyLoss() 
        train_data,test_data,valid_data = get_data(data_label,path,path_for_validation,train_test_sample_size = 100,batch_size=config["batch_size"],image_filter=preprocess_image , model = model_name)
        
        train_loader = train_data # instantiate your train data loader
        valid_loader = test_data # instantiate your validation data loader
        for epoch in range(10):
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
            valid_loss, valid_acc = validate(valid_loader, model, loss_fn, device=device)
            tune.report(loss=valid_loss, accuracy = valid_acc)
        return {"loss": valid_loss}



    def run_search():


        config = { 

            "batch_size":tune.choice([2,4,6, 8,10, 16]),
            "lr": tune.choice([1e-3,5e-3,1e-4,5e-4])
        }
        # Schduler to stop bad performing trails.
        scheduler = ASHAScheduler(
            metric = "loss",
            mode = "min",
            max_t=5,
            grace_period=1,
            reduction_factor=2)
        
    
        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"])
        result = tune.run(
            train_test_loop,
            resources_per_trial={"cpu": 12, "gpu":1},
            config=config,
            num_samples=24,
            scheduler=scheduler,
            keep_checkpoints_num=1,
            checkpoint_score_attr='loss',
            progress_reporter=reporter)
        
        
        # Extract the best trial run from the search.
        best_trial = result.get_best_trial(
            'loss', 'min', 'last'
        )
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")
        
    run_search()