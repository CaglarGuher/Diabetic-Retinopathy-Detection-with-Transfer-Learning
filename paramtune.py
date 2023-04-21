from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import torch
import pandas as pd 
from models import select_model
from get_data import get_data
from filters import preprocess_image
from train_test_eval import train , test
from models import model_dict
from ray.tune.search.optuna import OptunaSearch
from ray.air import session

data_label = pd.read_csv("C:/Users/PC/Desktop/retinopathy_data/data/labels/label.csv")
path = "C:/Users/PC/Desktop/retinopathy_data/data/test_train_images"
path_for_validation = "C:/Users/PC/Desktop/retinopathy_data/data/validation_images"
torch.cuda.empty_cache()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




for model_name in model_dict:
    
    print(f"currently  {model_name} is working")
    
    model = select_model(model_name)
    model.to(device)   

    def train_test_loop(config):

        optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
        loss_fn = torch.nn.CrossEntropyLoss() 
        train_data,test_data,valid_data = get_data(data_label,path,path_for_validation,train_test_sample_size = 100,
                                                   batch_size=config["batch_size"],image_filter=preprocess_image , model = model_name)
        
        train_loader = train_data 
        valid_loader = test_data 
        for epoch in range(10):
            train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
            valid_loss, valid_acc = test(valid_loader, model, loss_fn, device=device)
            session.report({"loss": valid_loss })
            
        return {"loss": valid_loss}
    

    def run_search():

        # Specify the hyperparameter search space.
        config = {
            "lr": tune.loguniform(1e-5, 1e-3)
        }
        
        # Set up the scheduler to early stop bad trials.
        scheduler = ASHAScheduler(
            metric="loss",
            mode="min",
            max_t=10,
            grace_period=1,
            reduction_factor=2
        )
        
        # Set up the progress reporter to display results during the search.
        reporter = CLIReporter(
            metric_columns=["loss", "training_iteration"],
            parameter_columns=["batch_size", "lr"]
        )
        
        # Set up the hyperparameter search using Optuna.
        result = tune.run(
            train_test_loop,
            config=config,
            num_samples=50,
            progress_reporter=reporter,
            resources_per_trial={"cpu": 1, "gpu": 1},
            checkpoint_at_end=True,
            metric="loss",
            mode="min",
            search_alg=OptunaSearch()
        )
        
        # Print the best hyperparameters and validation loss.
        best_trial = result.get_best_trial("loss", "min", "last")
        print(f"Best trial config: {best_trial.config}")
        print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

    run_search()