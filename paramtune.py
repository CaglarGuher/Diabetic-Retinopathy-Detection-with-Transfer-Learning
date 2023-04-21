from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune
import torch
import pandas as pd
from models import select_model
from Data_Prep import data_adjust
from utils import get_data,preprocess_image
from train_test_eval import train, test
from models import model_dict
from ray.tune.search.optuna import OptunaSearch
from ray.air import session
import ray
import json



def param_tuning(data_label,path,path_for_val,device):




    best_results = {}



    for model_name in model_dict:

        print(f"currently {model_name} is working")

        model = select_model(model_name)
        model.to(device)

        model_ref = ray.put(model)  # Store the model in the Ray object store

        def train_test_loop(config):

            model = ray.get(model_ref)  # Get the model from the Ray object store

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            loss_fn = torch.nn.CrossEntropyLoss()

            dataset = data_adjust(data_label,model,path)
            train_data, test_data, valid_data = get_data(
                data_label,
                dataset,
                path,
                path_for_val,
                train_test_sample_size=100,
                batch_size=config["batch_size"],
                image_filter=preprocess_image,
                model=model_name,
                validation = False
            )

            train_loader = train_data
            valid_loader = test_data
            for epoch in range(10):
                train_loss, train_acc = train(train_loader, model, loss_fn, optimizer, device=device)
                valid_loss, valid_acc = test(valid_loader, model, loss_fn, device=device)
                session.report({"loss": valid_loss})

            return {"loss": valid_loss}

        def run_search():

            # Specify the hyperparameter search space.
            config = {
                "lr": tune.choice([1e-4, 5e-4, 5e-3, 1e-3]),
                "batch_size": tune.choice([2, 4, 8, 16]),
            }

            # Set up the scheduler to early stop bad trials.
            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=2,
                grace_period=1,
                reduction_factor=2,
            )

            # Set up the progress reporter to display results during the search.
            reporter = CLIReporter(
                metric_columns=["loss", "training_iteration"],
                parameter_columns=["batch_size", "lr"],
            )

            # Set up the hyperparameter search using Optuna.
            result = tune.run(
                train_test_loop,
                config=config,
                num_samples=1,
                progress_reporter=reporter,
                resources_per_trial={"cpu": 0, "gpu": 1},
                checkpoint_at_end=True,
                metric="loss",
                mode="min",
                search_alg=OptunaSearch(),
                verbose= 1
            )


            # Print the best hyperparameters and validation loss.
            best_trial = result.get_best_trial("loss", "min", "last")
            best_results[model_name] = {
                "config": best_trial.config,
                "validation_loss": best_trial.last_result["loss"],
            }
            print(f"Best trial config: {best_trial.config}")
            print(f"Best trial final validation loss: {best_trial.last_result['loss']}")

        run_search()

    with open('best_results.json', 'w') as file:
        json.dump(best_results, file)  