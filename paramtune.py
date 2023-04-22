import logging
import json
import ray
import torch
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from ray.tune.suggest.optuna import OptunaSearch
from ray.tune.progress_reporter import CLIReporter
from data_prep import get_data
from models import model_dict,select_model
from utils import train,test,preprocess_image
from ray.air import session





logging.getLogger("ray.tune").setLevel(logging.WARNING)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] - %(message)s",
    handlers=[
        logging.FileHandler("param_tuning.log"),
        logging.StreamHandler()
    ]
)

def param_tuning(data_label, path, path_for_val, device):
    logging.info("Starting hyperparameter tuning")

    best_results = {}

    for model_name in model_dict:

        logging.info(f"Currently working on {model_name}")

        model = select_model(model_name)
        model.to(device)

        model_ray = ray.put(model) 

        def train_test_loop(config):
            
            model = ray.get(model_ray) 

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            loss_fn = torch.nn.CrossEntropyLoss()

            train_data, test_data, _ = get_data(
                data_label,
                path,
                path_for_val,
                train_test_sample_size=100,
                batch_size=config["batch_size"],
                image_filter=preprocess_image,
                model=model_name,
                validation=False
            )

            train_loader = train_data
            valid_loader = test_data
            
            for epoch in range(10):
                _, _ = train(train_loader, model, loss_fn, optimizer, device=device)
                valid_loss, _ = test(valid_loader, model, loss_fn, device=device)
                session.report({"loss": valid_loss})

            return {"loss": valid_loss}

        def run_search():
            config = {
                "lr": tune.choice([1e-4, 5e-4, 5e-3, 1e-3]),
                "batch_size": tune.choice([2, 4, 8, 16]),
            }

            scheduler = ASHAScheduler(
                metric="loss",
                mode="min",
                max_t=2,
                grace_period=1,
                reduction_factor=2,
            )

            reporter = CLIReporter(
                metric_columns=["loss", "training_iteration"],
                parameter_columns=["batch_size", "lr"],
            )

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
                verbose=1
            )

            best_trial = result.get_best_trial("loss", "min", "last")
            best_results[model_name] = {
                "config": best_trial.config,
                "validation_loss": best_trial.last_result["loss"],
            }
            logging.info(f"Best trial config: {best_trial.config}")
            logging.info(f"Best trial final validation loss: {best_trial.last_result['loss']}")

        run_search()

    logging.info("Hyperparameter tuning completed")

    with open('best_results.json', 'w') as file:
        json.dump(best_results, file)
        logging.info("Best results saved to best_results.json")