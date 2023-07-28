import logging
import json
import torch
from data_prep import get_data
from models import model_dict,select_model
from utils import train,test,preprocess_image
import optuna



logging.getLogger("optuna").setLevel(logging.WARNING)

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

        def train_test_loop(trial):
            config = {
                "lr": trial.suggest_categorical("lr", [1e-4, 5e-4, 5e-3, 1e-3]),
                "batch_size": trial.suggest_categorical("batch_size", [16]),
            }

            model = select_model(model_name)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
            loss_fn = torch.nn.CrossEntropyLoss()

            train_data, test_data, _ = get_data(
                data_label,
                path,
                path_for_val,
                train_test_sample_size=25,
                batch_size=config["batch_size"],
                image_filter=preprocess_image,
                model=model_name,
                validation=False
            )

            train_loader = train_data
            valid_loader = test_data

            for epoch in range(2):
                _, _ = train(train_loader, model, loss_fn, optimizer, device=device)
                valid_loss, _ = test(valid_loader, model, loss_fn, device=device)
                trial.report(valid_loss, epoch)

                if trial.should_prune():
                    raise optuna.TrialPruned()

            return valid_loss

        study = optuna.create_study(direction="minimize")
        study.optimize(train_test_loop, n_trials=1)  # You can adjust the number of trials as needed.

        best_trial = study.best_trial
        best_results[model_name] = {
            "config": best_trial.params,
            "validation_loss": best_trial.value,
        }
        logging.info(f"Best trial config: {best_trial.params}")
        logging.info(f"Best trial final validation loss: {best_trial.value}")

    logging.info("Hyperparameter tuning completed")

    with open('best_results.json', 'w') as file:
        json.dump(best_results, file)
        logging.info("Best results saved to best_results.json")