from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from ray import tune

def run_search(train_test_loop):


    config = { 
        "lr":tune.loguniform(1e-4,5e-3),
        "beta1":tune.loguniform(0.8,1),
        "beta2":tune.loguniform(0.95,1),
        "epoch":tune.choice([2, 4, 8, 16])
    }
    # Schduler to stop bad performing trails.
    scheduler = ASHAScheduler(
        metric = "loss",
        mode = "min",
        max_t=3,
        grace_period=2,
        reduction_factor=2)
    
   
    reporter = CLIReporter(
        metric_columns=["loss", "training_iteration"])
    result = tune.run(
        train_test_loop,
        resources_per_trial={"cpu": 0, "gpu":1},
        config=config,
        num_samples=30,
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
    
