#Put all imports here
import numpy as np
import os
import sys
import easy_lightning
from easy_lightning import easy_exp, easy_rec, easy_torch

print("easy_rec path:", easy_rec.__path__)



#every path should start from the project folder:
project_folder = "../"

#Config folder should contain hyperparameters configurations
cfg_folder = os.path.join(project_folder,"cfg/easy_rec_cfg")

#Data folder should contain raw and preprocessed data
data_folder = os.path.join(project_folder,"data")
raw_data_folder = os.path.join(data_folder,"raw")


# data loading
cfg = easy_exp.cfg.load_configuration("config_rec", config_path=cfg_folder)
cfg["data_params"]["data_folder"] = raw_data_folder

# Example of a sweep configuration
for _ in cfg.sweep(cfg["data_params"]["dataset_params"]["lookback"]):

    data, maps = easy_rec.data_generation_utils.preprocess_dataset(**cfg["data_params"])
    datasets = easy_rec.rec_torch.prepare_rec_datasets(data,**cfg["data_params"]["dataset_params"])


    # model loading
    cfg["model"]["loader_params"]["num_items"] = np.max(list(maps["sid"].values()))
    loaders = easy_rec.rec_torch.prepare_rec_data_loaders(datasets, data, **cfg["model"]["loader_params"])
    cfg["model"]["rec_model"]["num_items"] = np.max(list(maps["sid"].values()))
    cfg["model"]["rec_model"]["num_users"] = np.max(list(maps["uid"].values()))
    cfg["model"]["rec_model"]["lookback"] = cfg["data_params"]["dataset_params"]["lookback"]
    main_module = easy_rec.rec_torch.create_rec_model(**cfg["model"]["rec_model"])

    # check exp
    exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
    print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)


    if exp_found: exit() #TODO: make the notebook/script stop here if the experiment is already found


    trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)

    # Prepare callbacks and logger using the prepared trainer_params
    trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
    trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)

    # Prepare the trainer using the prepared trainer_params
    trainer = easy_torch.preparation.prepare_trainer(**trainer_params)

    model_params = cfg["model"].copy()

    model_params["loss"] = easy_torch.preparation.prepare_loss(cfg["model"]["loss"], easy_rec.losses)

    # Prepare the optimizer using configuration from cfg
    model_params["optimizer"] = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

    # Prepare the metrics using configuration from cfg
    model_params["metrics"] = easy_torch.preparation.prepare_metrics(cfg["model"]["metrics"], easy_rec.metrics)

    # Create the model using main_module, loss, and optimizer
    model = easy_torch.process.create_model(main_module, **model_params)


    # Prepare the emission tracker using configuration from cfg
    tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)


    # Train the model using the prepared trainer, model, and data loaders
    easy_torch.process.train_model(trainer, model, loaders, tracker=tracker, val_key=["val","test"])


    easy_torch.process.test_model(trainer, model, loaders, tracker=tracker)


    # Save experiment and print the current configuration
    #save_experiment_and_print_config(cfg)
    easy_exp.exp.save_experiment(cfg)

    # Print completion message
    print("Execution completed.")
    print("######################################################################")
    print()

