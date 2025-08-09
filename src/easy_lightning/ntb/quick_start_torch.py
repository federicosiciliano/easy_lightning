import os
import easy_lightning

from copy import deepcopy
# if you want to debug with the local code, change the name of the folder in easylightning2 from easy_lightning
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from easy_lightning import easy_exp, easy_rec, easy_torch, easy_data

# otherwise decomment this line
# from easy_lightning import easy_exp, easy_rec, easy_torch


def main():
    project_folder = "../"
    sys.path.insert(0, project_folder)
    cfg_folder = os.path.join(project_folder,"cfg/easy_torch_cfg")


    cfg = easy_exp.cfg.load_configuration("config_nn", config_path=cfg_folder)


    for _ in cfg.sweep('model.trainer_params.max_epochs'):

        exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
        print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)
        if exp_found: continue

        data, _ = easy_data.data.load_data(**cfg["data_params"])


        loaders = easy_torch.preparation.prepare_data_loaders(data, **cfg["model"]["loader_params"])


        cfg["model"]["in_channels"] = data["train_x"].shape[1]
        cfg["model"]["out_features"] = data["train_y"].shape[1]

        main_module = easy_torch.model.get_torchvision_model(**cfg["model"])


        # Set experiment_id in trainer_params
        trainer_params = easy_torch.preparation.prepare_experiment_id(cfg["model"]["trainer_params"], experiment_id)


        trainer_params["callbacks"] = easy_torch.preparation.prepare_callbacks(trainer_params)
        trainer_params["logger"] = easy_torch.preparation.prepare_logger(trainer_params)
        trainer = easy_torch.preparation.prepare_trainer(**trainer_params)


        loss = easy_torch.preparation.prepare_loss(cfg["model"]["loss"])

        optimizer = easy_torch.preparation.prepare_optimizer(**cfg["model"]["optimizer"])

        model = easy_torch.process.create_model(main_module, loss=loss, optimizer=optimizer)

        tracker = None
        profiler = None
        if cfg["model"]["emission_tracker"]['use']:   
            tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)
        if cfg["model"]["flops_profiler"]['use']:
            profiler =  easy_torch.preparation.prepare_flops_profiler(model, **cfg["model"]["flops_profiler"], experiment_id=experiment_id)


        easy_torch.process.train_model(trainer, model, loaders, tracker=tracker, profiler=profiler)


        easy_torch.process.test_model(trainer, model, loaders, tracker=tracker, profiler=profiler)

        easy_exp.exp.save_experiment(cfg)


if __name__ == "__main__":
    main()
    print("\n\nQuick start for torch finished successfully!\n\n")
