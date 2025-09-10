#Put all imports here
import os
import easy_lightning

from copy import deepcopy
# if you want to debug with the local code, change the name of the folder in easylightning2 from easy_lightning
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from easy_lightning import easy_exp, easy_rec, easy_torch

# otherwise decomment this line
# from easy_lightning import easy_exp, easy_rec, easy_torch

def main():
    #every path should start from the project folder:
    project_folder = "../"

    #Config folder should contain hyperparameters configurations
    cfg_folder = os.path.join(project_folder,"cfg/easy_rec_cfg")

    #Data folder should contain raw and preprocessed data
    data_folder = os.path.join(project_folder,"data")
    raw_data_folder = os.path.join(data_folder,"raw")


    # cfg loading
    cfg = easy_exp.cfg.load_configuration("config_rec", config_path=cfg_folder)
    cfg["data_params"]["data_folder"] = raw_data_folder


    # Example of a sweep configuration
    for _ in cfg.sweep('data_params.collator_params.lookback'):

        # check exp
        exp_found, experiment_id = easy_exp.exp.get_set_experiment_id(cfg)
        print("Experiment already found:", exp_found, "----> The experiment id is:", experiment_id)
        if exp_found: continue


        data, maps = easy_rec.preparation.prepare_rec_data(cfg)
        collator_params = deepcopy(cfg["data_params"]["collator_params"])
        loaders = easy_rec.preparation.prepare_rec_dataloaders(cfg, data, maps, collator_params=collator_params)

        # model loading
        main_module = easy_rec.preparation.prepare_rec_model(cfg, maps)

        # Prepare the trainer using the prepared trainer_params
        trainer = easy_torch.preparation.complete_prepare_trainer(cfg, experiment_id, additional_module=easy_rec)#, raytune=raytune)

        model = easy_torch.preparation.complete_prepare_model(cfg, main_module, easy_rec)

        tracker = None
        profiler = None
        if cfg["model"]["emission_tracker"]['use']:   
            tracker = easy_torch.preparation.prepare_emission_tracker(**cfg["model"]["emission_tracker"], experiment_id=experiment_id)
        if cfg["model"]["flops_profiler"]['use']:
            profiler =  easy_torch.preparation.prepare_flops_profiler(model, **cfg["model"]["flops_profiler"], experiment_id=experiment_id)

            
        easy_torch.process.train_model(trainer, model, loaders, val_key=["val","test"], tracker=tracker, profiler=profiler)
        
        easy_torch.process.test_model(trainer, model, loaders, test_key=["val","test","train"], tracker=tracker,  profiler=profiler)
        
        # Save experiment
        easy_exp.exp.save_experiment(cfg)
        
if __name__ == "__main__":
    main()
    print("\n\nQuick start for recommendation finished successfully!\n\n")