import wandb
from copy import deepcopy

def wandb_wrapper(cfg, fun, *args, **kwargs):
    wandb.init()
    sweep_cfg = wandb.config
    new_cfg = deepcopy(cfg)
    new_cfg.update(sweep_cfg)
    
    fun(new_cfg)

def wandb_sweep(cfg, fun, sweep_dict=None, adjust_run_cap = True, run_cap_for_distr=10):
    wandb.login()

    if sweep_dict is None:
        sweep_dict = cfg["__exp__"]["__sweep__"]
    if adjust_run_cap:
        sweep_dict["run_cap"] = count_run_cap(sweep_dict, run_cap_for_distr)
    sweep_dict["parameters"] = {k1:{k2:v2 for k2,v2 in v1.items() if k2 != "default"} for k1,v1 in sweep_dict["parameters"].items()}

    wandb_dict = cfg["__exp__"]["__wandb__"]

    sweep_id = wandb.sweep(sweep_dict, **wandb_dict)

    wandb.agent(sweep_id, function = lambda *args,**kwargs : wandb_wrapper(cfg,fun,*args,**kwargs))

def count_run_cap(sweep_dict, run_cap_for_distr=10):
    total_run_cap = 1
    for _, param_info in sweep_dict["parameters"].items():
        total_run_cap *= len(param_info.get('values',run_cap_for_distr))
    return total_run_cap