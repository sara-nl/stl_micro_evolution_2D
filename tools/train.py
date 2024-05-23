# Copyright (c) CAIRI AI Lab. All rights reserved

import os.path as osp
import warnings
import yaml
import wandb

warnings.filterwarnings("ignore")

from openstl.api import BaseExperiment
from openstl.utils import (
    create_parser,
    default_parser,
    get_dist_info,
    load_config,
    setup_multi_processes,
    update_config,
)


def load_yaml_config(yaml_file, yaml_name="default"):
    with open(yaml_file, "r") as file:
        all_configs = yaml.safe_load(file)
    return all_configs.get(yaml_name, all_configs["default"])


try:
    import nni

    has_nni = True
except ImportError:
    has_nni = False


if __name__ == "__main__":
    args = create_parser().parse_args()
    config = args.__dict__

    print(args.method)

    if has_nni:
        tuner_params = nni.get_next_parameter()
        config.update(tuner_params)

    cfg_path = (
        osp.join("../configs", args.dataname, f"{args.method}.py")
        if args.config_file is None
        else args.config_file
    )
    if args.overwrite:
        config = update_config(config, load_config(cfg_path), exclude_keys=["method"])
    else:
        loaded_cfg = load_config(cfg_path)
        config = update_config(
            config,
            loaded_cfg,
            exclude_keys=[
                # "method",
                # "batch_size",
                # "val_batch_size",
                "drop_path",
                "warmup_epoch",
            ],
        )
        print(args.method)
        # update values from experiment yaml:
        yfile = "../configs/kmc/experiments.yaml"  # config["yaml_file"]
        yaml_name = args.ex_name if args.ex_name else "default"
        yaml_config = load_yaml_config(yfile, yaml_name)

        config = update_config(
            config,
            yaml_config,
        )

        # load default values
        default_values = default_parser()
        for attribute in default_values.keys():
            if config[attribute] is None:
                config[attribute] = default_values[attribute]

    # set multi-process settings
    setup_multi_processes(config)

    wandb.init(
        project=f"Diagnostic",  # {config['method']} Reg_VTUNet openSTL_PF Diagnostic
        entity="mrotulo",
        sync_tensorboard=True,
    )

    print(">" * 35 + " training " + "<" * 35)

    exp = BaseExperiment(args)
    rank, _ = get_dist_info()
    exp.train()

    if rank == 0:
        print(">" * 35 + " testing  " + "<" * 35)
    mse = exp.test()

    if rank == 0 and has_nni:
        nni.report_final_result(mse)
