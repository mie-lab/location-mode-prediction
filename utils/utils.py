from msilib.schema import Error
import yaml
import random, torch, os
import numpy as np
import pandas as pd

import json

from datetime import datetime

from utils.train import trainNet, test, get_performance_dict
from utils.dataloader import gc_dataset, collate_fn, deepmove_collate_fn
from models.model import TransEncoder
from models.RNNs import RNNs
from models.Deepmove import Deepmove


def load_config(path):
    """
    Loads config file:
    Args:
        path (str): path to the config file
    Returns:
        config (dict): dictionary of the configuration parameters, merge sub_dicts
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    config = dict()
    for key, value in cfg.items():
        for k, v in value.items():
            config[k] = v

    return config


def setup_seed(seed):
    """
    fix random seed for deterministic training
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_trainedNets(config, model, train_loader, val_loader, device):

    # define the path to save, and save the configuration
    if config.networkName == "rnn" and config.attention:
        networkName = f"{config.dataset}_{config.networkName}_Attn"
    else:
        networkName = f"{config.dataset}_{config.networkName}"
    log_dir = f"./outputs/{networkName}_{str(int(datetime.now().timestamp()))}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    with open(log_dir + "/conf.json", "w") as fp:
        json.dump(config, fp, indent=4, sort_keys=True)

    best_model, performance = trainNet(config, model, train_loader, val_loader, device, log_dir=log_dir)
    performance["type"] = "vali"

    return best_model, performance, log_dir


def get_test_result(config, best_model, test_loader, device):

    return_dict, result_arr_user = test(config, best_model, test_loader, device)
    performance = get_performance_dict(return_dict)
    performance["type"] = "test"

    result_user_df = pd.DataFrame(result_arr_user).T
    result_user_df.columns = [
        "correct@1",
        "correct@3",
        "correct@5",
        "correct@10",
        "f1",
        "rr",
        "total",
    ]
    result_user_df.index.name = "user"

    return performance, result_user_df


def get_models(config, device):
    total_params = 0

    if config.networkName == "deepmove":
        model = Deepmove(config=config).to(device)
    elif config.networkName == "transformer":
        model = TransEncoder(config=config).to(device)
    elif config.networkName == "rnn":
        model = RNNs(config=config).to(device)
    else:
        raise Error
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total number of trainable parameters: ", total_params)

    return model


def get_dataloaders(config):

    kwds_train = {
        "shuffle": True,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }
    kwds_val = {
        "shuffle": False,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }
    kwds_test = {
        "shuffle": True,
        "num_workers": 0,
        "batch_size": config["batch_size"],
    }

    dataset_train = gc_dataset(
        config.source_root,
        data_type="train",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
    )
    dataset_val = gc_dataset(
        config.source_root,
        data_type="validation",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
    )
    dataset_test = gc_dataset(
        config.source_root,
        data_type="test",
        model_type=config.networkName,
        previous_day=config.previous_day,
        dataset=config.dataset,
    )

    if config.networkName == "deepmove":
        fn = deepmove_collate_fn
    else:
        fn = collate_fn

    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=fn, **kwds_train)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=fn, **kwds_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=fn, **kwds_test)

    print(len(train_loader), len(val_loader), len(test_loader))
    return train_loader, val_loader, test_loader
