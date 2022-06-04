import torch, os
import numpy as np
import pandas as pd

from easydict import EasyDict as edict

from utils.utils import (
    load_config,
    setup_seed,
    get_trainedNets,
    get_test_result,
    get_dataloaders,
    get_models,
)

setup_seed(41)


def single_run(config, device):
    result_ls = []

    for _ in range(1):
        print(config.if_embed_mode, config.if_loss_mode)
        # get data
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # get modelp
        model = get_models(config, device)

        # train
        model, perf, log_dir = get_trainedNets(config, model, train_loader, val_loader, device)
        print(perf)
        result_ls.append(perf)

        # test
        perf, test_df = get_test_result(config, model, test_loader, device)
        test_df.to_csv(log_dir + "/user_mode_detail.csv") 

        print(perf)
        result_ls.append(perf)

    return result_ls

if __name__ == "__main__":
    # deepmove
    # config_path = "./config/gc/deepmove.yml"
    # config_path = "./config/yumuv/deepmove.yml"

    # config_path = "./config/gc/lstm.yml"
    # config_path = "./config/yumuv/lstm.yml"

    # config_path = "./config/gc/lstm_attn.yml"
    # config_path = "./config/yumuv/lstm_attn.yml"

    config_path = "./config/gc/transformer.yml"
    # config_path = "./config/yumuv/transformer.yml"

    # config_path = "./config/gc/MobTCast.yml"
    # config_path = "./config/yumuv/MobTCast.yml"

    # config_path = "./config/gc/transformer_mode.yml"
    # config_path = "./config/yumuv/transformer_mode.yml"

    config = load_config(config_path)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    result_ls = []


    result_ls.extend(single_run(config, device))


    result_df = pd.DataFrame(result_ls)
    print(result_df)

    filename = f"./outputs/{config.networkName}.csv"
    result_df.to_csv(filename, index=False)
