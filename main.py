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

setup_seed(42)


if __name__ == "__main__":
    config_path = "./config/gc/deepmove.yml"
    # config_path = "./config/gc/lstm.yml"
    # config_path = "./config/gc/lstm_attn.yml"
    # config_path = "./config/gc/transformer.yml"

    config = load_config(config_path)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    # fc_dropout_ls = [0.1, 0.2]
    # if_embed_mode_ls = [True, False]
    # if_loss_mode_ls = [True, False]

    result_ls = []

    for _ in range(5):
        # print(config.fc_dropout, config.loc_emb_size, config.if_embed_mode, config.if_loss_mode)

        # get data
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # get modelp
        model = get_models(config, device)

        # train
        model, perf, log_dir = get_trainedNets(config, model, train_loader, val_loader, device)
        print(perf)
        # perf["fc_dropout"] = config.fc_dropout
        # perf["if_embed_mode"] = config.if_embed_mode
        # perf["if_loss_mode"] = config.if_loss_mode
        result_ls.append(perf)

        # test
        # model.load_state_dict(torch.load(r"D:\Code\NPP_mode\outputs\gc_transformer_1653486168\checkpoint.pt"))
        perf, result_user_df = get_test_result(config, model, test_loader, device)
        result_user_df.to_csv(log_dir + "/test_user.csv") 

        print(perf)
        # perf["fc_dropout"] = config.fc_dropout
        # perf["if_embed_mode"] = config.if_embed_mode
        # perf["if_loss_mode"] = config.if_loss_mode
        result_ls.append(perf)

    result_df = pd.DataFrame(result_ls)
    print(result_df)

    filename = f"./outputs/{config.networkName}.csv"
    result_df.to_csv(filename, index=False)
