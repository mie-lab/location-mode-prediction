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
    # config_path = "./config/geolife/deepmove.yml"
    # config_path = "./config/gc/deepmove.yml"

    # config_path = "./config/geolife/lstm.yml"
    # config_path = "./config/gc/lstm.yml"

    # config_path = "./config/geolife/lstm_attn.yml"
    # config_path = "./config/gc/lstm_attn.yml"

    config_path = "./config/gc/transformer.yml"

    config = load_config(config_path)
    config = edict(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")

    fc_dropout_ls = [0.1, 0.2]
    loc_emb_size_ls = [32, 64]
    # if_embed_mode_ls = [True, False]
    # if_loss_mode_ls = [True, False]

    result_ls = []
    for fc_dropout in fc_dropout_ls:
        config.fc_dropout = fc_dropout
        # for if_embed_mode in if_embed_mode_ls:
        #     config.if_embed_mode = if_embed_mode
        #     for if_loss_mode in if_loss_mode_ls:
        #         config.if_loss_mode = if_loss_mode
        for loc_emb_size in loc_emb_size_ls:
            config.loc_emb_size = loc_emb_size

            for _ in range(5):
                print(config.loc_emb_size,  config.if_loss_mode, config.if_embed_mode, config.fc_dropout)

                # get data
                train_loader, val_loader, test_loader = get_dataloaders(config)

                # get modelp
                model = get_models(config, device)

                # train
                model, perf, log_dir = get_trainedNets(config, model, train_loader, val_loader, device)
                print(perf)
                perf["loc_emb_size"] = config.loc_emb_size
                perf["if_loss_mode"] = config.if_loss_mode
                perf["if_embed_mode"] = config.if_embed_mode
                perf["fc_dropout"] = config.fc_dropout
                result_ls.append(perf)

                # test
                # model.load_state_dict(torch.load(r"D:\Code\NPP_mode\outputs\gc_transformer_1653486168\checkpoint.pt"))
                perf, result_user_df = get_test_result(config, model, test_loader, device)
                result_user_df.to_csv(log_dir + "/test_user.csv")

                print(perf)
                perf["loc_emb_size"] = config.loc_emb_size
                perf["if_loss_mode"] = config.if_loss_mode
                perf["if_embed_mode"] = config.if_embed_mode
                perf["fc_dropout"] = config.fc_dropout
                result_ls.append(perf)

    result_df = pd.DataFrame(result_ls)
    print(result_df)

    filename = f"./outputs/{config.networkName}_tune_layer4_True.csv"
    result_df.to_csv(filename, index=False)
