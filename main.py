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

    # if_embed_mode_ls = [True, False]
    # if_loss_mode_ls = [True, False]
    
    # for if_embed_mode in if_embed_mode_ls:
    #     config.if_embed_mode = if_embed_mode
    #     for if_loss_mode in if_loss_mode_ls:
    #         config.if_loss_mode = if_loss_mode
            
    result_ls = []
    for _ in range(5):

        # get data
        train_loader, val_loader, test_loader = get_dataloaders(config)

        # get modelp
        model = get_models(config, device)

        # train
        best_model, perf, log_dir = get_trainedNets(config, model, train_loader, val_loader, device)

        print(perf)
        result_ls.append(perf)

        # test
        perf, result_user_df = get_test_result(config, model, test_loader, device)

        print(perf)
        result_user_df.to_csv(log_dir + "/test_user.csv")
        result_ls.append(perf)

    result_df = pd.DataFrame(result_ls)
    print(result_df)
    
    filename = f"./outputs/{config.networkName}"
    # if if_loss_mode:
    #     filename = filename + "_loss"
    # if if_embed_mode:
    #     filename = filename + "_feature"
        
    filename = filename + f"_len{config.predict_length}.csv" 
    result_df.to_csv(filename, index=False)
