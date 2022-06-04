import sys, os
import pandas as pd
import numpy as np

import torch
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import f1_score

import time

from transformers import get_linear_schedule_with_warmup

from utils.earlystopping import EarlyStopping
from utils.dataloader import load_pk_file
from utils.train import get_performance_dict, send_to_device, calculate_correct_total_prediction, get_optimizer

def trainNet_mode(config, model, train_loader, val_loader, device, log_dir):

    performance = {}

    optim = get_optimizer(config, model)

    # define learning rate schedule
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=len(train_loader) * config.num_warmup_epochs,
        num_training_steps=len(train_loader) * config.num_training_epochs,
    )
    scheduler_ES = StepLR(optim, step_size=config.lr_step_size, gamma=config.lr_gamma)
    if config.verbose:
        print("Current learning rate: ", scheduler.get_last_lr()[0])

    # Time for printing
    training_start_time = time.time()
    globaliter = 0
    scheduler_count = 0

    # initialize the early_stopping object
    early_stopping = EarlyStopping(log_dir, patience=config["patience"], verbose=config.verbose)

    # Loop for n_epochs
    for epoch in range(config.max_epoch):
        # train for one epoch
        globaliter = train(
            config,
            model,
            train_loader,
            optim,
            device,
            epoch,
            scheduler,
            scheduler_count,
            globaliter
        )

        # At the end of the epoch, do a pass on the validation set
        return_dict = validate(config, model, val_loader, device)

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(return_dict, model)

        if early_stopping.early_stop:
            if config.verbose:
                print("=" * 50)
                print("Early stopping")
            if scheduler_count == 2:
                performance = get_performance_dict(early_stopping.best_return_dict)
                print(
                    "Training finished.\t Time: {:.2f}min.\t acc@1: {:.2f}%".format(
                        (time.time() - training_start_time) / 60,
                        performance["acc@1"],
                    )
                )

                break

            scheduler_count += 1
            model.load_state_dict(torch.load(log_dir + "/checkpoint.pt"))
            early_stopping.early_stop = False
            early_stopping.counter = 0
            scheduler_ES.step()

        if config.verbose:
            # print("Current learning rate: {:.5f}".format(scheduler.get_last_lr()[0]))
            # print("Current learning rate: {:.5f}".format(scheduler_ES.get_last_lr()[0]))
            print("Current learning rate: {:.5f}".format(optim.param_groups[0]["lr"]))
            print("=" * 50)

        if config.debug == True:
            break

    return model, performance


def train(
    config,
    model,
    train_loader,
    optim,
    device,
    epoch,
    scheduler,
    scheduler_count,
    globaliter
):

    model.train()

    running_loss = 0.0
    # 1, 3, 5, 10, rr, total
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    n_batches = len(train_loader)

    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
        
    # define start time
    start_time = time.time()
    optim.zero_grad()
    for i, inputs in enumerate(train_loader):
        globaliter += 1

        x, y, x_dict, y_mode = send_to_device(inputs, device, config)

        if config.if_loss_loc:
            logits_mode, logits_loc = model(x, x_dict, device)

            loss_size_loc = CEL(logits_loc, y.reshape(-1))
            loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))
            loss_size = loss_size_loc + loss_size_mode
        else:
            logits_mode = model(x, x_dict, device)
            loss_size = CEL(logits_mode, y_mode.reshape(-1))

        # optimize
        optim.zero_grad()
        loss_size.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
        optim.step()
        if scheduler_count == 0:
            scheduler.step()

        # Print statistics
        running_loss += loss_size.item()

        result_arr += calculate_correct_total_prediction(logits_mode, y_mode)

        if (config.verbose) and ((i + 1) % config["print_step"] == 0):
            print(
                "Epoch {}, {:.1f}%\t loss: {:.3f} acc@1: {:.2f} f1: {:.2f} mrr: {:.2f}, took: {:.2f}s \r".format(
                    epoch + 1,
                    100 * (i + 1) / n_batches,
                    running_loss / config["print_step"],
                    100 * result_arr[0] / result_arr[-1],
                    100 * result_arr[4] / config["print_step"],
                    100 * result_arr[5] / result_arr[-1],
                    time.time() - start_time,
                ),
                end="",
                flush=True,
            )

            # Reset running loss and time
            running_loss = 0.0
            result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
            start_time = time.time()

        if (config["debug"] == True) and (i > 20):
            break
    if config.verbose:
        print()
    return globaliter


def validate(config, model, data_loader, device):

    total_val_loss = 0
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    CEL = torch.nn.CrossEntropyLoss(reduction="mean", ignore_index=0)
    MSE = torch.nn.MSELoss(reduction="mean")
        

    # change to validation mode
    model.eval()
    with torch.no_grad():
        for inputs in data_loader:

            x, y, x_dict, y_mode = send_to_device(inputs, device, config)

            if config.if_loss_loc:
                logits_mode, logits_loc = model(x, x_dict, device)

                loss_size_loc = CEL(logits_loc, y.reshape(-1))
                loss_size_mode = CEL(logits_mode, y_mode.reshape(-1))
                loss_size = loss_size_loc + loss_size_mode
            else:
                logits_mode = model(x, x_dict, device)
                loss_size = CEL(logits_mode, y_mode.reshape(-1))

            total_val_loss += loss_size.item()

            result_arr += calculate_correct_total_prediction(logits_mode, y_mode.view(-1))

    val_loss = total_val_loss / len(data_loader)
    result_arr[4] = result_arr[4] / len(data_loader)

    if config.verbose:
        print(
            "Validation loss = {:.2f} acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                val_loss,
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return {
        "val_loss": val_loss,
        "correct@1": result_arr[0],
        "correct@3": result_arr[1],
        "correct@5": result_arr[2],
        "correct@10": result_arr[3],
        "f1": result_arr[4],
        "rr": result_arr[5],
        "total": result_arr[6],
    }


def test_mode(config, model, data_loader, device):

    # overall accuracy
    result_arr = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)

    # per user accuracy
    result_arr_user = {}
    count_user = {}
    for i in range(1, config.total_user_num):
        result_arr_user[i] = np.array([0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        count_user[i] = 0

    # change to validation mode
    model.eval()
    with torch.no_grad():

        for inputs in data_loader:
            x, _, x_dict, y_mode = send_to_device(inputs, device, config)

            if config.if_loss_loc:
                logits_mode, _ = model(x, x_dict, device)
            else:
                logits_mode = model(x, x_dict, device)
            
            
            # we get the per user accuracy
            user_arr = x_dict["user"].cpu().detach().numpy()
            unique = np.unique(user_arr)
            for user in unique:
                index = np.nonzero(user_arr == user)[0]

                result_arr_user[user] += calculate_correct_total_prediction(logits_mode[index, :], y_mode[index])
                count_user[user] += 1

            # overall accuracy
            result_arr += calculate_correct_total_prediction(logits_mode, y_mode.view(-1))

    # f1 score
    for i in range(1, config.total_user_num):
        result_arr_user[i][4] = result_arr_user[i][4] / count_user[i]
    result_arr[4] = result_arr[4] / len(data_loader)

    
    if config.verbose:
        print(
            "acc@1 = {:.2f} f1 = {:.2f} mrr = {:.2f}".format(
                100 * result_arr[0] / result_arr[-1],
                100 * result_arr[4],
                100 * result_arr[5] / result_arr[-1],
            ),
        )

    return (
        {
            "correct@1": result_arr[0],
            "correct@3": result_arr[1],
            "correct@5": result_arr[2],
            "correct@10": result_arr[3],
            "f1": result_arr[4],
            "rr": result_arr[5],
            "total": result_arr[6],
        },
        result_arr_user,
    )
