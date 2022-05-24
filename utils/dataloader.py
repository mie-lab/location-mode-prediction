import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
from pathlib import Path
import pickle as pickle

from joblib import Parallel, delayed
from sklearn.preprocessing import OrdinalEncoder
import os
import torch
from torch.nn.utils.rnn import pad_sequence

import trackintel as ti


class gc_dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        source_root,
        dataset="gc",
        data_type="train",
        previous_day=7,
        model_type="transformer",
        predict_length=1,
    ):
        self.root = source_root
        self.data_type = data_type
        self.previous_day = previous_day
        self.model_type = model_type
        self.dataset = dataset
        self.predict_length = predict_length

        self.data_dir = os.path.join(source_root, "temp")
        save_path = os.path.join(
            self.data_dir,
            f"{self.model_type}_{previous_day}_{predict_length}_{data_type}.pk",
        )

        if Path(save_path).is_file():
            self.data = pickle.load(open(save_path, "rb"))
        else:
            self.data = self.generate_data()

        self.len = len(self.data)

    def __len__(self):
        """Return the length of the current dataloader."""
        return self.len

    def __getitem__(self, idx):
        selected = self.data[idx]

        x_dict = {}
        # [sequence_len]
        x = torch.tensor(selected["X"], dtype=torch.int64)
        # [1]
        x_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
        # [sequence_len] in half an hour
        x_dict["time"] = torch.tensor(selected["start_min_X"] // 30, dtype=torch.int64)
        # [sequence_len]
        x_dict["mode"] = torch.tensor(selected["mode_X"], dtype=torch.int64)
        # [sequence_len]
        x_dict["length"] = torch.tensor(np.log(selected["length_X"]))
        
        tgt_dict={}
        # [2+predict_length]
        padding = np.zeros(self.predict_length)
        tgt = torch.tensor(np.concatenate((selected["X"][-2:], padding)), dtype=torch.int64)
        # [1]
        tgt_dict["user"] = torch.tensor(selected["user_X"][0], dtype=torch.int64)
        # [2+predict_length] in half an hour
        tgt_dict["time"] = torch.tensor(np.concatenate(( selected["start_min_X"][-2:] // 30, padding)), dtype=torch.int64)
        # [2+predict_length]
        tgt_dict["mode"] = torch.tensor(np.concatenate((selected["mode_X"][-2:], padding)), dtype=torch.int64)
        # [2+predict_length]
        tgt_dict["length"] = torch.tensor(np.concatenate(( np.log(selected["length_X"][-2:]), padding)) )
        

        if self.model_type == "deepmove":
            # [1]
            x_dict["history_count"] = torch.tensor(selected["history_count"])
            
        # [1]
        y = torch.tensor(selected["loc_Y"], dtype=torch.long)
        # [1]
        mode = torch.tensor(selected["mode_Y"], dtype=torch.long)
        
        return x, tgt, y, mode, x_dict, tgt_dict

    def generate_data(self):

        # the valid location ids for unifying comparision
        self.valid_ids = load_pk_file(os.path.join(self.root, f"valid_ids_{self.dataset}.pk"))

        ori_data = pd.read_csv(os.path.join(self.root, f"dataset_{self.dataset}.csv"))
        ori_data.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

        # truncate too long duration, >2days to 2 days
        ori_data.loc[ori_data["duration"] > 60 * 24 * 2 - 1, "duration"] = 60 * 24 * 2 - 1

        # classify the datasets, user dependent 0.6, 0.2, 0.2
        train_data, vali_data, test_data = self.splitDataset(ori_data)

        # encode unseen locations in validation and test into 0
        enc = OrdinalEncoder(
            dtype=np.int64,
            handle_unknown="use_encoded_value",
            unknown_value=-1,
        ).fit(train_data["location_id"].values.reshape(-1, 1))
        # add 2 to account for unseen locations and to account for 0 padding
        train_data["location_id"] = enc.transform(train_data["location_id"].values.reshape(-1, 1)) + 2
        vali_data["location_id"] = enc.transform(vali_data["location_id"].values.reshape(-1, 1)) + 2
        test_data["location_id"] = enc.transform(test_data["location_id"].values.reshape(-1, 1)) + 2

        print(
            train_data["location_id"].max(),
            train_data["location_id"].unique().shape[0],
        )

        train_records = self.preProcessDatasets(train_data, "train")
        validation_records = self.preProcessDatasets(vali_data, "validation")
        test_records = self.preProcessDatasets(test_data, "test")

        if self.data_type == "test":
            return test_records
        if self.data_type == "validation":
            return validation_records
        if self.data_type == "train":
            return train_records

    def splitDataset(self, totalData):
        """Split dataset into train, vali and test."""
        totalData = totalData.groupby("user_id").apply(self.getSplitDaysUser)

        train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
        vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
        test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

        # final cleaning
        train_data.drop(columns={"Dataset"}, inplace=True)
        vali_data.drop(columns={"Dataset"}, inplace=True)
        test_data.drop(columns={"Dataset"}, inplace=True)

        return train_data, vali_data, test_data

    def getSplitDaysUser(self, df):
        """Split the dataset according to the tracked day of each user."""
        maxDay = df["start_day"].max()
        train_split = maxDay * 0.6
        vali_split = maxDay * 0.8

        df["Dataset"] = "test"
        df.loc[df["start_day"] < train_split, "Dataset"] = "train"
        df.loc[
            (df["start_day"] >= train_split) & (df["start_day"] < vali_split),
            "Dataset",
        ] = "vali"

        return df

    def preProcessDatasets(self, data, dataset_type):
        """Generate the datasets and save to the disk."""
        valid_records = self.getValidSequence(data)

        save_path = os.path.join(
            self.data_dir,
            f"{self.model_type}_{self.previous_day}_{self.predict_length}_{dataset_type}.pk",
        )
        save_pk_file(save_path, valid_records)

        return valid_records

    def getValidSequence(self, input_df):
        valid_user_ls = applyParallel(input_df.groupby("user_id"), self.getValidSequenceUser, n_jobs=-1)
        return [item for sublist in valid_user_ls for item in sublist]

    def getValidSequenceUser(self, df):

        df.reset_index(drop=True, inplace=True)

        data_single_user = []
        min_days = df["start_day"].min()
        df["diff_day"] = df["start_day"] - min_days

        for index, row in df.iterrows():
            # exclude the first records
            if row["diff_day"] < self.previous_day:
                continue

            hist = df.iloc[:index]
            hist = hist.loc[(hist["start_day"] >= (row["start_day"] - self.previous_day))]

            predict = df.iloc[index : index + self.predict_length]

            # print(predict)
            # print(row)

            if len(predict) != self.predict_length:
                continue

            # should be in the valid user ids
            if len(hist) < 3:
                continue

            data_dict = {}
            # only for deepmove: consider the last 2 days as curr
            if self.model_type == "deepmove":
                # and all other as history
                data_dict["history_count"] = len(hist.loc[hist["start_day"] < (row["start_day"] - 1)])
                # the history sequence and the current sequence shall not be 0
                if data_dict["history_count"] == 0 or data_dict["history_count"] == len(hist):
                    continue
            data_dict["X"] = hist["location_id"].values
            data_dict["user_X"] = hist["user_id"].values
            data_dict["start_min_X"] = hist["start_min"].values
            data_dict["mode_X"] = hist["mode"].values
            data_dict["length_X"] = hist["length_m"].values

            # the next location is the Y
            data_dict["loc_Y"] = predict["location_id"].astype(int).values
            # the next mode is the Y
            data_dict["mode_Y"] = predict["mode"].astype(int).values
            # print(data_dict["loc_Y"])

            data_single_user.append(data_dict)

        return data_single_user


def applyParallel(dfGrouped, func, n_jobs, print_progress=True, **kwargs):
    """
    Funtion warpper to parallelize funtions after .groupby().
    Parameters
    ----------
    dfGrouped: pd.DataFrameGroupBy
        The groupby object after calling df.groupby(COLUMN).
    func: function
        Function to apply to the dfGrouped object, i.e., dfGrouped.apply(func).
    n_jobs: int
        The maximum number of concurrently running jobs. If -1 all CPUs are used. If 1 is given, no parallel
        computing code is used at all, which is useful for debugging. See
        https://joblib.readthedocs.io/en/latest/parallel.html#parallel-reference-documentation
        for a detailed description
    print_progress: boolean
        If set to True print the progress of apply.
    **kwargs:
        Other arguments passed to func.
    Returns
    -------
    pd.DataFrame:
        The result of dfGrouped.apply(func)
    """
    df_ls = Parallel(n_jobs=n_jobs)(
        delayed(func)(group, **kwargs) for _, group in tqdm(dfGrouped, disable=not print_progress)
    )
    return df_ls


def save_pk_file(save_path, data):
    with open(save_path, "wb") as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pk_file(save_path):
    return pickle.load(open(save_path, "rb"))


def collate_fn(batch):
    """function to collate data samples into batch tensors."""
    # x, tgt, y, mode, x_dict, tgt_dict
    x_batch, tgt_batch, y_batch, mode_batch = [], [], [], []

    # get one sample batch
    x_dict_batch = {"len": []}
    tgt_dict_batch = {}
    for key in batch[0][-1]:
        x_dict_batch[key] = []
        tgt_dict_batch[key] = []

    for x_sample, tgt_sample, y_sample, mode_sample, x_dict, tgt_dict in batch:
        x_batch.append(x_sample)
        tgt_batch.append(tgt_sample)
        y_batch.append(y_sample)
        mode_batch.append(mode_sample)

        x_dict_batch["len"].append(len(x_sample))
        for key in x_dict:
            x_dict_batch[key].append(x_dict[key])
            
        for key in tgt_dict:
            tgt_dict_batch[key].append(tgt_dict[key])

    x_batch = pad_sequence(x_batch)
    tgt_batch = pad_sequence(tgt_batch)
    y_batch = pad_sequence(y_batch)
    mode_batch = pad_sequence(mode_batch)

    x_dict_batch["user"] = torch.tensor(x_dict_batch["user"], dtype=torch.int64)
    x_dict_batch["len"] = torch.tensor(x_dict_batch["len"], dtype=torch.int64)
    for key in x_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        x_dict_batch[key] = pad_sequence(x_dict_batch[key])
    tgt_dict_batch["user"] = torch.tensor(tgt_dict_batch["user"], dtype=torch.int64)
    for key in tgt_dict_batch:
        if key in ["user", "len", "history_count"]:
            continue
        tgt_dict_batch[key] = pad_sequence(tgt_dict_batch[key])

    return x_batch, tgt_batch, y_batch, mode_batch, x_dict_batch, tgt_dict_batch


def deepmove_collate_fn(batch):
    """function to collate data samples into batch tensors."""
    history_batch, curr_batch, tgt_batch = [], [], []

    # the context
    history_dict_batch = {
        "user": [],
        "weekday": [],
        "time": [],
        "len": [],
    }
    curr_dict_batch = {
        "user": [],
        "weekday": [],
        "time": [],
        "len": [],
    }

    for src_sample, tgt_sample, return_dict in batch:
        history_len = return_dict["history_count"]

        history_batch.append(src_sample[:history_len])
        curr_batch.append(src_sample[history_len:])
        tgt_batch.append(tgt_sample)

        history_dict_batch["len"].append(history_len)
        history_dict_batch["user"].append(return_dict["user"])
        history_dict_batch["weekday"].append(return_dict["weekday"][:history_len])
        history_dict_batch["time"].append(return_dict["time"][:history_len])

        curr_dict_batch["len"].append(len(src_sample[history_len:]))
        curr_dict_batch["user"].append(return_dict["user"])
        curr_dict_batch["weekday"].append(return_dict["weekday"][history_len:])
        curr_dict_batch["time"].append(return_dict["time"][history_len:])

    history_batch = pad_sequence(history_batch)
    curr_batch = pad_sequence(curr_batch)
    tgt_batch = torch.tensor(tgt_batch, dtype=torch.int64)

    history_dict_batch["len"] = torch.tensor(history_dict_batch["len"], dtype=torch.int64)
    history_dict_batch["user"] = torch.tensor(history_dict_batch["user"], dtype=torch.int64)
    history_dict_batch["weekday"] = pad_sequence(history_dict_batch["weekday"])
    history_dict_batch["time"] = pad_sequence(history_dict_batch["time"])

    curr_dict_batch["len"] = torch.tensor(curr_dict_batch["len"], dtype=torch.int64)
    curr_dict_batch["user"] = torch.tensor(curr_dict_batch["user"], dtype=torch.int64)
    curr_dict_batch["weekday"] = pad_sequence(curr_dict_batch["weekday"])
    curr_dict_batch["time"] = pad_sequence(curr_dict_batch["time"])

    return (
        (history_batch, curr_batch),
        tgt_batch,
        (history_dict_batch, curr_dict_batch),
    )


def test_dataloader(train_loader):

    batch_size = train_loader.batch_size

    ave_shape = 0
    y_shape = 0
    tgt_shape = 0
    for batch_idx, (x, tgt, Y, mode, x_dict, tgt_dict) in tqdm(enumerate(train_loader)):
        # print("batch_idx ", batch_idx)
        # print(inputs.shape)
        ave_shape += x.shape[0]
        tgt_shape += tgt.shape[0]
        y_shape += Y.shape[0]
        # print(inputs)
        # print(mode)
        # print(dict)
        # print(dict["user"].shape)

        # print(, batch_len)

        # print(data)
        # print(target)
        # print(dict)
        # if batch_idx > 10:
        #     break
    print(ave_shape / len(train_loader))
    print(tgt_shape / len(train_loader))
    print(y_shape / len(train_loader))


if __name__ == "__main__":
    source_root = r"./data/"

    dataset_train = gc_dataset(
        source_root,
        data_type="train",
        dataset="gc",
        previous_day=7,
        predict_length=5,
    )
    dataset_val = gc_dataset(
        source_root,
        data_type="validation",
        dataset="gc",
        previous_day=7,
        predict_length=5,
    )
    dataset_test = gc_dataset(
        source_root,
        data_type="test",
        dataset="gc",
        previous_day=7,
        predict_length=5,
    )

    kwds = {"shuffle": False, "num_workers": 0, "batch_size": 2}
    train_loader = torch.utils.data.DataLoader(dataset_train, collate_fn=collate_fn, **kwds)
    val_loader = torch.utils.data.DataLoader(dataset_val, collate_fn=collate_fn, **kwds)
    test_loader = torch.utils.data.DataLoader(dataset_test, collate_fn=collate_fn, **kwds)

    test_dataloader(train_loader)
    test_dataloader(val_loader)
    test_dataloader(test_loader)
