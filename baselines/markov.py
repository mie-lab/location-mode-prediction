import pickle
import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import os

from scipy import stats
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["figure.dpi"] = 300
matplotlib.rcParams["xtick.labelsize"] = 13
matplotlib.rcParams["ytick.labelsize"] = 13
np.set_printoptions(precision=4)
np.set_printoptions(suppress=True)

colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
np.random.seed(0)


def splitDataset(totalData):
    """Split dataset into train, vali and test."""
    totalData = totalData.groupby("user_id").apply(getSplitDaysUser)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def getSplitDaysUser(df):
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


def markov_transition_prob(df, n=1):
    COLUMNS = [f"loc_{i+1}" for i in range(n)]
    COLUMNS.append("toLoc")

    locSequence = pd.DataFrame(columns=COLUMNS)

    locSequence["toLoc"] = df.iloc[n:]["location_id"].values
    for i in range(n):
        locSequence[f"loc_{i+1}"] = df.iloc[i : -n + i]["location_id"].values
    return locSequence.groupby(by=COLUMNS).size().to_frame("size").reset_index()


def get_true_pred_pair(locSequence, df, n=1):
    testSeries = df["location_id"].values

    true_ls = []
    pred_ls = []

    for i in range(testSeries.shape[0] - n):
        locCurr = testSeries[i : i + n + 1]
        numbLoc = n

        # loop until finds a match
        while True:
            res_df = locSequence
            for j in range(n - numbLoc, n):
                res_df = res_df.loc[res_df[f"loc_{j+1}"] == locCurr[j]]
            res_df = res_df.sort_values(by="size", ascending=False)

            if res_df.shape[0]:  # if the dataframe contains entry, stop finding
                # choose the location which are visited most often for the matches
                pred = res_df["toLoc"].drop_duplicates().values
                break
            # decrese the number of location history considered
            numbLoc -= 1
            if numbLoc == 0:  # if even the instant last location is not in the location set
                # choose the location which are visited most often globally
                pred = locSequence.sort_values(by="size", ascending=False)["toLoc"].drop_duplicates().values
                break

        true_ls.append(locCurr[-1])
        pred_ls.append(pred)
    return true_ls, pred_ls


#


def get_performance_measure(true_ls, pred_ls):
    acc_ls = [1, 5, 10]

    res = []
    # total number
    res.append(len(true_ls))
    for top_acc in acc_ls:

        correct = 0
        for true, pred in zip(true_ls, pred_ls):
            if true in pred[:top_acc]:
                correct += 1
        res.append(correct)

    f1 = f1_score(true_ls, [pred[0] for pred in pred_ls], average="weighted")
    res.append(f1)

    # rr
    rank_ls = []
    for true, pred in zip(true_ls, pred_ls):
        rank = (np.nonzero(pred == true)[0] + 1).astype(float)
        if len(rank):
            rank_ls.append(rank[0])
        else:
            rank_ls.append(0)
    rank = np.array(rank_ls)

    # c =
    if rank.sum() != 0:
        rank = np.divide(1, rank, out=np.zeros_like(rank), where=rank != 0)
    # rank[rank == np.inf] = 0
    # append the result
    res.append(rank.sum())

    return pd.Series(res, index=["total", "correct@1", "correct@5", "correct@10", "f1", "rr"])


def get_markov_res(train, test, n=2):
    locSeq_df = markov_transition_prob(train, n=n)

    true_ls, pred_ls = get_true_pred_pair(locSeq_df, test, n=n)

    # print(locSeq)
    return get_performance_measure(true_ls, pred_ls)


#

if __name__ == "__main__":
    #  the number of previous locations considered (n-Markov)
    n = 1

    #
    source_root = r".\data"
    # "gc" or "yumuv"
    dataset = "yumuv"
    # read data
    inputData = pd.read_csv(os.path.join(source_root, f"dataSet_{dataset}.csv"))
    inputData.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)

    # split data
    train_data, vali_data, test_data = splitDataset(inputData)

    print(train_data.shape, vali_data.shape, test_data.shape)

    true_all_ls = []
    pred_all_ls = []
    res_ls = []
    for user in tqdm(train_data["user_id"].unique()):
        # get the train and test sets for each user
        curr_train = train_data.loc[train_data["user_id"] == user]
        curr_test = test_data.loc[test_data["user_id"] == user]

        # get the results
        res = get_markov_res(curr_train, curr_test, n=n)
        res_ls.append(res)


    result = pd.DataFrame(res_ls)

    print(result["correct@1"].sum() / result["total"].sum() * 100)
    print(result["correct@5"].sum() / result["total"].sum() * 100)
    print(result["correct@10"].sum() / result["total"].sum() * 100)
    print(result["rr"].sum() / result["total"].sum() * 100)
    print(result["f1"].mean() * 100)
