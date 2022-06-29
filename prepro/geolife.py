import json
import os
import pickle as pickle
from sklearn.preprocessing import OrdinalEncoder
from pathlib import Path

import pandas as pd
import numpy as np
import geopandas as gpd
from tqdm import tqdm
import datetime

from joblib import Parallel, delayed
import multiprocessing

from shapely import wkt
from gensim.corpora import Dictionary
from gensim.models import TfidfModel, LdaModel
import gensim

# trackintel
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

# from config import config


def get_npp_dataset(epsilon=50, dataset="gc"):
    """Construct the raw staypoint with location id dataset from GC data."""
    # read file storage
    DBLOGIN_FILE = os.path.join(".", "dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    if dataset == "gc":
        ## read and change name to trackintel format
        df = pd.read_csv(os.path.join(CONFIG[f"raw_{dataset}"], "stps.csv"))
        gdf = _preprocess_to_trackintel(df)

        ## select valid user
        quality_path = os.path.join(".", "data", "quality", f"{dataset}_slide_filtered.csv")
        if Path(quality_path).is_file():
            valid_user = pd.read_csv(quality_path)["user_id"].values
        else:
            valid_user = _calculate_user_quality(CONFIG, quality_path, dataset)
        gdf = gdf.loc[gdf["user_id"].isin(valid_user)]

        # select only switzerland records
        swissBoundary = gpd.read_file(os.path.join(".", "data", "swiss", "swiss_1903+.shp"))
        # print("Before spatial filtering: ", gdf.shape[0])
        gdf = _filter_within_swiss(gdf, swissBoundary)
        # print("After spatial filtering: ", gdf.shape[0])

    elif dataset == "geolife":
        pfs, _ = read_geolife(CONFIG["raw_geolife"], print_progress=True)
        # generate staypoints
        pfs, sp = pfs.as_positionfixes.generate_staypoints(
            time_threshold=5.0, gap_threshold=1e6, print_progress=True, n_jobs=-1
        )
        # create activity flag
        sp = sp.as_staypoints.create_activity_flag(time_threshold=15)

        ## select valid user
        quality_path = os.path.join(".", "data", "quality", f"{dataset}_slide_filtered.csv")
        if Path(quality_path).is_file():
            valid_user = pd.read_csv(quality_path)["user_id"].values
        else:
            valid_user = _calculate_user_quality(CONFIG, quality_path, dataset, pfs, sp)

        gdf = sp.loc[sp["user_id"].isin(valid_user)]

    # filter activity staypoints
    gdf = gdf.loc[gdf["activity"] == True]

    # generate locations
    sp, locs = gdf.as_staypoints.generate_locations(
        epsilon=epsilon,
        num_samples=2,
        distance_metric="haversine",
        agg_level="dataset",
        n_jobs=-1,
    )
    # filter noise staypoints
    sp = sp.loc[~sp["location_id"].isna()].copy()
    print("After filter non-location staypoints: ", sp.shape[0])

    sp = sp[
        [
            "user_id",
            "started_at",
            "finished_at",
            "geom",
            "location_id",
        ]
    ]
    # merge staypoints
    sp_merged = sp.as_staypoints.merge_staypoints(
        triplegs=pd.DataFrame([]),
        max_time_gap="1min",
        agg={"location_id": "first"},
    )
    print("After staypoints merging: ", sp_merged.shape[0])
    # recalculate staypoint duration
    sp_merged["duration"] = (sp_merged["finished_at"] - sp_merged["started_at"]).dt.total_seconds() // 60

    # get the time info
    sp_time = sp_merged.groupby("user_id").apply(_get_time_info)
    sp_time.drop(columns={"finished_at", "started_at"}, inplace=True)
    sp_time.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp_time = sp_time.reset_index(drop=True)

    # filter infrequent locations
    sp = _filter_infrequent(sp_time, min_count=5)

    print("After filter infrequent location: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]
    filtered_locs.as_locations.to_csv(os.path.join(".", "data", f"locations_{dataset}.csv"))
    print("Location size: ", sp["location_id"].unique().shape[0], filtered_locs.shape[0])

    # final cleaning, reassign ids
    sp["location_id"] = sp["location_id"].astype(int)
    sp["user_id"] = sp["user_id"].astype(int)
    sp.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp.index.name = "id"
    sp.reset_index(inplace=True)
    # sp.to_csv(os.path.join(".", "data", f"dataSet_{dataset}.csv"))

    print("User size: ", sp["user_id"].unique().shape[0])

    _filter_sp_history(sp, dataset)


def _filter_sp_history(sp, dataset):
    """To unify the comparision between different previous days"""
    # encoder user
    enc = OrdinalEncoder(dtype=np.int64)
    sp["user_id"] = enc.fit_transform(sp["user_id"].values.reshape(-1, 1)) + 1

    # classify the datasets, user dependent 0.6, 0.2, 0.2
    train_data, vali_data, test_data = _split_dataset(sp)

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

    # the days to consider when generating final_valid_id
    if dataset == "gc":
        previous_day_ls = list(np.arange(14) + 1)
    else:
        previous_day_ls = [7]
    all_ids = sp[["id"]].copy()

    # for each previous_day, get the valid staypoint id
    for previous_day in tqdm(previous_day_ls):
        valid_ids = _get_valid_sequence(train_data, previous_day=previous_day)
        valid_ids.extend(_get_valid_sequence(vali_data, previous_day=previous_day))
        valid_ids.extend(_get_valid_sequence(test_data, previous_day=previous_day))

        all_ids[f"{previous_day}"] = 0
        all_ids.loc[all_ids["id"].isin(valid_ids), f"{previous_day}"] = 1

    # get the final valid staypoint id
    all_ids.set_index("id", inplace=True)
    final_valid_id = all_ids.loc[all_ids.sum(axis=1) == all_ids.shape[1]].reset_index()["id"].values

    # filter the user again based on final_valid_id:
    # if an user has no record in final_valid_id, we discard the user
    valid_users = sp.loc[sp["id"].isin(final_valid_id), "user_id"].unique()
    filtered_sp = sp.loc[sp["user_id"].isin(valid_users)]
    # extra filter for geolife:
    # the lack of records might create no records in train, validation or test sets,
    # we additionally enforce each user need to have at least 350 staypoints
    if dataset == "geolife":
        user_size = filtered_sp.groupby("user_id").size()
        filtered_sp = filtered_sp.loc[filtered_sp["user_id"].isin(user_size[user_size > 150].index)].copy()

    # after user filter, we reencode the users:
    # to ensure the user_id is continues
    # we do not need to encode the user_id again in dataloader.py
    enc = OrdinalEncoder(dtype=np.int64)
    filtered_sp["user_id"] = enc.fit_transform(filtered_sp["user_id"].values.reshape(-1, 1)) + 1

    # save the valid_ids and dataset
    data_path = f"./data/valid_ids_{dataset}.pk"
    with open(data_path, "wb") as handle:
        pickle.dump(final_valid_id, handle, protocol=pickle.HIGHEST_PROTOCOL)
    filtered_sp.to_csv(f"./data/dataSet_{dataset}.csv", index=False)

    print("Final user size: ", filtered_sp["user_id"].unique().shape[0])


def _split_dataset(totalData):
    """Split dataset into train, vali and test."""
    totalData = totalData.groupby("user_id").apply(_get_split_days_user)

    train_data = totalData.loc[totalData["Dataset"] == "train"].copy()
    vali_data = totalData.loc[totalData["Dataset"] == "vali"].copy()
    test_data = totalData.loc[totalData["Dataset"] == "test"].copy()

    # final cleaning
    train_data.drop(columns={"Dataset"}, inplace=True)
    vali_data.drop(columns={"Dataset"}, inplace=True)
    test_data.drop(columns={"Dataset"}, inplace=True)

    return train_data, vali_data, test_data


def _get_split_days_user(df):
    """Split the dataset according to the tracked day of each user."""
    maxDay = df["start_day"].max()
    train_split = maxDay * 0.6
    validation_split = maxDay * 0.8

    df["Dataset"] = "test"
    df.loc[df["start_day"] < train_split, "Dataset"] = "train"
    df.loc[(df["start_day"] >= train_split) & (df["start_day"] < validation_split), "Dataset"] = "vali"

    return df


def _calculate_user_quality(config, file_path, dataset, pfs=None, sp=None):
    if dataset == "gc":
        sp = pd.read_csv(os.path.join(config["quality_gc"], "stps_act_user_50.csv"))
        trips = pd.read_csv(os.path.join(config["quality_gc"], "trips_hir.csv"))

        sp.rename(
            columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at", "dur_s": "duration"},
            inplace=True,
        )
        trips.rename(
            columns={"userid": "user_id", "startt": "started_at", "endt": "finished_at", "dur_s": "duration"},
            inplace=True,
        )
    elif dataset == "geolife":
        # generate triplegs
        pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
        # generate trips
        sp, tpls, trips = generate_trips(sp, tpls, gap_threshold=15)

    trips["started_at"] = pd.to_datetime(trips["started_at"]).dt.tz_localize(None)
    trips["finished_at"] = pd.to_datetime(trips["finished_at"]).dt.tz_localize(None)
    sp["started_at"] = pd.to_datetime(sp["started_at"]).dt.tz_localize(None)
    sp["finished_at"] = pd.to_datetime(sp["finished_at"]).dt.tz_localize(None)
    # merge trips and staypoints
    print("starting merge", sp.shape, trips.shape)
    sp["type"] = "sp"
    trips["type"] = "trip"
    df_all = pd.merge(sp, trips, how="outer")
    df_all = _split_overlaps(df_all, granularity="day", max_iter=1200)
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()
    print("finished merge", df_all.shape)
    print("*" * 50)

    if dataset == "gc":
        end_period = datetime.datetime(2017, 12, 26)
        df_all = df_all.loc[df_all["finished_at"] < end_period]
        day_filter = 300
        window_size = 10
        min_thres = 0.6
        mean_thres = 0.7
    else:
        df_all = df_all.groupby("user_id", as_index=False).apply(__alter_diff)
        day_filter = 50
        window_size = 5
        min_thres = 0.5
        mean_thres = 0.6

    print(len(df_all["user_id"].unique()))

    # gc1
    # get quality
    total_quality = temporal_tracking_quality(df_all, granularity="all")
    # get tracking days
    total_quality["days"] = (
        df_all.groupby("user_id").apply(lambda x: (x["finished_at"].max() - x["started_at"].min()).days).values
    )

    user_filter_day = total_quality.loc[(total_quality["days"] > day_filter)].reset_index(drop=True)["user_id"].unique()

    sliding_quality = (
        df_all.groupby("user_id").apply(__get_tracking_quality, window_size=window_size).reset_index(drop=True)
    )

    # filter based on days
    print(sliding_quality["user_id"].unique().shape[0])
    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]
    print(filter_after_day["user_id"].unique().shape[0])

    # SBB
    filter_after_user_quality = (
        filter_after_day.groupby("user_id")
        .apply(__filter_user, min_thres=min_thres, mean_thres=mean_thres)
        .reset_index(drop=True)
        .dropna()
    )
    filter_after_user_quality = filter_after_user_quality.groupby("user_id", as_index=False)["quality"].mean()

    print("final selected user", filter_after_user_quality.shape[0])

    filter_after_user_quality.to_csv(file_path, index=False)

    return filter_after_user_quality["user_id"].values


def __alter_diff(df):
    df.sort_values(by="started_at", inplace=True)
    df["diff"] = pd.NA
    df["st_next"] = pd.NA

    diff = df.iloc[1:]["started_at"].reset_index(drop=True) - df.iloc[:-1]["finished_at"].reset_index(drop=True)
    df["diff"][:-1] = diff.dt.total_seconds()
    df["st_next"][:-1] = df.iloc[1:]["started_at"].reset_index(drop=True)

    df.loc[df["diff"] < 0, "finished_at"] = df.loc[df["diff"] < 0, "st_next"]

    df["started_at"], df["finished_at"] = pd.to_datetime(df["started_at"]), pd.to_datetime(df["finished_at"])
    df["duration"] = (df["finished_at"] - df["started_at"]).dt.total_seconds()

    # print(df.loc[df["diff"] < 0])
    df.drop(columns=["diff", "st_next"], inplace=True)
    df.drop(index=df[df["duration"] <= 0].index, inplace=True)

    return df


def __get_tracking_quality(df, window_size):

    weeks = (df["finished_at"].max() - df["started_at"].min()).days // 7
    start_date = df["started_at"].min().date()

    quality_list = []
    # construct the sliding week gdf
    for i in range(0, weeks - window_size + 1):
        curr_start = datetime.datetime.combine(start_date + datetime.timedelta(weeks=i), datetime.time())
        curr_end = datetime.datetime.combine(curr_start + datetime.timedelta(weeks=window_size), datetime.time())

        # the total df for this time window
        cAll_gdf = df.loc[(df["started_at"] >= curr_start) & (df["finished_at"] < curr_end)]
        if cAll_gdf.shape[0] == 0:
            continue
        total_sec = (curr_end - curr_start).total_seconds()

        quality_list.append([i, cAll_gdf["duration"].sum() / total_sec])
    ret = pd.DataFrame(quality_list, columns=["timestep", "quality"])
    ret["user_id"] = df["user_id"].unique()[0]
    return ret


def __filter_user(df, min_thres, mean_thres):
    consider = df.loc[df["quality"] != 0]
    if (consider["quality"].min() > min_thres) and (consider["quality"].mean() > mean_thres):
        return df


def _preprocess_to_trackintel(df):
    """Change dataframe to trackintel compatible format"""
    df.rename(
        columns={
            "userid": "user_id",
            "startt": "started_at",
            "endt": "finished_at",
            "dur_s": "duration",
        },
        inplace=True,
    )
    # drop invalid
    df.drop(index=df[df["duration"] < 0].index, inplace=True)

    # read the time info
    df["started_at"] = pd.to_datetime(df["started_at"])
    df["finished_at"] = pd.to_datetime(df["finished_at"])
    df["started_at"] = df["started_at"].dt.tz_localize(tz="utc")
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz="utc")

    # choose only activity staypoints
    df.set_index("id", inplace=True)
    tqdm.pandas(desc="Load geometry")
    df["geom"] = df["geom"].progress_apply(wkt.loads)

    return gpd.GeoDataFrame(df, crs="EPSG:4326", geometry="geom")


def _filter_within_swiss(stps, swissBound):
    """Spatial filtering of staypoints."""
    # save a copy of the original projection
    init_crs = stps.crs
    # project to projected system
    stps = stps.to_crs(swissBound.crs)

    ## parallel for speeding up
    stps["within"] = _apply_parallel(stps["geom"], _apply_extract, swissBound)
    sp_swiss = stps[stps["within"] == True].copy()
    sp_swiss.drop(columns=["within"], inplace=True)

    return sp_swiss.to_crs(init_crs)


def _apply_extract(df, swissBound):
    """The func for _apply_parallel: judge whether inside a shp."""
    tqdm.pandas(desc="pandas bar")
    shp = swissBound["geometry"].to_numpy()[0]
    return df.progress_apply(lambda x: shp.contains(x))


def _apply_parallel(df, func, other, n=-1):
    """parallel apply for spending up."""
    if n is None:
        n = -1
    dflength = len(df)
    cpunum = multiprocessing.cpu_count()
    if dflength < cpunum:
        spnum = dflength
    if n < 0:
        spnum = cpunum + n + 1
    else:
        spnum = n or 1

    sp = list(range(dflength)[:: int(dflength / spnum + 0.5)])
    sp.append(dflength)
    slice_gen = (slice(*idx) for idx in zip(sp[:-1], sp[1:]))
    results = Parallel(n_jobs=n, verbose=0)(delayed(func)(df[slc], other) for slc in slice_gen)
    return pd.concat(results)


def _get_time_info(df):
    min_day = pd.to_datetime(df["started_at"].min().date())
    df["started_at"] = df["started_at"].dt.tz_localize(tz=None)
    df["finished_at"] = df["finished_at"].dt.tz_localize(tz=None)

    df["start_day"] = (df["started_at"] - min_day).dt.days
    df["end_day"] = (df["finished_at"] - min_day).dt.days

    df["start_min"] = df["started_at"].dt.hour * 60 + df["started_at"].dt.minute
    df["end_min"] = df["finished_at"].dt.hour * 60 + df["finished_at"].dt.minute
    df.loc[df["end_min"] == 0, "end_min"] = 24 * 60

    df["weekday"] = df["started_at"].dt.weekday
    return df


def _filter_infrequent(df, min_count=5):
    """filter infrequent locations"""
    value_counts = df["location_id"].value_counts()
    valid = value_counts[value_counts > min_count]

    return df.loc[df["location_id"].isin(valid.index)].copy()


if __name__ == "__main__":

    # dataset = {"gc", "geolife"}
    dataset = "gc"
    epsilon = 20

    get_npp_dataset(epsilon=epsilon, dataset=dataset)
