import json
import os
from pathlib import Path

import pandas as pd
import geopandas as gpd
import numpy as np

import datetime
import psycopg2
import argparse

# trackintel
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

# own function
from utils import filter_duplicates, get_time, get_mode, preprocess_to_trackintel


def get_npp_dataset(config, epsilon=50, dataset="gc"):
    """Construct the raw staypoint with location id dataset."""
    ## read and change name to trackintel format
    sp = pd.read_csv(os.path.join(config[f"raw_{dataset}"], "stps.csv"))
    tpls = pd.read_csv(os.path.join(config[f"raw_{dataset}"], "tpls.csv"))

    # initial cleaning
    sp.rename(columns={"activity": "is_activity"}, inplace=True)

    # transform to trackintel format
    sp = preprocess_to_trackintel(sp)
    tpls = preprocess_to_trackintel(tpls)

    # get the length
    tpls_proj = tpls.to_crs("EPSG:2056")
    tpls["length_m"] = tpls_proj.length

    # ensure the timeline of sp and tpls does not overlap
    sp_no_overlap_time, tpls_no_overlap_time = filter_duplicates(sp.copy().reset_index(), tpls.reset_index())

    # the trackintel trip generation
    sp, tpls, trips = generate_trips(sp_no_overlap_time, tpls_no_overlap_time, add_geometry=False)

    ## select valid user
    quality_path = os.path.join(".", "data", "quality", f"{dataset}_slide_filtered.csv")
    if Path(quality_path).is_file():
        valid_users = pd.read_csv(quality_path)["user_id"].values
    else:
        parent = Path(quality_path).parent.absolute()
        if not os.path.exists(parent):
            os.makedirs(parent)
        valid_users = _calculate_user_quality(sp.copy(), trips.copy(), quality_path, dataset)
    sp = sp.loc[sp["user_id"].isin(valid_users)]
    tpls = tpls.loc[tpls["user_id"].isin(valid_users)]
    trips = trips.loc[trips["user_id"].isin(valid_users)]

    groupsize = tpls.groupby("trip_id").size().to_frame(name="triplegNum").reset_index()
    tpls_group = tpls.merge(groupsize, on="trip_id")

    # trips only with 1 triplegs
    res1 = tpls_group.loc[tpls_group["triplegNum"] == 1][["trip_id", "length_m", "mode"]].copy()

    # get the mode and length of remaining trips
    remain = tpls_group.loc[tpls_group["triplegNum"] != 1].copy()
    remain.sort_values(by="length_m", inplace=True, ascending=False)
    mode = remain.groupby("trip_id").head(1).reset_index(drop=True)[["mode", "trip_id"]]

    length = remain.groupby("trip_id")["length_m"].sum().reset_index()
    res2 = mode.merge(length, on="trip_id")
    # concat the results
    res = pd.concat([res1, res2])
    res.rename(columns={"trip_id": "id"}, inplace=True)
    res.set_index("id", inplace=True)

    trips_with_main_mode = trips.join(res, how="left")
    trips_with_main_mode_cate = get_mode(trips_with_main_mode, dataset=dataset)

    print(trips_with_main_mode_cate["mode"].value_counts())

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True].drop(columns=["is_activity", "trip_id", "next_trip_id"])

    # generate locations
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=2, distance_metric="haversine", agg_level="dataset", n_jobs=-1, print_progress=True
    )
    # filter noise staypoints
    valid_sp = sp.loc[~sp["location_id"].isna()].copy()
    # print("After filter non-location staypoints: ", sp.shape[0])

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]

    filtered_locs.as_locations.to_csv(os.path.join(".", "data", f"locations_{dataset}.csv"))

    # merge staypoint with trips info
    sp = valid_sp.loc[~valid_sp["prev_trip_id"].isna()].reset_index().copy()
    trips = (
        trips_with_main_mode_cate.drop(columns=["started_at", "finished_at", "user_id"])
        .reset_index()
        .rename(columns={"id": "trip_id"})
        .copy()
    )

    # sp
    sp["prev_trip_id"] = sp["prev_trip_id"].astype(float)
    trips["trip_id"] = trips["trip_id"].astype(float)

    merged_sp = sp.merge(trips, left_on="prev_trip_id", right_on="trip_id", how="left")
    sp = merged_sp.loc[~merged_sp["trip_id"].isna()].drop(
        columns=["origin_staypoint_id", "prev_trip_id", "destination_staypoint_id"]
    )

    sp_time = sp.groupby("user_id").apply(get_time)

    # get the time info
    sp_time.drop(columns={"finished_at", "started_at", "geom"}, inplace=True)
    sp_time.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp_time = sp_time.reset_index(drop=True)
    sp_time["location_id"] = sp_time["location_id"].astype(int)
    sp_time["user_id"] = sp_time["user_id"].astype(int)

    sp_time.to_csv(f"./data/dataSet_{dataset}.csv", index=False)


def _calculate_user_quality(sp, trips, file_path, dataset):
    sp["started_at"] = sp["started_at"].dt.tz_localize(tz=None)
    sp["finished_at"] = sp["finished_at"].dt.tz_localize(tz=None)
    trips["started_at"] = trips["started_at"].dt.tz_localize(tz=None)
    trips["finished_at"] = trips["finished_at"].dt.tz_localize(tz=None)

    # merge trips and staypoints
    sp["type"] = "sp"
    trips["type"] = "trip"
    df_all = pd.merge(sp, trips, how="outer")
    df_all = _split_overlaps(df_all, granularity="day")
    df_all["duration"] = (df_all["finished_at"] - df_all["started_at"]).dt.total_seconds()

    if dataset == "gc":
        end_period = datetime.datetime(2017, 12, 26, tzinfo=None)
        df_all = df_all.loc[df_all["finished_at"] < end_period]
        day_filter = 300
        window_size = 10
        min_thres = 0.6
        mean_thres = 0.7
    elif dataset == "yumuv":
        day_filter = 30
        window_size = 4
        min_thres = 0.5
        mean_thres = 0.6

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
    filter_after_day = sliding_quality.loc[sliding_quality["user_id"].isin(user_filter_day)]

    # filter based on quanlity
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


def get_con(LOGIN_DATA):
    con = psycopg2.connect(
        dbname=LOGIN_DATA["database"],
        user=LOGIN_DATA["user"],
        password=LOGIN_DATA["password"],
        host=LOGIN_DATA["host"],
        port=LOGIN_DATA["port"],
    )
    return con


def read_data(config):
    # read the trips
    print("Reading staypoints!")
    sp = gpd.read_postgis(
        sql="SELECT * FROM raw_myway.story_line_only_fk_cg",
        con=get_con(config),
        geom_col="geom_stay",
        index_col="id",
    )

    print("Reading triplegs!")
    tpls = gpd.read_postgis(
        sql="SELECT * FROM raw_myway.story_line_only_fk_cg",
        con=get_con(config),
        geom_col="geom_track",
        index_col="id",
    )

    sp = sp.loc[sp["story_line_type"] == "STAY"]
    tpls = tpls.loc[tpls["story_line_type"] == "TRACK"]

    sp.drop(
        columns=[
            "story_line_type",
            "track_mode",
            "inserted",
            "updated",
            "deleted",
            "confirmed",
            "geom_track",
            "study_id",
        ],
        inplace=True,
    )
    tpls.drop(
        columns=[
            "story_line_type",
            "inserted",
            "updated",
            "deleted",
            "confirmed",
            "geom_stay",
            "study_id",
            "stay_purpose",
        ],
        inplace=True,
    )

    sp.rename(columns={"user_fk": "user_id", "stay_purpose": "purpose", "geom_stay": "geom"}, inplace=True)
    tpls.rename(columns={"user_fk": "user_id", "track_mode": "mode", "geom_track": "geom"}, inplace=True)

    sp["is_activity"] = sp["finished_at"] - sp["started_at"] > datetime.timedelta(minutes=25)
    sp.loc[sp["purpose"] == "wait", "is_activity"] = False

    # index management
    sp.sort_values(by=["user_id", "started_at"], inplace=True)
    tpls.sort_values(by=["user_id", "started_at"], inplace=True)

    sp.reset_index(drop=True, inplace=True)
    sp.index.name = "id"

    tpls.reset_index(drop=True, inplace=True)
    tpls.index.name = "id"

    # save
    sp.to_csv(os.path.join(config[f"raw_yumuv"], "stps.csv"))
    tpls.to_csv(os.path.join(config[f"raw_yumuv"], "tpls.csv"))


if __name__ == "__main__":
    # read file storage
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    # dataset = {"gc", "yumuv"}
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?", help="Dataset to preprocess", default="gc")
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_npp_dataset(epsilon=args.epsilon, dataset=args.dataset, config=CONFIG)
