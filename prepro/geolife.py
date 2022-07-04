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
import argparse

# trackintel
from trackintel.analysis.tracking_quality import temporal_tracking_quality, _split_overlaps
from trackintel.io.dataset_reader import read_geolife
from trackintel.preprocessing.triplegs import generate_trips
import trackintel as ti

# from config import config
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

if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "dblogin.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    # dataset = {"gc", "geolife"}
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?", help="Dataset to preprocess", default="geolife")
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_npp_dataset(epsilon=args.epsilon, dataset=args.dataset, config=CONFIG)
