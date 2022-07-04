import json
import os
from pathlib import Path
import pandas as pd
import argparse

# trackintel
from trackintel.io.dataset_reader import read_geolife, geolife_add_modes_to_triplegs
from trackintel.preprocessing.triplegs import generate_trips
from trackintel.analysis.labelling import predict_transport_mode
from trackintel.geogr.distances import calculate_haversine_length
import trackintel as ti

# from config import config
from utils import get_time


def get_npp_dataset(config, epsilon=50, dataset="gc"):
    """Construct the raw staypoint with location id dataset."""
    ## read
    pfs, mode_labels = read_geolife(os.path.join(config[f"raw_geolife"], "data"), print_progress=True)
    # generate staypoints, triplegs and trips
    pfs, sp = pfs.as_positionfixes.generate_staypoints(
        time_threshold=5.0, gap_threshold=1e6, print_progress=True, n_jobs=-1
    )
    sp["duration"] = (sp["finished_at"] - sp["started_at"]).dt.total_seconds()
    
    pfs, tpls = pfs.as_positionfixes.generate_triplegs(sp)
    tpls = geolife_add_modes_to_triplegs(tpls, mode_labels)

    sp = ti.analysis.labelling.create_activity_flag(sp, time_threshold=15)

    sp, tpls, trips = ti.preprocessing.triplegs.generate_trips(sp, tpls, gap_threshold=15, add_geometry=False)

    # assign mode
    tpls["pred_mode"] = predict_transport_mode(tpls)["mode"]
    tpls.loc[tpls["mode"].isna(), "mode"] = tpls.loc[tpls["mode"].isna(), "pred_mode"]
    tpls.drop(columns={"pred_mode"}, inplace=True)

    # get the length
    tpls["length_m"] = calculate_haversine_length(tpls)

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
    trips_with_main_mode = trips_with_main_mode[~trips_with_main_mode["mode"].isna()]
    trips_with_main_mode_cate = get_mode_geolife(trips_with_main_mode)

    print(trips_with_main_mode_cate["mode"].value_counts())

    # filter activity staypoints
    sp = sp.loc[sp["is_activity"] == True].drop(columns=["is_activity", "trip_id", "next_trip_id"])

    # generate locations
    sp, locs = sp.as_staypoints.generate_locations(
        epsilon=epsilon, num_samples=2, distance_metric="haversine", agg_level="dataset", n_jobs=-1, print_progress=True
    )
    # filter noise staypoints
    valid_sp = sp.loc[~sp["location_id"].isna()].copy()

    # save locations
    locs = locs[~locs.index.duplicated(keep="first")]
    filtered_locs = locs.loc[locs.index.isin(sp["location_id"].unique())]

    path = Path(os.path.join(".", "data"))
    if not os.path.exists(path):
        os.makedirs(path)
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
    sp_time.drop(columns={"finished_at", "started_at", "geom", "elevation"}, inplace=True)
    sp_time.sort_values(by=["user_id", "start_day", "start_min"], inplace=True)
    sp_time = sp_time.reset_index(drop=True)
    sp_time["location_id"] = sp_time["location_id"].astype(int)
    sp_time["user_id"] = sp_time["user_id"].astype(int)

    sp_time.to_csv(os.path.join(".", "data", f"dataSet_{dataset}.csv"), index=False)


def get_mode_geolife(df):
    # slow_mobility
    df.loc[df["mode"] == "slow_mobility", "mode"] = "slow"
    df.loc[df["mode"] == "bike", "mode"] = "slow"
    df.loc[df["mode"] == "walk", "mode"] = "slow"
    df.loc[df["mode"] == "run", "mode"] = "slow"

    # motorized_mobility
    df.loc[df["mode"] == "motorized_mobility", "mode"] = "motorized"
    df.loc[df["mode"] == "bus", "mode"] = "motorized"
    df.loc[df["mode"] == "car", "mode"] = "motorized"
    df.loc[df["mode"] == "subway", "mode"] = "motorized"
    df.loc[df["mode"] == "taxi", "mode"] = "motorized"
    df.loc[df["mode"] == "train", "mode"] = "motorized"
    df.loc[df["mode"] == "boat", "mode"] = "motorized"

    # fast_mobility
    df.loc[df["mode"] == "fast_mobility", "mode"] = "fast"
    df.loc[df["mode"] == "airplane", "mode"] = "fast"
    return df


if __name__ == "__main__":
    DBLOGIN_FILE = os.path.join(".", "paths.json")
    with open(DBLOGIN_FILE) as json_file:
        CONFIG = json.load(json_file)

    # dataset = {"gc", "geolife"}
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=str, nargs="?", help="Dataset to preprocess", default="geolife")
    parser.add_argument("epsilon", type=int, nargs="?", help="epsilon for dbscan to detect locations", default=20)
    args = parser.parse_args()

    get_npp_dataset(epsilon=args.epsilon, dataset=args.dataset, config=CONFIG)
