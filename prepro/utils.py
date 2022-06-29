

import pandas as pd

def filter_duplicates(sp, tpls):

    # merge trips and staypoints
    sp["type"] = "sp"
    tpls["type"] = "tpl"
    df_all = pd.merge(sp, tpls, how="outer")

    df_all = df_all.groupby("user_id", as_index=False).apply(_alter_diff)
    sp = df_all.loc[df_all["type"] == "sp"].drop(columns=["type"])
    tpls = df_all.loc[df_all["type"] == "tpl"].drop(columns=["type"])

    # sp = sp[["id", "user_id", "started_at", "finished_at", "geom", "duration", "purpose", "is_activity"]]
    sp = sp[["id", "user_id", "started_at", "finished_at", "geom", "duration", "is_activity"]]
    tpls = tpls[["id", "user_id", "started_at", "finished_at", "geom", "length_m", "duration", "mode"]]
    # tpls = tpls[["id", "user_id", "started_at", "finished_at", "geom", "duration", "mode"]]

    return sp.set_index("id"), tpls.set_index("id")


def _alter_diff(df):
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


def get_time(df):
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


def get_mode(df, dataset="gc"):
    if dataset == "gc":
        # Bicycle
        df.loc[df["mode"] == "Mode::Ebicycle", "mode"] = "Mode::Bicycle"
        # df.loc[df["mode"] == "Mode::Bicycle", "mode"] = "Mode::Bicycle"

        # Car
        df.loc[df["mode"] == "Mode::Ecar", "mode"] = "Mode::Car"
        # df.loc[df["mode"] == "Mode::Car", "mode"] = "Mode::Car"

        # Walk
        # df.loc[df["mode"] == "Mode::Walk", "mode"] = "Mode::Walk"

        # Bus
        # df.loc[df["mode"] == "Mode::Bus", "mode"] = "Mode::Bus"
        df.loc[df["mode"] == "Mode::Boat", "mode"] = "Mode::Bus"

        # Tram
        # df.loc[df["mode"] == "Mode::Tram", "mode"] = "Mode::Tram"

        # Train
        # df.loc[df["mode"] == "Mode::Train", "mode"] = "Mode::Train"

        # other
        df.loc[df["mode"] == "Mode::Ski", "mode"] = "Mode::Other"
        df.loc[df["mode"] == "Mode::Airplane", "mode"] = "Mode::Other"
        df.loc[df["mode"] == "Mode::Coach", "mode"] = "Mode::Other"
    elif dataset == "yumuv":
        # Bicycle
        df.loc[df["mode"] == "ebicycle", "mode"] = "bicycle"
        df.loc[df["mode"] == "kick_scooter", "mode"] = "bicycle"
        # df.loc[df["mode"] == "bicycle", "mode"] = "bicycle"

        # Car
        df.loc[df["mode"] == "ecar", "mode"] = "car"
        df.loc[df["mode"] == "motorbike", "mode"] = "car"
        # df.loc[df["mode"] == "car", "mode"] = "car"

        # Walk
        # df.loc[df["mode"] == "walk", "mode"] = "walk"

        # Bus
        # df.loc[df["mode"] == "bus", "mode"] = "bus"
        df.loc[df["mode"] == "boat", "mode"] = "bus"

        # Tram
        # df.loc[df["mode"] == "tram", "mode"] = "tram"

        # Train
        # df.loc[df["mode"] == "train", "mode"] = "train"

        # other
        df.loc[df["mode"] == "ski", "mode"] = "other"
        df.loc[df["mode"] == "airplane", "mode"] = "other"
        df.loc[df["mode"] == "coach", "mode"] = "other"

    return df
