#!/usr/bin/env python3
"""
Full training pipeline.

Run:
    python train.py                # trains on full 37M-row dataset
    python train.py --sample       # trains on 1M-row sample (fast iteration)

Produces model.pkl containing all inference artifacts (model + lookup tables).
"""
from __future__ import annotations

import argparse
import io
import json
import pickle
import time
import zipfile
from pathlib import Path

import geopandas as gpd
import lightgbm as lgb
import numpy as np
import pandas as pd
import requests

from features import FEATURE_NAMES, build_single

DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = Path(__file__).parent / "model.pkl"

SMOOTHING_K = 50  # Bayesian smoothing: blend local estimate toward global mean


# ---------------------------------------------------------------------------
# Zone centroids
# ---------------------------------------------------------------------------

def _fetch_zone_centroids() -> dict:
    """Return {zone_id: (lat, lon)}, downloading shapefile once if needed."""
    cached = DATA_DIR / "zone_centroids.json"
    if cached.exists():
        with open(cached) as f:
            raw = json.load(f)
        return {int(k): tuple(v) for k, v in raw.items()}

    print("  downloading TLC zone shapefile...")
    url = "https://d37ci6vzurychx.cloudfront.net/misc/taxi_zones.zip"
    r = requests.get(url, timeout=60)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    z.extractall("/tmp/taxi_zones")

    gdf = gpd.read_file("/tmp/taxi_zones/taxi_zones/taxi_zones.shp").to_crs(epsg=4326)
    gdf["lat"] = gdf.geometry.centroid.y
    gdf["lon"] = gdf.geometry.centroid.x

    centroids = {int(row.LocationID): (float(row.lat), float(row.lon))
                 for _, row in gdf.iterrows()}
    with open(cached, "w") as f:
        json.dump(centroids, f)
    return centroids


# ---------------------------------------------------------------------------
# Target encoding with Bayesian smoothing
# ---------------------------------------------------------------------------

def _smooth(n: int, local: float, global_mean: float) -> float:
    return (n * local + SMOOTHING_K * global_mean) / (n + SMOOTHING_K)


def _compute_zone_stats(df: pd.DataFrame, global_mean: float) -> tuple[dict, dict, dict]:
    """
    Compute smoothed target statistics for zone pairs and individual zones.

    Uses bulk pandas agg() instead of Python for-loops to minimize wall time
    on 37M rows. Time-stratified stats (rush/offpeak/weekend) capture how
    route durations shift across different traffic regimes.
    """
    ts = pd.to_datetime(df["requested_at"])
    hour = ts.dt.hour.values
    dow  = ts.dt.dayofweek.values

    is_weekend = dow >= 5
    is_rush    = (~is_weekend) & (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19)))
    is_offpeak = (~is_rush) & (~is_weekend)

    pu = df["pickup_zone"].values
    do = df["dropoff_zone"].values
    y  = df["duration_seconds"].values

    # Bulk groupby for overall pair stats
    tmp = pd.DataFrame({"pu": pu, "do": do, "y": y,
                        "rush": is_rush, "offpeak": is_offpeak, "weekend": is_weekend})

    overall = tmp.groupby(["pu", "do"])["y"].agg(n="count", mean="mean", p50="median")
    rush_agg    = tmp[tmp["rush"]].groupby(["pu", "do"])["y"].mean().rename("rush_mean")
    offpeak_agg = tmp[tmp["offpeak"]].groupby(["pu", "do"])["y"].mean().rename("offpeak_mean")
    weekend_agg = tmp[tmp["weekend"]].groupby(["pu", "do"])["y"].mean().rename("weekend_mean")

    merged = overall.join(rush_agg, how="left").join(offpeak_agg, how="left").join(weekend_agg, how="left")
    merged.fillna(global_mean, inplace=True)

    pair_stats: dict = {}
    for (pi, di), row in merged.iterrows():
        n = int(row["n"])
        pair_stats[(int(pi), int(di))] = {
            "mean":         _smooth(n, row["mean"],         global_mean),
            "p50":          _smooth(n, row["p50"],          global_mean),
            "rush_mean":    _smooth(n, row["rush_mean"],    global_mean),
            "offpeak_mean": _smooth(n, row["offpeak_mean"], global_mean),
            "weekend_mean": _smooth(n, row["weekend_mean"], global_mean),
        }

    # Zone-level stats (bulk)
    pu_agg = tmp.groupby("pu")["y"].agg(n="count", mean="mean")
    do_agg = tmp.groupby("do")["y"].agg(n="count", mean="mean")

    pu_stats = {int(z): {"mean": _smooth(int(r["n"]), r["mean"], global_mean)}
                for z, r in pu_agg.iterrows()}
    do_stats = {int(z): {"mean": _smooth(int(r["n"]), r["mean"], global_mean)}
                for z, r in do_agg.iterrows()}

    return pair_stats, pu_stats, do_stats


# ---------------------------------------------------------------------------
# Bulk feature builder (training only)
# ---------------------------------------------------------------------------

def _build_features_bulk(
    df: pd.DataFrame,
    zone_centroids: dict,
    zone_pair_stats: dict,
    pu_zone_stats: dict,
    do_zone_stats: dict,
    global_mean: float,
) -> np.ndarray:
    ts = pd.to_datetime(df["requested_at"])
    pu = df["pickup_zone"].values.astype(int)
    do = df["dropoff_zone"].values.astype(int)
    n = len(df)

    hour  = ts.dt.hour.values
    dow   = ts.dt.dayofweek.values
    month = ts.dt.month.values
    doy   = ts.dt.dayofyear.values
    pcount = df["passenger_count"].values.astype(int)

    is_weekend = (dow >= 5).astype(np.float32)
    is_rush    = (((hour >= 7) & (hour <= 9)) | ((hour >= 17) & (hour <= 19))).astype(np.float32)
    is_rush   *= (1 - is_weekend)

    TAU = 2 * np.pi
    hour_sin = np.sin(TAU * hour  / 24).astype(np.float32)
    hour_cos = np.cos(TAU * hour  / 24).astype(np.float32)
    dow_sin  = np.sin(TAU * dow   / 7 ).astype(np.float32)
    dow_cos  = np.cos(TAU * dow   / 7 ).astype(np.float32)
    mth_sin  = np.sin(TAU * month / 12).astype(np.float32)
    mth_cos  = np.cos(TAU * month / 12).astype(np.float32)
    doy_sin  = np.sin(TAU * doy   / 365).astype(np.float32)
    doy_cos  = np.cos(TAU * doy   / 365).astype(np.float32)

    DEFAULT = (DEFAULT_LAT, DEFAULT_LON) = (40.7580, -73.9855)
    pu_lat = np.array([zone_centroids.get(z, DEFAULT)[0] for z in pu], dtype=np.float32)
    pu_lon = np.array([zone_centroids.get(z, DEFAULT)[1] for z in pu], dtype=np.float32)
    do_lat = np.array([zone_centroids.get(z, DEFAULT)[0] for z in do], dtype=np.float32)
    do_lon = np.array([zone_centroids.get(z, DEFAULT)[1] for z in do], dtype=np.float32)

    R = 6371.0
    dlat = np.radians(do_lat - pu_lat)
    dlon = np.radians(do_lon - pu_lon)
    a = (np.sin(dlat / 2) ** 2
         + np.cos(np.radians(pu_lat)) * np.cos(np.radians(do_lat)) * np.sin(dlon / 2) ** 2)
    dist_km = (2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))).astype(np.float32)

    # Build 2D lookup tables for vectorized zone-pair stat retrieval (59x faster
    # than a Python for-loop over 37M rows).
    MZ = 266
    pm_arr = np.full((MZ, MZ), global_mean, np.float32)
    p50_arr = np.full((MZ, MZ), global_mean, np.float32)
    rush_arr = np.full((MZ, MZ), global_mean, np.float32)
    op_arr = np.full((MZ, MZ), global_mean, np.float32)
    wk_arr = np.full((MZ, MZ), global_mean, np.float32)
    for (pi, di), s in zone_pair_stats.items():
        pm_arr[pi, di]   = s.get("mean",         global_mean)
        p50_arr[pi, di]  = s.get("p50",          global_mean)
        rush_arr[pi, di] = s.get("rush_mean",    global_mean)
        op_arr[pi, di]   = s.get("offpeak_mean", global_mean)
        wk_arr[pi, di]   = s.get("weekend_mean", global_mean)

    pu_arr = np.full(MZ, global_mean, np.float32)
    do_arr = np.full(MZ, global_mean, np.float32)
    for z, s in pu_zone_stats.items():
        pu_arr[z] = s.get("mean", global_mean)
    for z, s in do_zone_stats.items():
        do_arr[z] = s.get("mean", global_mean)

    pair_mean    = pm_arr[pu, do]
    pair_p50     = p50_arr[pu, do]
    pair_rush    = rush_arr[pu, do]
    pair_offpeak = op_arr[pu, do]
    pair_weekend = wk_arr[pu, do]
    pu_mean      = pu_arr[pu]
    do_mean      = do_arr[do]

    X = np.column_stack([
        pu, do, pcount,
        hour, dow, month,
        is_weekend, is_rush,
        hour_sin, hour_cos,
        dow_sin, dow_cos,
        mth_sin, mth_cos,
        doy_sin, doy_cos,
        pu_lat, pu_lon, do_lat, do_lon,
        dist_km,
        pair_mean, pair_p50,
        pair_rush, pair_offpeak, pair_weekend,
        pu_mean, do_mean,
    ]).astype(np.float32)
    return X


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true",
                        help="Train on 1M-row sample instead of full dataset")
    args = parser.parse_args()

    train_path = DATA_DIR / ("sample_1M.parquet" if args.sample else "train.parquet")
    dev_path   = DATA_DIR / "dev.parquet"
    for p in (train_path, dev_path):
        if not p.exists():
            raise SystemExit(f"Missing {p.name}. Run `python data/download_data.py` first.")

    print(f"Loading {'sample' if args.sample else 'full'} training data...")
    train = pd.read_parquet(train_path)
    dev   = pd.read_parquet(dev_path)
    print(f"  train: {len(train):,} rows  |  dev: {len(dev):,} rows")

    print("Loading zone centroids...")
    zone_centroids = _fetch_zone_centroids()
    print(f"  {len(zone_centroids)} zones")

    print("Computing zone statistics...")
    global_mean = float(train["duration_seconds"].mean())
    print(f"  global mean: {global_mean:.1f}s")
    pair_stats, pu_stats, do_stats = _compute_zone_stats(train, global_mean)
    print(f"  {len(pair_stats):,} zone pairs")

    print("Building features...")
    t0 = time.time()
    X_train = _build_features_bulk(train, zone_centroids, pair_stats, pu_stats, do_stats, global_mean)
    X_dev   = _build_features_bulk(dev,   zone_centroids, pair_stats, pu_stats, do_stats, global_mean)
    y_train = train["duration_seconds"].values.astype(np.float32)
    y_dev   = dev["duration_seconds"].values.astype(np.float32)
    print(f"  shape: {X_train.shape}  ({time.time()-t0:.0f}s)")

    print("\nTraining LightGBM (L1/MAE objective — matches eval metric)...")
    model = lgb.LGBMRegressor(
        objective="regression_l1",
        n_estimators=3000,
        num_leaves=255,
        learning_rate=0.02,
        min_child_samples=100,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.5,
        n_jobs=-1,
        random_state=42,
        verbose=-1,
    )
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_dev, y_dev)],
        eval_metric="mae",
        callbacks=[
            lgb.early_stopping(stopping_rounds=100, verbose=True),
            lgb.log_evaluation(period=200),
        ],
    )
    print(f"  trained in {time.time()-t0:.0f}s  |  best iter: {model.best_iteration_}")

    train_preds = model.predict(X_train)
    dev_preds   = model.predict(X_dev)
    train_mae = float(np.mean(np.abs(train_preds - y_train)))
    dev_mae   = float(np.mean(np.abs(dev_preds   - y_dev)))

    print(f"\n{'='*42}")
    print(f"Train MAE  : {train_mae:.1f}s")
    print(f"Dev MAE    : {dev_mae:.1f}s")
    print(f"Baseline   : 351.0s")
    print(f"Improvement: {351.0 - dev_mae:+.1f}s")
    print(f"{'='*42}")

    # Feature importance
    fi = sorted(zip(FEATURE_NAMES, model.feature_importances_), key=lambda x: -x[1])
    print("\nTop-10 feature importances:")
    for name, imp in fi[:10]:
        print(f"  {name:<25} {imp:>8,.0f}")

    artifact = {
        "model":           model,
        "zone_centroids":  zone_centroids,
        "zone_pair_stats": pair_stats,
        "pu_zone_stats":   pu_stats,
        "do_zone_stats":   do_stats,
        "global_mean":     global_mean,
        "feature_names":   FEATURE_NAMES,
        "dev_mae":         dev_mae,
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f, protocol=5)
    size_mb = MODEL_PATH.stat().st_size / 1e6
    print(f"\nSaved model.pkl ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
