"""Shared feature engineering — used by both train.py and predict.py."""
from __future__ import annotations

from math import asin, cos, pi, radians, sin, sqrt

import numpy as np

TAU = 2 * pi
DEFAULT_LAT, DEFAULT_LON = 40.7580, -73.9855  # Times Square fallback

FEATURE_NAMES = [
    "pickup_zone", "dropoff_zone", "passenger_count",
    "hour", "dow", "month",
    "is_weekend", "is_rush_hour",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "month_sin", "month_cos",
    "doy_sin", "doy_cos",
    "pu_lat", "pu_lon", "do_lat", "do_lon",
    "haversine_km",
    "pair_mean", "pair_p50",
    "pair_rush_mean", "pair_offpeak_mean", "pair_weekend_mean",
    "pu_mean", "do_mean",
]


def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km."""
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    return 2 * R * asin(sqrt(max(0.0, min(1.0, a))))


def build_single(
    pickup_zone: int,
    dropoff_zone: int,
    hour: int,
    dow: int,
    month: int,
    day_of_year: int,
    passenger_count: int,
    zone_centroids: dict,
    zone_pair_stats: dict,
    pu_zone_stats: dict,
    do_zone_stats: dict,
    global_mean: float,
) -> np.ndarray:
    """Return a 1×N feature array for one inference request."""
    pu_lat, pu_lon = zone_centroids.get(pickup_zone, (DEFAULT_LAT, DEFAULT_LON))
    do_lat, do_lon = zone_centroids.get(dropoff_zone, (DEFAULT_LAT, DEFAULT_LON))

    dist_km = _haversine(pu_lat, pu_lon, do_lat, do_lon)

    is_weekend = int(dow >= 5)
    is_rush = int(not is_weekend and ((7 <= hour <= 9) or (17 <= hour <= 19)))

    pair = zone_pair_stats.get((pickup_zone, dropoff_zone), {})
    pu_s = pu_zone_stats.get(pickup_zone, {})
    do_s = do_zone_stats.get(dropoff_zone, {})

    x = np.array([[
        pickup_zone, dropoff_zone, passenger_count,
        hour, dow, month,
        is_weekend, is_rush,
        sin(TAU * hour / 24), cos(TAU * hour / 24),
        sin(TAU * dow / 7),  cos(TAU * dow / 7),
        sin(TAU * month / 12), cos(TAU * month / 12),
        sin(TAU * day_of_year / 365), cos(TAU * day_of_year / 365),
        pu_lat, pu_lon, do_lat, do_lon,
        dist_km,
        pair.get("mean", global_mean),
        pair.get("p50", global_mean),
        pair.get("rush_mean", global_mean),
        pair.get("offpeak_mean", global_mean),
        pair.get("weekend_mean", global_mean),
        pu_s.get("mean", global_mean),
        do_s.get("mean", global_mean),
    ]], dtype=np.float32)
    return x
