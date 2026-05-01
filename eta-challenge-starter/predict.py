"""Submission interface — Gobblecube grader calls predict() once per request.

The signature is fixed. Everything inside (model type, features) is ours.
"""
from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

import numpy as np

from features import build_single

_MODEL_PATH = Path(__file__).parent / "model.pkl"

with open(_MODEL_PATH, "rb") as _f:
    _ART = pickle.load(_f)

_MODEL           = _ART["model"]
_ZONE_CENTROIDS  = _ART["zone_centroids"]
_PAIR_STATS      = _ART["zone_pair_stats"]
_PU_STATS        = _ART["pu_zone_stats"]
_DO_STATS        = _ART["do_zone_stats"]
_GLOBAL_MEAN     = _ART["global_mean"]


def predict(request: dict) -> float:
    """Predict trip duration in seconds.

    Input schema:
        {
            "pickup_zone":     int,
            "dropoff_zone":    int,
            "requested_at":    str,   # ISO 8601
            "passenger_count": int,
        }
    """
    ts = datetime.fromisoformat(request["requested_at"])
    x = build_single(
        pickup_zone=int(request["pickup_zone"]),
        dropoff_zone=int(request["dropoff_zone"]),
        hour=ts.hour,
        dow=ts.weekday(),
        month=ts.month,
        day_of_year=ts.timetuple().tm_yday,
        passenger_count=int(request["passenger_count"]),
        zone_centroids=_ZONE_CENTROIDS,
        zone_pair_stats=_PAIR_STATS,
        pu_zone_stats=_PU_STATS,
        do_zone_stats=_DO_STATS,
        global_mean=_GLOBAL_MEAN,
    )
    return float(_MODEL.predict(x)[0])
