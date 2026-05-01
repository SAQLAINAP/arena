# CLAUDE.md — ETA Challenge Implementation Notes

## Project Structure

```
eta-challenge-starter/
├── train.py              ← Full training pipeline (run this to reproduce model.pkl)
├── predict.py            ← Inference interface; predict(request: dict) -> float
├── features.py           ← Shared feature definitions (imported by both)
├── grade.py              ← Grader harness (provided by Gobblecube)
├── baseline.py           ← Original XGBoost baseline (kept for reference)
├── Dockerfile            ← Inference-only image
├── requirements.txt      ← Inference dependencies only
├── model.pkl             ← Trained model + all lookup tables
├── data/
│   ├── download_data.py  ← One-time download of 2023 TLC data
│   ├── zone_centroids.json  ← Cached zone lat/lon centroids
│   └── schema.md         ← Input schema docs
└── tests/
    └── test_submission.py ← Interface smoke tests
```

## What model.pkl Contains

`model.pkl` is a Python dict (pickled with protocol 5) with:
- `"model"`: trained `LGBMRegressor` (L1/MAE objective)
- `"zone_centroids"`: `{zone_id: (lat, lon)}` for 263 NYC taxi zones
- `"zone_pair_stats"`: `{(pu, do): {"mean", "p50", "rush_mean", "offpeak_mean", "weekend_mean"}}`
- `"pu_zone_stats"`: `{zone_id: {"mean"}}` — average duration from each pickup zone
- `"do_zone_stats"`: `{zone_id: {"mean"}}` — average duration to each dropoff zone
- `"global_mean"`: fallback for unseen zone pairs
- `"feature_names"`: list of 28 feature names in column order
- `"dev_mae"`: MAE on dev set at time of training

## Feature Engineering (28 features)

All features are derived from the 4 inference-time inputs:
`pickup_zone`, `dropoff_zone`, `requested_at`, `passenger_count`

**Temporal (14 features)**:
`hour`, `dow`, `month`, `is_weekend`, `is_rush_hour`
`hour_sin/cos`, `dow_sin/cos`, `month_sin/cos`, `doy_sin/cos`

**Spatial (5 features)**:
`pu_lat`, `pu_lon`, `do_lat`, `do_lon`, `haversine_km`

**Zone pair target encoding (5 features)**:
`pair_mean`, `pair_p50`, `pair_rush_mean`, `pair_offpeak_mean`, `pair_weekend_mean`

**Zone-level target encoding (3 features)**:
`pu_mean`, `do_mean`, `passenger_count`
(passenger_count is the 3rd in the spatial group)

## Training

```bash
python train.py [--sample]
```

1. Fetches TLC zone shapefile → computes centroids → `data/zone_centroids.json`
2. Computes Bayesian-smoothed target stats for all zone pairs
3. Builds 28-feature matrix (vectorized NumPy — avoids 37M-row Python loops)
4. Trains LightGBM with `objective=regression_l1` (MAE loss = matches eval metric)
5. Early stopping on dev MAE (patience=100 rounds)
6. Saves model + all lookup tables to `model.pkl`

## Key Design Decisions

**L1 objective**: Training with MAE loss instead of default MSE. MSE optimizes for the conditional mean (skewed by outlier trips). MAE optimizes for the conditional median, which directly minimizes the eval metric. Improvement: −19s MAE.

**Bayesian smoothing** (K=50): Zone pairs with few observations are regularized toward the global mean. Prevents the model from memorizing rare routes with noisy estimates.

**Zone centroid from shapefile**: Using actual polygon centroids (not random points). Cached after first download to avoid repeat network calls.

**Vectorized lookup tables**: Feature building uses 2D NumPy arrays `pair_arr[pu, do]` instead of Python dict loops, giving 59x speedup on 37M rows.

## Inference

`predict(request)` in `predict.py`:
1. Parse `requested_at` → extract hour, dow, month, day_of_year
2. Look up zone centroids → compute haversine distance (pure math, O(1))
3. Look up zone pair stats from in-memory dict (O(1))
4. Call `model.predict(1×28 array)` → return float

Inference is fully offline. No network calls. Runtime: ~0.5ms per prediction.
