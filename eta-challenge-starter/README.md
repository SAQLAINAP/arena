# ETA Challenge — NYC Taxi Trip Duration Prediction

Predict NYC yellow taxi trip duration (seconds) given pickup zone, dropoff zone, request time, and passenger count.

**Val MAE: 264s | Baseline: 351s | Improvement: 87s (25%)**

---

## What I Built

A LightGBM regressor trained on 37M NYC TLC yellow taxi trips from 2023. The core improvement over the baseline was replacing 6 weak features with 28 engineered features — primarily zone-pair target encoding, haversine distance from zone centroids, and time-stratified route statistics. The model is serialized as a single `model.pkl` containing the trained model plus all lookup tables needed for offline inference.

---

## How to Run

### Prerequisites
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install lightgbm geopandas  # training only; not required at inference
```

### Download data (one-time, ~500 MB)
```bash
python data/download_data.py
```

### Train
```bash
python train.py          # full 37M rows, ~15 min
python train.py --sample # 1M-row sample, ~2 min
```

### Predict (local grading)
```bash
python grade.py                              # dev set, 50k sample
python grade.py data/dev.parquet out.csv     # full dev set
```

### Run tests
```bash
python -m pytest tests/ -v
```

### Docker
```bash
docker build -t eta-submission .
docker run --rm -v $(pwd)/data:/work eta-submission /work/dev.parquet /work/out.csv
```

---

## My Approach

**1. Data download & baseline audit**  
The starter kit had an XGBoost baseline at 351s MAE using 6 features. I ran it first to confirm the benchmark.

**2. Zone centroids from TLC shapefile**  
Downloaded `taxi_zones.zip` from TLC, extracted with geopandas, computed polygon centroids in WGS84. This gives each zone a (lat, lon) for distance computation. Cached to `data/zone_centroids.json`.

**3. Target encoding — zone pair statistics**  
For each (pickup_zone, dropoff_zone) pair, computed mean and median trip duration from training data. Applied Bayesian smoothing to handle sparse pairs:  
`smoothed = (n × local + 50 × global_mean) / (n + 50)`  
Also computed time-stratified variants: `rush_mean`, `offpeak_mean`, `weekend_mean` — because JFK→Midtown at 8am weekday is fundamentally different from 2am Saturday.

**4. Haversine distance**  
Without trip distance in the schema, the great-circle distance between zone centroids is the best continuous distance proxy. It's the #4 most impactful feature by gain importance.

**5. Cyclical temporal encoding**  
`hour_sin = sin(2π × hour / 24)`, `hour_cos = cos(2π × hour / 24)`, similarly for day-of-week, month, and day-of-year. Prevents tree models from treating hour=23 and hour=0 as maximally different when they're temporally adjacent.

**6. L1 (MAE) objective**  
The single biggest algorithmic improvement: switched LightGBM from MSE loss to MAE loss. MSE minimizes to the conditional mean (pulled up by outlier trips). MAE minimizes to the conditional median, which matches the evaluation metric exactly.

---

## What Worked

| Change | Dev MAE | Delta |
|--------|---------|-------|
| Baseline (XGBoost, 6 features) | 351.0s | — |
| LightGBM + target encoding + haversine (L2) | 282.3s | −68.7s |
| + Day-of-year cyclical + time-stratified zone stats | 280.9s | −1.4s |
| **+ L1/MAE objective** | **264.4s** | **−16.5s** |

Most impactful features by gain importance:
1. `pair_p50` — zone pair median duration (238T gain)
2. `pair_mean` — zone pair mean duration (89T gain)
3. `pair_offpeak_mean` — off-peak route stat (32T gain)
4. `haversine_km` — straight-line distance (25T gain)

The split-count importance was misleading — temporal features topped it because the model makes many fine-grained time-of-day splits. Gain importance correctly showed each spatial split is far more valuable.

---

## What Didn't Work

**More trees**: Early stopping triggered at ~170 iterations (of 3000 max). Training 300 trees without early stopping gave 281.7s — marginally worse than 280.9s. The dev set (Dec 18-31 holiday period) diverges from training distribution quickly.

**`same_zone` feature**: Zero gain importance — entirely redundant with haversine_km≈0 and zone pair stats already capturing same-zone trips. Dropped.

---

## What I'd Try Next

1. **Road distance from zone centroids**: OSRM or a road network distance (at training time only) to replace haversine with actual driving distance. River crossings and bridge availability make road/crow-flies ratio vary significantly by zone pair.

2. **Zone area features**: Large zones (JFK covers many terminals) have higher within-zone pickup variance than small Manhattan blocks. Zone polygon area from the shapefile adds geometric uncertainty.

3. **Weather data**: NOAA Central Park / JFK / LGA hourly observations. Rain typically adds 10-20% to NYC taxi duration.

4. **Model stacking**: Train a meta-model on out-of-fold predictions from LightGBM + XGBoost + a simple zone-pair lookup. Stacking typically gives 5-10s additional improvement.

5. **Temporal embeddings**: Learn zone representations capturing borough-level traffic structure rather than raw integer IDs.

---

## Score

| Set | MAE |
|-----|-----|
| Dev (Dec 18-31 2023, 50k sample via grade.py) | 264.4s |
| Baseline | 351.0s |
| Improvement | 86.6s (25%) |

Note: The dev set covers the Christmas/New Year holiday period. The held-out 2024 eval set, if drawn from a more representative distribution, should show similar or better performance.

---

## Submission

- LinkedIn: [Saqlain Ahmed](https://www.linkedin.com/in/saqlain-ahmed-p)
