# Your Submission: Writeup Template

---

## Your final score

Dev MAE: **264.4 s** (grade.py 50k sample) — baseline was 351.0 s

---

## Your approach, in one paragraph

LightGBM regressor (174 trees, 255 leaves, lr=0.02) trained on 36.7M NYC TLC 2023 yellow taxi trips. The main feature engineering moves were: (1) zone-pair target encoding — precomputing mean and median trip duration for every (pickup_zone, dropoff_zone) pair from training data, with Bayesian smoothing to handle sparse pairs; (2) time-stratified variants of those stats (rush-hour mean, off-peak mean, weekend mean) so the model sees that JFK→Midtown at 8am ≠ 2am Saturday; (3) haversine distance between zone polygon centroids downloaded from the TLC shapefile — the best available distance proxy since actual trip distance is not in the schema; (4) cyclical sin/cos encoding of hour, day-of-week, month, and day-of-year to avoid artificial discontinuities at temporal boundaries. The single biggest algorithmic change was switching from MSE to MAE loss (L1 objective): MSE trains toward the conditional mean, pulled up by outlier long trips; MAE trains toward the conditional median, which is exactly what's being measured. That one switch gave ~19 s improvement with no feature changes.

---

## What you tried that didn't work

**More trees**: Early stopping triggered at ~170 iterations (of 3000 max). Training 300 trees without early stopping gave 281.7 s — marginally worse than 280.9 s with early stopping. The dev set covers Dec 18-31 (Christmas/New Year), which has very different traffic patterns from the rest of 2023 training data. The model can't learn holiday-specific patterns, so validation loss stops improving quickly regardless of tree count. More trees just overfit to non-holiday patterns.

**`same_zone` feature**: Built a boolean for pickup_zone == dropoff_zone. Zero gain importance — entirely captured by haversine_km ≈ 0 and the zone-pair stats already encoding same-zone trips. Dropped it.

**Split-count feature importance as a guide**: The standard LightGBM feature importance (split count) showed temporal features (hour, month, dow) dominating. Took this at face value initially and focused more on temporal engineering. Checking gain-based importance later showed the real picture: pair_p50 was #1 by a factor of 3x over pair_mean, and haversine_km was #4 — spatial features had far higher value per split even though they needed fewer splits total. Misleading metric.

---

## Where AI tooling sped you up most

Claude Code (this entire solution was built with it in agentic mode). Biggest speedups:

- **Feature engineering iteration**: Describing the target encoding + Bayesian smoothing approach in the prompt and having it generate the correct vectorized NumPy 2D lookup implementation immediately, rather than writing and debugging the Python for-loop first. The 59x speedup catch (Python loop vs NumPy 2D index on 37M rows) was flagged and fixed mid-session.
- **Gain vs split importance insight**: Claude identified that split-count importance was misleading and ran the gain-based check, which completely changed the interpretation of which features mattered.
- **L1 vs L2 objective**: The "match your train loss to your eval metric" insight was surfaced proactively — I hadn't planned to test it.

Where it fell short: Claude can't run Docker locally (not installed on the machine), so the Dockerfile was verified by inspection only. It also can't make network calls to external routing APIs (OSRM) to get actual road distances between zone centroids, which would likely be the next biggest MAE gain.

---

## Next experiments

Road distance from zone centroids via OSRM or a pre-built distance matrix. Haversine assumes a sphere and ignores road topology — river crossings, bridge bottlenecks, and tunnel access patterns are invisible to it. A one-time pre-computation of road distances between all 263×263 zone centroid pairs would replace haversine with something the model could actually trust. My estimate: another 15-25 s MAE reduction.

---

## How to reproduce

```bash
# 1. Install deps
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install lightgbm geopandas requests  # training only

# 2. Download 2023 TLC data (~500 MB, ~5 min)
python data/download_data.py

# 3. Train (downloads zone shapefile on first run, ~15 min on laptop)
python train.py

# 4. Verify
python -m pytest tests/ -v
python grade.py
```

---

*Total time spent on this challenge: ~6 hours.*
