"""Monthly refit for the live forecaster (runs on the laptop, where the full history is).

    python -m nowcaster.live.train            # writes <data>/live/bundle.pkl.gz
    python -m nowcaster.live.train --upload   # ... and uploads it to the GitHub release `live-models`

Same rule as the backtest: models are fit on every row whose label was published before
the refit (target + 3 h < cutoff), from FULL_WEATHER_START. The bundle holds, per BA:
  m3, m4      fitted M3 (with its B4b coefficients) and M4 range models
  seed        the last SEED_DAYS days of backtest forecasts (M3, B4b, DA, actual) so the
              peak ensembles have a 60-day error library and a 90-day stretch window from
              the first live hour; live history replaces it as it accumulates
  daily_peaks daily peak demand since 2023 (for the yearly high-day line)
"""

from __future__ import annotations

import argparse
import gzip
import pickle
import subprocess
import time
from pathlib import Path

import pandas as pd

from ..config import REPO_ROOT, data_dir, lag_settings, load_bas
from ..features.build import FULL_WEATHER_START, load_eia, load_panel, panel_path
from ..features.panel import clean_hourly, op_day
from ..models.gbm import m3_fit
from ..models.intervals import m4_fit

SEED_DAYS = 160
SEED_COLS = ["ba", "issue_utc", "target_utc", "op_day", "horizon", "da_posted", "da", "B4b", "M3", "actual"]
RELEASE_TAG = "live-models"
BUNDLE_NAME = "bundle.pkl.gz"


def bundle_path() -> Path:
    return data_dir() / "live" / BUNDLE_NAME


def daily_peaks(eia: pd.DataFrame, code: str, tz: str, since="2023-01-01") -> pd.Series:
    act = clean_hourly(eia[eia["ba"] == code])["actual"]
    act = act[act.index >= since]
    return act.groupby(pd.to_datetime(op_day(act.index, tz))).max()


def train_ba(code: str, tz: str, eia: pd.DataFrame, cutoff: pd.Timestamp) -> dict:
    p = load_panel(code)
    lag = pd.Timedelta(hours=int(lag_settings()["demand_available_hours_after_hour_end"]))
    t = pd.DatetimeIndex(p["target_utc"])
    train = p[(t + lag < cutoff) & (t >= pd.Timestamp(FULL_WEATHER_START, tz="UTC"))]
    seed = pd.read_parquet(data_dir() / "backtest" / f"pred_{code}.parquet", columns=SEED_COLS)
    seed = seed[pd.to_datetime(seed["op_day"]) >= pd.to_datetime(seed["op_day"]).max() - pd.Timedelta(days=SEED_DAYS)]
    return {"m3": m3_fit(train), "m4": m4_fit(train), "seed": seed.reset_index(drop=True),
            "daily_peaks": daily_peaks(eia, code, tz), "train_rows": len(train),
            "label_through": str(t[t + lag < cutoff].max())}


def build_bundle(cutoff: pd.Timestamp | None = None) -> dict:
    cutoff = cutoff or pd.Timestamp.now(tz="UTC")
    eia = load_eia()
    out = {"trained_at": str(pd.Timestamp.now(tz="UTC").floor("s")), "cutoff": str(cutoff), "bas": {}}
    for code, ba in load_bas().items():
        if not panel_path(code).exists():
            continue
        t0 = time.time()
        out["bas"][code] = train_ba(code, ba.tz, eia, cutoff)
        print(f"{code}: {out['bas'][code]['train_rows']} training rows, labels to "
              f"{out['bas'][code]['label_through']}, {time.time() - t0:.0f}s", flush=True)
    return out


def save(bundle: dict, path: Path | None = None) -> Path:
    path = path or bundle_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb", compresslevel=6) as f:
        pickle.dump(bundle, f, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def load(path: Path) -> dict:
    with gzip.open(path, "rb") as f:
        return pickle.load(f)


def upload(path: Path) -> None:
    """Replace the bundle on the GitHub release (created on first use)."""
    run = lambda *a: subprocess.run(["gh", *a], cwd=REPO_ROOT, check=True, capture_output=True, text=True)
    try:
        run("release", "view", RELEASE_TAG)
    except subprocess.CalledProcessError:
        run("release", "create", RELEASE_TAG, "--title", "Live models",
            "--notes", "Model bundle for the hourly live forecaster (replaced at each monthly refit).")
    run("release", "upload", RELEASE_TAG, str(path), "--clobber")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--upload", action="store_true")
    a = ap.parse_args(argv)
    path = save(build_bundle())
    print(f"bundle: {path} ({path.stat().st_size / 1e6:.1f} MB)")
    if a.upload:
        upload(path)
        print(f"uploaded to release {RELEASE_TAG}")


if __name__ == "__main__":
    main()
