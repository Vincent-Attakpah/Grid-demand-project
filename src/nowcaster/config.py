"""Configuration: BA definitions from config/bas.yaml, data directory, API keys."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
BAS_YAML = REPO_ROOT / "config" / "bas.yaml"


@dataclass(frozen=True)
class City:
    name: str
    state: str
    lat: float
    lon: float
    pop: int
    active_from: str | None = None   # ISO date; city weighted only from this date (inclusive)
    active_until: str | None = None  # ISO date; city weighted only before this date


@dataclass(frozen=True)
class BA:
    code: str
    name: str
    tz: str
    priority: str = "core"
    cities: tuple[City, ...] = field(default_factory=tuple)
    notes: str = ""


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict:
    with open(path or BAS_YAML, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_bas(include_logging_only: bool = False) -> dict[str, BA]:
    cfg = load_config()
    out: dict[str, BA] = {}
    for code, d in cfg["bas"].items():
        cities = tuple(City(**c) for c in d.get("cities", []))
        out[code] = BA(code=code, name=d["name"], tz=d["tz"], priority=d.get("priority", "core"),
                       cities=cities, notes=d.get("notes", ""))
    if include_logging_only:
        for code, d in cfg.get("logging_only", {}).items():
            out[code] = BA(code=code, name=d["name"], tz=d["tz"], priority="logging_only")
    return out


def weather_settings() -> dict:
    return load_config()["weather"]


def lag_settings() -> dict:
    return load_config()["lags"]


def data_dir() -> Path:
    """Root for local data. Defaults outside OneDrive-synced folders: ~/nowcaster-data."""
    p = Path(os.environ.get("NOWCASTER_DATA", Path.home() / "nowcaster-data"))
    p.mkdir(parents=True, exist_ok=True)
    return p


def eia_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY is not set (get a free key at https://www.eia.gov/opendata/).")
    return key
