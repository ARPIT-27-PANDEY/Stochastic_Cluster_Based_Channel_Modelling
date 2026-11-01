"""Loading and validation of the model parameters (Table I) and simulation settings.

All numerical constants of the model live in ``config/channel_parameters.yaml``;
this module only parses them into typed containers so that the rest of the code
never hardcodes a paper value.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"

# Cluster classes used throughout the code (Sec. V: three separately modelled cases)
LOS_CLUSTER = 0   #: the dominant LoS cluster
LOS = 1           #: any other cluster in a LoS channel
NLOS = 2          #: any cluster in an NLoS channel
CLASS_KEYS = {LOS_CLUSTER: "los_cluster", LOS: "los", NLOS: "nlos"}


@dataclass(frozen=True)
class GammaParams:
    shape: float
    scale: float

    def __post_init__(self) -> None:
        if self.shape <= 0 or self.scale <= 0:
            raise ValueError(f"Gamma parameters must be positive, got {self}")

    @property
    def mean(self) -> float:
        return self.shape * self.scale


@dataclass(frozen=True)
class LogNormalParams:
    mu: float
    sigma: float

    def __post_init__(self) -> None:
        if self.sigma <= 0:
            raise ValueError(f"log-normal sigma must be positive, got {self}")

    @property
    def median(self) -> float:
        return float(np.exp(self.mu))

    @property
    def mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2))


@dataclass(frozen=True)
class AngleModel:
    """Centroid model: 'fixed', 'uniform' or 'bimodal_laplace'."""

    type: str
    value: float = 0.0
    weights: tuple[float, ...] = ()
    centers: tuple[float, ...] = ()
    decays: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        if self.type not in {"fixed", "uniform", "bimodal_laplace"}:
            raise ValueError(f"unknown angle model {self.type}")
        if self.type == "bimodal_laplace":
            if not (len(self.weights) == len(self.centers) == len(self.decays)):
                raise ValueError("bimodal Laplace needs equal-length weights/centers/decays")
            if abs(sum(self.weights) - 1.0) > 1e-6:
                raise ValueError("bimodal Laplace weights must sum to one")


@dataclass(frozen=True)
class PowerParams:
    """NCP[dB] ~ N(alpha + 10 beta log10(r), (rho + 10 eta log10(r))^2), r = d/d_LoS."""

    alpha: float
    beta: float
    rho: float
    eta: float


@dataclass
class ChannelParameters:
    """Typed view of ``channel_parameters.yaml``."""

    raw: dict[str, Any]
    fc_hz: float
    c: float
    dynamic_range_db: float
    n_los_additional: GammaParams
    n_nlos_minus_one: GammaParams
    arrival: dict[str, GammaParams]
    delay_spread: dict[str, LogNormalParams]
    doa_spread: dict[str, LogNormalParams]
    dod_spread: dict[str, LogNormalParams]
    doa_centroid: dict[str, AngleModel]
    dod_centroid: dict[str, AngleModel]
    power: dict[str, PowerParams]
    min_sigma_db: float
    bimodal_laplace_mode: str = "truncated"
    arrival_source: str = "table"
    notes: list[str] = field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _angle_model(d: dict[str, Any]) -> AngleModel:
    return AngleModel(
        type=d["type"],
        value=float(d.get("value", 0.0)),
        weights=tuple(float(x) for x in d.get("weights", ())),
        centers=tuple(float(x) for x in d.get("centers", ())),
        decays=tuple(float(x) for x in d.get("decays", ())),
    )


def load_channel_parameters(path: str | Path | None = None, arrival_source: str | None = None,
                            bimodal_laplace_mode: str | None = None) -> ChannelParameters:
    """Parse the parameter file. ``arrival_source`` overrides 'table'/'text' selection."""
    path = Path(path) if path is not None else CONFIG_DIR / "channel_parameters.yaml"
    raw = _load_yaml(path)
    src = arrival_source or raw["arrival_time_m"]["source"]
    if src not in ("table", "text"):
        raise ValueError("arrival_source must be 'table' or 'text'")
    arr = raw["arrival_time_m"][src]
    ln = lambda block: {k: LogNormalParams(float(v["mu"]), float(v["sigma"])) for k, v in block.items()}
    return ChannelParameters(
        raw=raw,
        fc_hz=float(raw["measurement"]["carrier_frequency_hz"]),
        c=float(raw["speed_of_light_m_s"]),
        dynamic_range_db=float(raw["measurement"]["dynamic_range_db"]),
        n_los_additional=GammaParams(**{k: float(v) for k, v in raw["cluster_number"]["los_additional"].items()}),
        n_nlos_minus_one=GammaParams(**{k: float(v) for k, v in raw["cluster_number"]["nlos_minus_one"].items()}),
        arrival={k: GammaParams(float(v["shape"]), float(v["scale"])) for k, v in arr.items()},
        delay_spread=ln(raw["delay_spread_m"]),
        doa_spread=ln(raw["doa_spread"]),
        dod_spread=ln(raw["dod_spread"]),
        doa_centroid={k: _angle_model(v) for k, v in raw["doa_centroid_rad"].items()},
        dod_centroid={k: _angle_model(v) for k, v in raw["dod_centroid_rad"].items()},
        power={k: PowerParams(**{kk: float(vv) for kk, vv in v.items()}) for k, v in raw["cluster_power"].items() if isinstance(v, dict)},
        min_sigma_db=float(raw["cluster_power"].get("min_sigma_db", 0.1)),
        bimodal_laplace_mode=bimodal_laplace_mode or raw.get("bimodal_laplace_mode", "truncated"),
        arrival_source=src,
    )


def load_simulation_config(path: str | Path | None = None) -> dict[str, Any]:
    path = Path(path) if path is not None else CONFIG_DIR / "simulation_config.yaml"
    return _load_yaml(path)


def get_profile(name: str, cfg: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = cfg or load_simulation_config()
    if name not in cfg["profiles"]:
        raise KeyError(f"unknown profile '{name}', choose from {list(cfg['profiles'])}")
    return dict(cfg["profiles"][name])
