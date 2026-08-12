"""Typed, reproducible configuration for the complete VDP pipeline."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .trip_builder import TripBuildConfig


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name, {}) or {}
    if not isinstance(value, Mapping):
        raise ValueError(f"Configuration section {name!r} must be a mapping")
    return value


@dataclass(frozen=True)
class DataConfig:
    raw_data: str = "data/sample/raw_data.csv"
    work_dir: str = "artifacts/current"
    trips_file: str = "trips.csv"
    points_file: str = "trip_points.csv"


@dataclass(frozen=True)
class DatasetConfig:
    prefix_fractions: tuple[float, ...] = (0.0, 0.25, 0.5)
    validation_vin_fraction: float = 0.15
    test_vin_fraction: float = 0.15
    split_seed: int = 42
    grid_cell_size_degrees: float = 0.01
    max_history_trips: int = 20
    max_trajectory_points: int = 100


@dataclass(frozen=True)
class BaselineConfig:
    n_estimators: int = 160
    min_samples_leaf: int = 1
    max_depth: int | None = None
    n_jobs: int = -1


@dataclass(frozen=True)
class KerasConfig:
    encoder: str = "GRU"
    recurrent_units: int = 64
    recurrent_layers: int = 1
    dense_units: tuple[int, ...] = (128, 64)
    dropout: float = 0.25
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 60
    early_stopping_patience: int = 8


@dataclass(frozen=True)
class EvaluationConfig:
    top_k: tuple[int, ...] = (1, 3, 5)
    recall_distance_km: tuple[float, ...] = (1.0, 5.0, 10.0, 25.0)


@dataclass(frozen=True)
class AppConfig:
    config_version: int = 1
    project_name: str = "vehicle-destination-prediction"
    random_seed: int = 42
    data: DataConfig = field(default_factory=DataConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    baseline: BaselineConfig = field(default_factory=BaselineConfig)
    keras: KerasConfig = field(default_factory=KerasConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    trip_builder: TripBuildConfig = field(default_factory=TripBuildConfig)
    project_root: Path = field(default_factory=Path, compare=False, repr=False)

    def validate(self) -> None:
        if self.config_version != 1:
            raise ValueError("Only config_version: 1 is supported")
        if not self.project_name.strip():
            raise ValueError("project_name cannot be empty")
        fractions = self.dataset.prefix_fractions
        if not fractions or any(value < 0 or value >= 1 for value in fractions):
            raise ValueError("dataset.prefix_fractions must be in [0, 1)")
        if len(set(fractions)) != len(fractions):
            raise ValueError("dataset.prefix_fractions cannot contain duplicates")
        if not 0 < self.dataset.validation_vin_fraction < 0.5:
            raise ValueError("dataset.validation_vin_fraction must be in (0, 0.5)")
        if not 0 < self.dataset.test_vin_fraction < 0.5:
            raise ValueError("dataset.test_vin_fraction must be in (0, 0.5)")
        if (
            self.dataset.validation_vin_fraction
            + self.dataset.test_vin_fraction
            >= 0.8
        ):
            raise ValueError("validation and test VIN fractions leave too little training data")
        if self.dataset.grid_cell_size_degrees <= 0:
            raise ValueError("dataset.grid_cell_size_degrees must be positive")
        if self.dataset.max_history_trips < 0:
            raise ValueError("dataset.max_history_trips must be non-negative")
        if self.dataset.max_trajectory_points < 2:
            raise ValueError("dataset.max_trajectory_points must be at least 2")
        if self.baseline.n_estimators < 10:
            raise ValueError("baseline.n_estimators must be at least 10")
        if self.keras.encoder not in {
            "GRU",
            "LSTM",
            "BidirectionalGRU",
            "BidirectionalLSTM",
        }:
            raise ValueError("keras.encoder is not supported")
        if self.keras.recurrent_units <= 0 or self.keras.recurrent_layers <= 0:
            raise ValueError("Keras recurrent sizes must be positive")
        if not 0 <= self.keras.dropout < 1:
            raise ValueError("keras.dropout must be in [0, 1)")
        if self.keras.learning_rate <= 0:
            raise ValueError("keras.learning_rate must be positive")
        if not self.evaluation.top_k or min(self.evaluation.top_k) < 1:
            raise ValueError("evaluation.top_k values must be positive")
        self.trip_builder.validate()

    def path(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.project_root / path).resolve()

    def to_dict(self, *, include_runtime: bool = False) -> dict[str, Any]:
        payload = asdict(self)
        payload.pop("project_root", None)
        if include_runtime:
            payload["project_root"] = str(self.project_root)
        return payload

    @property
    def reproducibility_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _merge_dataclass(cls: type, values: Mapping[str, Any]):
    allowed = set(cls.__dataclass_fields__)
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unknown {cls.__name__} settings: {unknown}")
    converted = dict(values)
    for name in ("prefix_fractions", "dense_units", "top_k", "recall_distance_km"):
        if name in converted:
            converted[name] = tuple(converted[name])
    return cls(**converted)


def load_config(path: str | Path | None = None) -> AppConfig:
    """Load YAML settings and resolve paths relative to the project root."""
    if path is None:
        project_root = Path.cwd().resolve()
        raw: Mapping[str, Any] = {}
    else:
        config_path = Path(path).expanduser().resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, Mapping):
            raise ValueError("The configuration root must be a mapping")
        raw = loaded
        project_root = config_path.parent.parent.resolve()

    known = {
        "config_version",
        "project_name",
        "random_seed",
        "data",
        "dataset",
        "baseline",
        "keras",
        "evaluation",
        "trip_builder",
    }
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"Unknown top-level configuration settings: {unknown}")

    config = AppConfig(
        config_version=int(raw.get("config_version", 1)),
        project_name=str(raw.get("project_name", AppConfig.project_name)),
        random_seed=int(raw.get("random_seed", AppConfig.random_seed)),
        data=_merge_dataclass(DataConfig, _section(raw, "data")),
        dataset=_merge_dataclass(DatasetConfig, _section(raw, "dataset")),
        baseline=_merge_dataclass(BaselineConfig, _section(raw, "baseline")),
        keras=_merge_dataclass(KerasConfig, _section(raw, "keras")),
        evaluation=_merge_dataclass(EvaluationConfig, _section(raw, "evaluation")),
        trip_builder=TripBuildConfig.from_mapping(_section(raw, "trip_builder")),
        project_root=project_root,
    )
    config.validate()
    return config


def save_resolved_config(config: AppConfig, path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = config.to_dict(include_runtime=True)
    payload["reproducibility_hash"] = config.reproducibility_hash
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(payload, handle, sort_keys=False)
    temporary.replace(target)
    return target
