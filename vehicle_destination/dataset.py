"""Leakage-safe sample construction and VIN-disjoint dataset splitting."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .config import DatasetConfig
from .geo import GridEncoder, bearing_degrees, haversine_km, path_distance_km


FEATURE_COLUMNS = (
    "origin_latitude",
    "origin_longitude",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "weekend",
    "prefix_fraction",
    "prefix_last_latitude",
    "prefix_last_longitude",
    "prefix_displacement_km",
    "prefix_path_km",
    "prefix_bearing_sin",
    "prefix_bearing_cos",
    "history_count",
    "last_destination_latitude",
    "last_destination_longitude",
    "mean_destination_latitude",
    "mean_destination_longitude",
)


@dataclass(frozen=True)
class SplitManifest:
    train_vins: tuple[str, ...]
    validation_vins: tuple[str, ...]
    test_vins: tuple[str, ...]
    seed: int

    def split_for(self, vin: str) -> str:
        if vin in self.train_vins:
            return "train"
        if vin in self.validation_vins:
            return "validation"
        if vin in self.test_vins:
            return "test"
        raise KeyError(f"VIN {vin!r} is absent from the split manifest")

    def to_dict(self) -> dict[str, object]:
        return {
            "train_vins": list(self.train_vins),
            "validation_vins": list(self.validation_vins),
            "test_vins": list(self.test_vins),
            "seed": self.seed,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, object]) -> "SplitManifest":
        return cls(
            train_vins=tuple(str(value) for value in raw["train_vins"]),
            validation_vins=tuple(str(value) for value in raw["validation_vins"]),
            test_vins=tuple(str(value) for value in raw["test_vins"]),
            seed=int(raw["seed"]),
        )


def pseudonymize_identifier(value: str) -> str:
    return "vehicle-" + hashlib.sha256(value.encode("utf-8")).hexdigest()[:10]


def parse_trajectory(value: object) -> list[list[float]]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) < 2:
        raise ValueError("Trajectory must contain at least two points")
    result: list[list[float]] = []
    for point in parsed:
        if not isinstance(point, Sequence) or len(point) < 2:
            raise ValueError("Every trajectory point must contain latitude and longitude")
        latitude, longitude = float(point[0]), float(point[1])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Trajectory contains an invalid coordinate")
        result.append([latitude, longitude])
    return result


def make_vin_split(vins: Iterable[str], config: DatasetConfig) -> SplitManifest:
    """Create deterministic, fully VIN-disjoint train/validation/test partitions."""
    unique = np.asarray(sorted({str(value) for value in vins}), dtype=object)
    if len(unique) < 3:
        raise ValueError("At least three VINs are required for VIN-disjoint splitting")
    rng = np.random.default_rng(config.split_seed)
    shuffled = unique[rng.permutation(len(unique))]
    n_test = max(1, int(round(len(unique) * config.test_vin_fraction)))
    n_validation = max(1, int(round(len(unique) * config.validation_vin_fraction)))
    if n_test + n_validation >= len(unique):
        n_validation = 1
        n_test = 1
    test_vins = tuple(sorted(str(value) for value in shuffled[:n_test]))
    validation_vins = tuple(
        sorted(str(value) for value in shuffled[n_test : n_test + n_validation])
    )
    train_vins = tuple(sorted(str(value) for value in shuffled[n_test + n_validation :]))
    manifest = SplitManifest(train_vins, validation_vins, test_vins, config.split_seed)
    assert not (set(train_vins) & set(validation_vins))
    assert not (set(train_vins) & set(test_vins))
    assert not (set(validation_vins) & set(test_vins))
    return manifest


def observed_prefix_points(
    trajectory: list[list[float]], fraction: float
) -> list[list[float]]:
    """Return only observed points and always exclude the final destination point."""
    if not 0 <= fraction < 1:
        raise ValueError("prefix fraction must be in [0, 1)")
    maximum = max(1, len(trajectory) - 1)
    count = 1 if fraction == 0 else max(1, int(math.ceil(maximum * fraction)))
    return trajectory[: min(count, maximum)]


def features_from_observation(
    *,
    start: pd.Timestamp,
    prefix: Sequence[Sequence[float]],
    prefix_fraction: float,
    history: Sequence[dict[str, float]],
) -> dict[str, float]:
    """Build the stable feature contract from information known at inference time.

    This public helper is shared by training-sample construction and editable
    VIN/scenario inference. It deliberately accepts an observed prefix instead
    of a complete target trajectory, which makes destination leakage harder to
    introduce in downstream interfaces.
    """
    if not 0 <= float(prefix_fraction) < 1:
        raise ValueError("prefix fraction must be in [0, 1)")
    if not prefix:
        raise ValueError("Observed trajectory prefix must contain at least one point")
    normalized_prefix = [[float(point[0]), float(point[1])] for point in prefix]
    for latitude, longitude in normalized_prefix:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Observed trajectory prefix contains an invalid coordinate")
    origin = normalized_prefix[0]
    last = normalized_prefix[-1]
    bearing = bearing_degrees(origin, last) if len(normalized_prefix) > 1 else 0.0
    history_destinations = list(history)[-20:]
    if history_destinations:
        last_history = history_destinations[-1]
        mean_latitude = float(
            np.mean([item["latitude"] for item in history_destinations])
        )
        mean_longitude = float(
            np.mean([item["longitude"] for item in history_destinations])
        )
    else:
        last_history = {"latitude": origin[0], "longitude": origin[1]}
        mean_latitude, mean_longitude = origin
    start = pd.Timestamp(start)
    hour = start.hour + start.minute / 60.0
    weekday = start.dayofweek
    month = start.month - 1
    return {
        "origin_latitude": origin[0],
        "origin_longitude": origin[1],
        "hour_sin": math.sin(2 * math.pi * hour / 24),
        "hour_cos": math.cos(2 * math.pi * hour / 24),
        "weekday_sin": math.sin(2 * math.pi * weekday / 7),
        "weekday_cos": math.cos(2 * math.pi * weekday / 7),
        "month_sin": math.sin(2 * math.pi * month / 12),
        "month_cos": math.cos(2 * math.pi * month / 12),
        "weekend": float(weekday >= 5),
        "prefix_fraction": float(prefix_fraction),
        "prefix_last_latitude": last[0],
        "prefix_last_longitude": last[1],
        "prefix_displacement_km": haversine_km(
            origin[0], origin[1], last[0], last[1]
        ),
        "prefix_path_km": path_distance_km(normalized_prefix),
        "prefix_bearing_sin": math.sin(math.radians(bearing)),
        "prefix_bearing_cos": math.cos(math.radians(bearing)),
        "history_count": float(len(history_destinations)),
        "last_destination_latitude": float(last_history["latitude"]),
        "last_destination_longitude": float(last_history["longitude"]),
        "mean_destination_latitude": mean_latitude,
        "mean_destination_longitude": mean_longitude,
    }


def _feature_row(
    row: pd.Series,
    trajectory: list[list[float]],
    prefix_fraction: float,
    history: list[dict[str, float]],
) -> tuple[dict[str, float], list[list[float]], list[dict[str, float]]]:
    start = pd.Timestamp(row["StartTrip"])
    prefix = observed_prefix_points(trajectory, prefix_fraction)
    history_destinations = history[-20:]
    features = features_from_observation(
        start=start,
        prefix=prefix,
        prefix_fraction=prefix_fraction,
        history=history_destinations,
    )
    return features, prefix, history_destinations


def prepare_samples(
    trips: pd.DataFrame,
    config: DatasetConfig,
    *,
    prefix_fractions: Sequence[float] | None = None,
    manifest: SplitManifest | None = None,
) -> tuple[pd.DataFrame, SplitManifest]:
    """Build departure-time samples using only earlier completed trips as history."""
    required = {
        "VIN",
        "trip_id",
        "StartTrip",
        "EndTrip",
        "Trajectory",
        "end_latitude",
        "end_longitude",
    }
    missing = sorted(required - set(trips.columns))
    if missing:
        raise ValueError(f"Trips table is missing required columns: {missing}")
    frame = trips.copy()
    frame["StartTrip"] = pd.to_datetime(frame["StartTrip"], utc=True, errors="raise")
    frame["EndTrip"] = pd.to_datetime(frame["EndTrip"], utc=True, errors="raise")
    frame = frame.sort_values(["VIN", "StartTrip", "trip_id"]).reset_index(drop=True)
    fractions = tuple(prefix_fractions or config.prefix_fractions)
    manifest = manifest or make_vin_split(frame["VIN"].unique(), config)
    grid = GridEncoder(config.grid_cell_size_degrees)
    records: list[dict[str, object]] = []

    for vin, group in frame.groupby("VIN", sort=True):
        history: list[dict[str, float]] = []
        previous_end: pd.Timestamp | None = None
        for _, row in group.iterrows():
            start = pd.Timestamp(row["StartTrip"])
            if previous_end is not None and previous_end >= start:
                raise ValueError(f"Overlapping or unsorted trips detected for VIN {vin}")
            trajectory = parse_trajectory(row["Trajectory"])
            actual_latitude = float(row["end_latitude"])
            actual_longitude = float(row["end_longitude"])
            actual_cell = grid.encode(actual_latitude, actual_longitude)
            for fraction in fractions:
                features, prefix, visible_history = _feature_row(
                    row, trajectory, float(fraction), history[-config.max_history_trips :]
                )
                records.append(
                    {
                        "sample_id": f"{row['trip_id']}@{float(fraction):.3f}",
                        "VIN": str(vin),
                        "vehicle_id": pseudonymize_identifier(str(vin)),
                        "trip_id": str(row["trip_id"]),
                        "StartTrip": start,
                        "EndTrip": pd.Timestamp(row["EndTrip"]),
                        "split": manifest.split_for(str(vin)),
                        "actual_latitude": actual_latitude,
                        "actual_longitude": actual_longitude,
                        "actual_cell": actual_cell,
                        "trajectory": trajectory,
                        "trajectory_prefix": prefix,
                        "history_sequence": visible_history,
                        **features,
                    }
                )
            history.append(
                {
                    "latitude": actual_latitude,
                    "longitude": actual_longitude,
                    "completed_at": pd.Timestamp(row["EndTrip"]).timestamp(),
                }
            )
            previous_end = pd.Timestamp(row["EndTrip"])

    samples = pd.DataFrame(records)
    if samples.empty:
        raise ValueError("No model samples were created")
    if set(FEATURE_COLUMNS) - set(samples.columns):
        raise AssertionError("Internal feature contract is incomplete")
    if "VIN" in FEATURE_COLUMNS or "actual_latitude" in FEATURE_COLUMNS:
        raise AssertionError("Identifiers and targets cannot be model features")
    return samples, manifest


def feature_matrix(samples: pd.DataFrame) -> np.ndarray:
    values = samples.loc[:, FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("Model features contain missing or non-finite values")
    return values


def coordinate_targets(samples: pd.DataFrame) -> np.ndarray:
    values = samples[["actual_latitude", "actual_longitude"]].to_numpy(dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("Destination targets contain missing or non-finite values")
    return values
