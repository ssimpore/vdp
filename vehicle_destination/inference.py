"""VIN-driven, editable and leakage-safe destination inference inputs."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import pandas as pd

from .config import DatasetConfig
from .dataset import (
    FEATURE_COLUMNS,
    SplitManifest,
    features_from_observation,
    observed_prefix_points,
    parse_trajectory,
    pseudonymize_identifier,
)
from .geo import GridEncoder, haversine_km


def _coordinate_points(value: Sequence[Sequence[float]] | None) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    points: list[tuple[float, float]] = []
    for point in value:
        if len(point) < 2:
            raise ValueError("Every custom prefix point needs latitude and longitude")
        latitude, longitude = float(point[0]), float(point[1])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Custom prefix contains an invalid coordinate")
        points.append((latitude, longitude))
    if not points:
        raise ValueError("Custom prefix must contain at least one point")
    return tuple(points)


@dataclass(frozen=True)
class VinInferenceRequest:
    """All information used to build one editable inference scenario.

    ``reference_trip_id`` supplies a traceable starting point and an optional
    actual destination for comparison. The destination never enters the model
    feature matrix. Every field that changes model inputs is explicit here and
    is therefore hashable, cacheable, and exportable.
    """

    vin: str
    reference_trip_id: str
    departure_time: str | None = None
    origin_latitude: float | None = None
    origin_longitude: float | None = None
    prefix_fraction: float = 0.0
    prefix_points: tuple[tuple[float, float], ...] | None = None
    history_trip_ids: tuple[str, ...] | None = None
    history_limit: int | None = None
    top_k: int = 5

    def __post_init__(self) -> None:
        object.__setattr__(self, "vin", str(self.vin).strip())
        object.__setattr__(self, "reference_trip_id", str(self.reference_trip_id).strip())
        object.__setattr__(self, "prefix_points", _coordinate_points(self.prefix_points))
        if self.history_trip_ids is not None:
            object.__setattr__(
                self,
                "history_trip_ids",
                tuple(dict.fromkeys(str(value).strip() for value in self.history_trip_ids)),
            )
        if not self.vin:
            raise ValueError("VIN cannot be empty")
        if not self.reference_trip_id:
            raise ValueError("reference_trip_id cannot be empty")
        if not 0 <= float(self.prefix_fraction) < 1:
            raise ValueError("prefix_fraction must be in [0, 1)")
        if self.top_k < 1:
            raise ValueError("top_k must be positive")
        if self.history_limit is not None and self.history_limit < 0:
            raise ValueError("history_limit must be non-negative")
        if (self.origin_latitude is None) != (self.origin_longitude is None):
            raise ValueError("Origin latitude and longitude must be supplied together")
        if self.origin_latitude is not None:
            if not -90 <= float(self.origin_latitude) <= 90:
                raise ValueError("origin_latitude must be in [-90, 90]")
            if not -180 <= float(self.origin_longitude) <= 180:
                raise ValueError("origin_longitude must be in [-180, 180]")

    def to_dict(self) -> dict[str, Any]:
        return {
            "vin": self.vin,
            "reference_trip_id": self.reference_trip_id,
            "departure_time": self.departure_time,
            "origin_latitude": self.origin_latitude,
            "origin_longitude": self.origin_longitude,
            "prefix_fraction": float(self.prefix_fraction),
            "prefix_points": (
                [list(point) for point in self.prefix_points]
                if self.prefix_points is not None
                else None
            ),
            "history_trip_ids": (
                list(self.history_trip_ids) if self.history_trip_ids is not None else None
            ),
            "history_limit": self.history_limit,
            "top_k": int(self.top_k),
        }

    @property
    def reproducibility_hash(self) -> str:
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class PreparedInference:
    sample: pd.DataFrame
    reference_trip: pd.Series
    provenance: dict[str, Any]


def _normalized_trips(trips: pd.DataFrame) -> pd.DataFrame:
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
    frame["VIN"] = frame["VIN"].astype(str)
    frame["trip_id"] = frame["trip_id"].astype(str)
    frame["StartTrip"] = pd.to_datetime(frame["StartTrip"], utc=True, errors="raise")
    frame["EndTrip"] = pd.to_datetime(frame["EndTrip"], utc=True, errors="raise")
    return frame.sort_values(["VIN", "StartTrip", "trip_id"]).reset_index(drop=True)


def inference_reference(
    trips: pd.DataFrame,
    *,
    vin: str,
    reference_trip_id: str,
    prefix_fraction: float,
    history_limit: int,
) -> dict[str, Any]:
    """Return UI/CLI defaults derived only from the selected VIN and trip."""
    frame = _normalized_trips(trips)
    selected = frame.loc[
        frame["VIN"].eq(str(vin)) & frame["trip_id"].eq(str(reference_trip_id))
    ]
    if selected.empty:
        raise KeyError(f"Trip {reference_trip_id!r} does not belong to VIN {vin!r}")
    trip = selected.iloc[0]
    trajectory = parse_trajectory(trip["Trajectory"])
    prefix = observed_prefix_points(trajectory, float(prefix_fraction))
    eligible = frame.loc[
        frame["VIN"].eq(str(vin))
        & frame["EndTrip"].lt(pd.Timestamp(trip["StartTrip"]))
        & ~frame["trip_id"].eq(str(reference_trip_id))
    ]
    history_ids = eligible.tail(max(0, int(history_limit)))["trip_id"].tolist()
    return {
        "vin": str(vin),
        "reference_trip_id": str(reference_trip_id),
        "departure_time": pd.Timestamp(trip["StartTrip"]).isoformat(),
        "origin_latitude": float(trajectory[0][0]),
        "origin_longitude": float(trajectory[0][1]),
        "prefix_fraction": float(prefix_fraction),
        "prefix_points": prefix,
        "eligible_history_trip_ids": eligible["trip_id"].tolist(),
        "history_trip_ids": history_ids,
        "actual_latitude": float(trip["end_latitude"]),
        "actual_longitude": float(trip["end_longitude"]),
    }


def _translated_prefix(
    reference_prefix: Sequence[Sequence[float]],
    origin_latitude: float,
    origin_longitude: float,
) -> list[list[float]]:
    delta_latitude = origin_latitude - float(reference_prefix[0][0])
    delta_longitude = origin_longitude - float(reference_prefix[0][1])
    result = [
        [float(point[0]) + delta_latitude, float(point[1]) + delta_longitude]
        for point in reference_prefix
    ]
    for latitude, longitude in result:
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Translated prefix leaves the valid coordinate range")
    return result


def prepare_vin_inference(
    trips: pd.DataFrame,
    config: DatasetConfig,
    request: VinInferenceRequest,
    manifest: SplitManifest,
) -> PreparedInference:
    """Create exactly one model-ready row from editable, traceable VIN inputs."""
    frame = _normalized_trips(trips)
    selected = frame.loc[
        frame["VIN"].eq(request.vin)
        & frame["trip_id"].eq(request.reference_trip_id)
    ]
    if selected.empty:
        raise KeyError(
            f"Trip {request.reference_trip_id!r} does not belong to VIN {request.vin!r}"
        )
    reference = selected.iloc[0].copy()
    reference_trajectory = parse_trajectory(reference["Trajectory"])
    departure = pd.Timestamp(request.departure_time or reference["StartTrip"])
    if departure.tzinfo is None:
        departure = departure.tz_localize("UTC")
    else:
        departure = departure.tz_convert("UTC")

    reference_prefix = observed_prefix_points(
        reference_trajectory, float(request.prefix_fraction)
    )
    reference_origin = reference_prefix[0]
    origin_latitude = float(
        reference_origin[0]
        if request.origin_latitude is None
        else request.origin_latitude
    )
    origin_longitude = float(
        reference_origin[1]
        if request.origin_longitude is None
        else request.origin_longitude
    )
    if request.prefix_points is None:
        prefix = _translated_prefix(
            reference_prefix, origin_latitude, origin_longitude
        )
        prefix_source = (
            "reference_trip"
            if [origin_latitude, origin_longitude] == list(reference_origin)
            else "translated_reference_trip"
        )
    else:
        prefix = [list(point) for point in request.prefix_points]
        distance_to_origin = haversine_km(
            origin_latitude, origin_longitude, prefix[0][0], prefix[0][1]
        )
        if distance_to_origin > 0.03:
            raise ValueError(
                "Custom prefix must start at the configured origin (within 30 m)"
            )
        prefix_source = "custom_points"
    if float(request.prefix_fraction) == 0.0:
        prefix = [[origin_latitude, origin_longitude]]

    eligible = frame.loc[
        frame["VIN"].eq(request.vin)
        & frame["EndTrip"].lt(departure)
        & ~frame["trip_id"].eq(request.reference_trip_id)
    ].copy()
    eligible_ids = eligible["trip_id"].tolist()
    if request.history_trip_ids is None:
        limit = (
            config.max_history_trips
            if request.history_limit is None
            else min(request.history_limit, config.max_history_trips)
        )
        history_frame = eligible.tail(limit) if limit else eligible.iloc[0:0]
        history_source = "automatic_previous_trips"
    else:
        invalid = sorted(set(request.history_trip_ids) - set(eligible_ids))
        if invalid:
            raise ValueError(
                "History trips must belong to the VIN and end before departure: "
                + ", ".join(invalid)
            )
        order = {trip_id: index for index, trip_id in enumerate(eligible_ids)}
        selected_ids = sorted(request.history_trip_ids, key=order.__getitem__)
        if len(selected_ids) > config.max_history_trips:
            raise ValueError(
                f"At most {config.max_history_trips} history trips are supported"
            )
        history_frame = eligible.set_index("trip_id").loc[selected_ids].reset_index()
        history_source = "user_selected_previous_trips"
    history = [
        {
            "latitude": float(row.end_latitude),
            "longitude": float(row.end_longitude),
            "completed_at": pd.Timestamp(row.EndTrip).timestamp(),
        }
        for row in history_frame.itertuples(index=False)
    ]
    features = features_from_observation(
        start=departure,
        prefix=prefix,
        prefix_fraction=float(request.prefix_fraction),
        history=history,
    )
    actual_latitude = float(reference["end_latitude"])
    actual_longitude = float(reference["end_longitude"])
    actual_cell = GridEncoder(config.grid_cell_size_degrees).encode(
        actual_latitude, actual_longitude
    )
    split = manifest.split_for(request.vin)
    record: dict[str, Any] = {
        "sample_id": f"scenario:{request.reproducibility_hash[:16]}",
        "scenario_id": request.reproducibility_hash,
        "VIN": request.vin,
        "vehicle_id": pseudonymize_identifier(request.vin),
        "trip_id": request.reference_trip_id,
        "reference_trip_id": request.reference_trip_id,
        "StartTrip": departure,
        "EndTrip": pd.Timestamp(reference["EndTrip"]),
        "split": split,
        "actual_latitude": actual_latitude,
        "actual_longitude": actual_longitude,
        "actual_cell": actual_cell,
        "trajectory": reference_trajectory,
        "trajectory_prefix": prefix,
        "history_sequence": history,
        **features,
    }
    sample = pd.DataFrame([record])
    missing_features = sorted(set(FEATURE_COLUMNS) - set(sample.columns))
    if missing_features:
        raise AssertionError(f"Inference feature contract is incomplete: {missing_features}")
    provenance = {
        "request_hash": request.reproducibility_hash,
        "vin": request.vin,
        "reference_trip_id": request.reference_trip_id,
        "departure_time": departure.isoformat(),
        "input_sources": {
            "departure_time": "user_override" if request.departure_time else "reference_trip",
            "origin": (
                "user_override"
                if request.origin_latitude is not None
                else "reference_trip"
            ),
            "trajectory_prefix": prefix_source,
            "history": history_source,
            "actual_destination": "reference_trip_comparison_only",
        },
        "history_trip_ids": history_frame["trip_id"].tolist(),
        "eligible_history_trip_ids": eligible_ids,
        "observed_prefix_points": len(prefix),
        "model_feature_count": len(FEATURE_COLUMNS),
        "scientific_contract": {
            "vin_is_model_feature": False,
            "actual_destination_is_model_feature": False,
            "future_history_used": False,
            "final_reference_point_in_prefix": False,
        },
    }
    return PreparedInference(sample=sample, reference_trip=reference, provenance=provenance)


def request_from_mapping(raw: Mapping[str, Any]) -> VinInferenceRequest:
    """Parse a JSON/YAML-compatible inference request."""
    allowed = {
        "vin",
        "reference_trip_id",
        "departure_time",
        "origin_latitude",
        "origin_longitude",
        "prefix_fraction",
        "prefix_points",
        "history_trip_ids",
        "history_limit",
        "top_k",
    }
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise ValueError(f"Unknown inference request settings: {unknown}")
    return VinInferenceRequest(
        vin=str(raw["vin"]),
        reference_trip_id=str(raw["reference_trip_id"]),
        departure_time=(
            None if raw.get("departure_time") in {None, ""} else str(raw["departure_time"])
        ),
        origin_latitude=(
            None if raw.get("origin_latitude") is None else float(raw["origin_latitude"])
        ),
        origin_longitude=(
            None if raw.get("origin_longitude") is None else float(raw["origin_longitude"])
        ),
        prefix_fraction=float(raw.get("prefix_fraction", 0.0)),
        prefix_points=(
            None
            if raw.get("prefix_points") is None
            else tuple(tuple(point) for point in raw["prefix_points"])
        ),
        history_trip_ids=(
            None
            if raw.get("history_trip_ids") is None
            else tuple(str(value) for value in raw["history_trip_ids"])
        ),
        history_limit=(
            None if raw.get("history_limit") is None else int(raw["history_limit"])
        ),
        top_k=int(raw.get("top_k", 5)),
    )
