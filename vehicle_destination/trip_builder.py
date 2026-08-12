#!/usr/bin/env python3
"""Reconstruct model-ready vehicle trips from long-format telemetry.

The input format used by this project stores one signal per row. Journey start
and end records carry scalar latitude/longitude/altitude values, while periodic
records carry JSON arrays in ``timeSeriesValue``. This module pairs journey
triggers per VIN, cleans and restricts raw rows to usable trip sources, aligns
the signal arrays, validates every candidate, and produces one-row-per-trip,
one-row-per-point, accepted-source, and audit tables.

Example
-------
python build_trips.py ../upload/raw_data.csv \
    --config trip_build_config.yaml \
    --output-dir outputs
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TextIO

import pandas as pd


EARTH_RADIUS_M = 6_371_008.8
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TripBuildConfig:
    """Validated settings for raw-to-trip reconstruction."""

    # Raw column mapping.
    vin_column: str = "vin"
    trigger_column: str = "triggerOrContext"
    signal_column: str = "name"
    scalar_column: str = "scalarValue"
    series_column: str = "timeSeriesValue"
    timestamp_column: str = "vehicleCollectionTime"
    ingestion_timestamp_column: str = "inCdpTime"
    metadata_columns: tuple[str, ...] = ("fuel", "hyb", "model")

    # Trigger and signal names.
    start_trigger: str = "Vehicle/Trigger/StartOfJourney"
    end_trigger: str = "Vehicle/Trigger/EndOfJourney"
    trajectory_trigger: str = "UCD/Sent/Trigger/Trg_periodic_5_seconds"
    latitude_signal: str = "locationLatitude"
    longitude_signal: str = "locationLongitude"
    altitude_signal: str = "lastKnownLocationAltitude"

    # Eligibility and plausibility limits.
    include_vins: tuple[str, ...] = ()
    exclude_vins: tuple[str, ...] = ()
    min_duration_seconds: float = 1.0
    max_duration_seconds: float = 86_400.0
    min_points: int = 2
    min_distance_m: float = 0.0
    max_distance_m: float = 1_000_000.0
    max_segment_speed_kmh: float = 250.0
    endpoint_match_tolerance_m: float = 30.0

    # Processing behavior.
    timezone: str = "UTC"
    series_length_policy: str = "reject"  # reject | truncate
    add_missing_trigger_endpoints: bool = True
    fallback_to_trigger_endpoints: bool = True
    remove_exact_duplicates: bool = True
    coordinate_decimals: int = 7

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any] | None) -> "TripBuildConfig":
        """Create a config from the nested YAML structure used by this package."""
        raw = raw or {}
        columns = raw.get("columns", {}) or {}
        triggers = raw.get("triggers", {}) or {}
        signals = raw.get("signals", {}) or {}
        eligibility = raw.get("eligibility", {}) or {}
        processing = raw.get("processing", {}) or {}

        values: dict[str, Any] = {
            "vin_column": columns.get("vin", cls.vin_column),
            "trigger_column": columns.get("trigger", cls.trigger_column),
            "signal_column": columns.get("signal", cls.signal_column),
            "scalar_column": columns.get("scalar_value", cls.scalar_column),
            "series_column": columns.get("series_value", cls.series_column),
            "timestamp_column": columns.get("timestamp", cls.timestamp_column),
            "ingestion_timestamp_column": columns.get(
                "ingestion_timestamp", cls.ingestion_timestamp_column
            ),
            "metadata_columns": tuple(
                columns.get("metadata", list(cls.metadata_columns)) or []
            ),
            "start_trigger": triggers.get("start", cls.start_trigger),
            "end_trigger": triggers.get("end", cls.end_trigger),
            "trajectory_trigger": triggers.get(
                "trajectory", cls.trajectory_trigger
            ),
            "latitude_signal": signals.get("latitude", cls.latitude_signal),
            "longitude_signal": signals.get("longitude", cls.longitude_signal),
            "altitude_signal": signals.get("altitude", cls.altitude_signal),
            "include_vins": tuple(eligibility.get("include_vins", []) or []),
            "exclude_vins": tuple(eligibility.get("exclude_vins", []) or []),
            "min_duration_seconds": eligibility.get(
                "min_duration_seconds", cls.min_duration_seconds
            ),
            "max_duration_seconds": eligibility.get(
                "max_duration_seconds", cls.max_duration_seconds
            ),
            "min_points": eligibility.get("min_points", cls.min_points),
            "min_distance_m": eligibility.get(
                "min_distance_m", cls.min_distance_m
            ),
            "max_distance_m": eligibility.get(
                "max_distance_m", cls.max_distance_m
            ),
            "max_segment_speed_kmh": eligibility.get(
                "max_segment_speed_kmh", cls.max_segment_speed_kmh
            ),
            "endpoint_match_tolerance_m": eligibility.get(
                "endpoint_match_tolerance_m", cls.endpoint_match_tolerance_m
            ),
            "timezone": processing.get("timezone", cls.timezone),
            "series_length_policy": processing.get(
                "series_length_policy", cls.series_length_policy
            ),
            "add_missing_trigger_endpoints": processing.get(
                "add_missing_trigger_endpoints", cls.add_missing_trigger_endpoints
            ),
            "fallback_to_trigger_endpoints": processing.get(
                "fallback_to_trigger_endpoints", cls.fallback_to_trigger_endpoints
            ),
            "remove_exact_duplicates": processing.get(
                "remove_exact_duplicates", cls.remove_exact_duplicates
            ),
            "coordinate_decimals": processing.get(
                "coordinate_decimals", cls.coordinate_decimals
            ),
        }
        config = cls(**values)
        config.validate()
        return config

    def validate(self) -> None:
        if self.series_length_policy not in {"reject", "truncate"}:
            raise ValueError("processing.series_length_policy must be reject or truncate")
        if self.min_duration_seconds < 0:
            raise ValueError("eligibility.min_duration_seconds must be >= 0")
        if self.max_duration_seconds <= self.min_duration_seconds:
            raise ValueError(
                "eligibility.max_duration_seconds must exceed min_duration_seconds"
            )
        if self.min_points < 2:
            raise ValueError("eligibility.min_points must be >= 2")
        if self.min_distance_m < 0 or self.max_distance_m <= self.min_distance_m:
            raise ValueError("distance limits are invalid")
        if self.max_segment_speed_kmh <= 0:
            raise ValueError("eligibility.max_segment_speed_kmh must be > 0")
        if self.endpoint_match_tolerance_m < 0:
            raise ValueError("endpoint_match_tolerance_m must be >= 0")
        if not 0 <= self.coordinate_decimals <= 12:
            raise ValueError("processing.coordinate_decimals must be between 0 and 12")
        overlap = set(self.include_vins) & set(self.exclude_vins)
        if overlap:
            raise ValueError(f"VINs cannot be both included and excluded: {sorted(overlap)}")

    @property
    def reproducibility_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class TripBuildProgress:
    """Non-sensitive progress snapshot for VIN-level trip reconstruction."""

    completed_vins: int
    total_vins: int
    accepted_trips: int
    rejected_trips: int

    @property
    def fraction(self) -> float:
        """Return progress as a value in the closed interval [0, 1]."""
        if self.total_vins == 0:
            return 1.0
        return min(1.0, max(0.0, self.completed_vins / self.total_vins))


TripProgressCallback = Callable[[TripBuildProgress], None]


class TerminalTripProgressBar:
    """Render trip progress on one terminal line without external dependencies."""

    def __init__(self, *, stream: TextIO | None = None, width: int = 32) -> None:
        if width < 10:
            raise ValueError("Progress bar width must be at least 10 characters")
        self.stream = stream if stream is not None else sys.stderr
        self.width = width
        self._last_length = 0
        self._finished = False

    def __call__(self, progress: TripBuildProgress) -> None:
        if self._finished:
            return
        completed = round(progress.fraction * self.width)
        bar = "#" * completed + "-" * (self.width - completed)
        percentage = round(progress.fraction * 100)
        line = (
            f"[vehicle-destination] Trip reconstruction progress [{bar}] "
            f"{percentage:3d}% ({progress.completed_vins:,}/{progress.total_vins:,} VINs) "
            f"accepted={progress.accepted_trips:,} rejected={progress.rejected_trips:,}"
        )
        padding = " " * max(0, self._last_length - len(line))
        self.stream.write(f"\r{line}{padding}")
        self._last_length = len(line)
        if progress.completed_vins >= progress.total_vins:
            self.stream.write("\n")
            self._finished = True
        self.stream.flush()


@dataclass
class BuildResult:
    trips: pd.DataFrame
    points: pd.DataFrame
    cleaned_source: pd.DataFrame
    rejected: pd.DataFrame
    cleaning_audit: pd.DataFrame
    audit: dict[str, Any]


@dataclass
class RawCleaningResult:
    """Canonical trip-source rows plus a non-sensitive discard audit."""

    data: pd.DataFrame
    discarded: pd.DataFrame
    audit: dict[str, Any]


def load_config(path: str | Path | None = None) -> TripBuildConfig:
    """Load YAML configuration, or return validated defaults when omitted."""
    if path is None:
        config = TripBuildConfig()
        config.validate()
        return config
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("PyYAML is required when --config is used") from exc
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    if not isinstance(raw, Mapping):
        raise ValueError("The YAML root must be a mapping")
    return TripBuildConfig.from_mapping(raw)


def read_raw_table(path: str | Path) -> pd.DataFrame:
    """Read a supported raw telemetry table without modifying it."""
    path = Path(path).expanduser().resolve()
    suffix = path.suffix.lower()
    LOGGER.info("Reading raw telemetry from %s", path)
    if suffix in {".csv", ".txt"}:
        frame = pd.read_csv(path)
    elif suffix in {".parquet", ".pq"}:
        frame = pd.read_parquet(path)
    else:
        raise ValueError(f"Unsupported input format {suffix!r}; use CSV or Parquet")
    LOGGER.info(
        "Loaded %s raw rows with %d columns",
        f"{len(frame):,}",
        len(frame.columns),
    )
    return frame


def _source_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _duplicate_key(value: Any) -> Any:
    """Convert a cell to a stable, hashable value for duplicate detection."""
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def _normalize_series_cell(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def clean_raw_data(
    raw: pd.DataFrame,
    config: TripBuildConfig | None = None,
) -> RawCleaningResult:
    """Clean raw telemetry before trip reconstruction.

    The source DataFrame is never modified. Rows are canonicalized, validated,
    deduplicated, VIN-filtered, and restricted to the exact trigger/signal pairs
    that can contribute a start, destination, or periodic trajectory point.
    Discarded raw values are not copied to an output artifact; only row IDs and
    reason codes are retained for a reproducible audit.
    """
    config = config or TripBuildConfig()
    config.validate()
    LOGGER.info("Cleaning %s raw telemetry rows", f"{len(raw):,}")
    required = {
        config.vin_column,
        config.trigger_column,
        config.signal_column,
        config.scalar_column,
        config.series_column,
        config.timestamp_column,
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"Missing required raw columns: {missing}")

    LOGGER.info("Normalizing telemetry columns and timestamps")
    rename = {
        config.vin_column: "vin",
        config.trigger_column: "trigger",
        config.signal_column: "signal",
        config.scalar_column: "scalar",
        config.series_column: "series",
        config.timestamp_column: "timestamp",
    }
    if config.ingestion_timestamp_column in raw.columns:
        rename[config.ingestion_timestamp_column] = "ingestion_timestamp"
    data = raw.rename(columns=rename).copy().reset_index(drop=True)
    data.insert(0, "_raw_row_id", range(len(data)))
    data["vin"] = data["vin"].astype("string").str.strip()
    data["trigger"] = data["trigger"].astype("string").str.strip()
    data["signal"] = data["signal"].astype("string").str.strip()
    data["timestamp"] = pd.to_datetime(data["timestamp"], utc=True, errors="coerce")
    if "ingestion_timestamp" in data.columns:
        data["ingestion_timestamp"] = pd.to_datetime(
            data["ingestion_timestamp"], utc=True, errors="coerce"
        )
    data["scalar"] = pd.to_numeric(data["scalar"], errors="coerce")
    data["series"] = data["series"].map(_normalize_series_cell)

    discarded_rows: list[dict[str, Any]] = []

    def discard(mask: pd.Series, reason: str) -> None:
        nonlocal data
        mask = mask.fillna(False).astype(bool)
        if not mask.any():
            return
        discarded_rows.extend(
            {"source_row_id": int(row_id), "reason": reason}
            for row_id in data.loc[mask, "_raw_row_id"].tolist()
        )
        data = data.loc[~mask].copy()

    discard(data["vin"].isna() | data["vin"].eq(""), "missing_vin")
    discard(data["trigger"].isna() | data["trigger"].eq(""), "missing_trigger")
    discard(data["signal"].isna() | data["signal"].eq(""), "missing_signal")
    discard(data["timestamp"].isna(), "invalid_timestamp")

    if config.remove_exact_duplicates and not data.empty:
        LOGGER.info("Checking for exact duplicate rows")
        comparison_columns = [c for c in data.columns if c != "_raw_row_id"]
        keys = data[comparison_columns].apply(
            lambda row: tuple(_duplicate_key(value) for value in row), axis=1
        )
        discard(keys.duplicated(keep="first"), "exact_duplicate")

    if config.include_vins:
        discard(~data["vin"].isin(config.include_vins), "vin_not_included")
    if config.exclude_vins:
        discard(data["vin"].isin(config.exclude_vins), "vin_excluded")

    LOGGER.info("Filtering rows to configured journey triggers and signals")
    allowed_signals = {
        config.latitude_signal,
        config.longitude_signal,
        config.altitude_signal,
    }
    allowed_pairs = {
        (config.start_trigger, signal) for signal in allowed_signals
    } | {
        (config.end_trigger, signal) for signal in allowed_signals
    } | {
        (config.trajectory_trigger, signal) for signal in allowed_signals
    }
    relevant_mask = pd.Series(
        [
            (trigger, signal) in allowed_pairs
            for trigger, signal in zip(data["trigger"], data["signal"])
        ],
        index=data.index,
        dtype=bool,
    )
    discard(~relevant_mask, "irrelevant_trigger_or_signal")

    scalar_mask = data["trigger"].isin({config.start_trigger, config.end_trigger})
    discard(scalar_mask & data["scalar"].isna(), "invalid_scalar_value")
    series_mask = data["trigger"].eq(config.trajectory_trigger)
    discard(series_mask & data["series"].isna(), "missing_series_value")

    LOGGER.info("Validating periodic trajectory arrays")
    invalid_series_ids: list[int] = []
    for row_id, series in data.loc[
        series_mask & data["series"].notna(), ["_raw_row_id", "series"]
    ].itertuples(index=False, name=None):
        try:
            parsed = _parse_series(series)
            if not parsed:
                raise ValueError("empty trajectory array")
        except (TypeError, ValueError):
            invalid_series_ids.append(int(row_id))
    if invalid_series_ids:
        discard(data["_raw_row_id"].isin(invalid_series_ids), "invalid_series_value")

    data = data.sort_values(
        ["vin", "timestamp", "trigger", "signal", "_raw_row_id"]
    ).reset_index(drop=True)
    discarded = pd.DataFrame(discarded_rows, columns=["source_row_id", "reason"])
    if not discarded.empty:
        discarded = discarded.sort_values(["source_row_id", "reason"]).reset_index(
            drop=True
        )
    counts = (
        discarded["reason"].value_counts().sort_index().astype(int).to_dict()
        if not discarded.empty
        else {}
    )
    audit = {
        "raw_rows_received": int(len(raw)),
        "rows_retained_for_reconstruction": int(len(data)),
        "rows_discarded_before_reconstruction": int(len(discarded)),
        "preconstruction_discard_counts": counts,
    }
    LOGGER.info(
        "Cleaning complete: retained %s rows and discarded %s",
        f"{len(data):,}",
        f"{len(discarded):,}",
    )
    return RawCleaningResult(data=data, discarded=discarded, audit=audit)


def _last_number_with_source(rows: pd.DataFrame) -> tuple[float | None, int | None]:
    if rows.empty:
        return None, None
    numeric = pd.to_numeric(rows["scalar"], errors="coerce")
    valid = rows.loc[numeric.notna()].copy()
    if valid.empty:
        return None, None
    selected = valid.iloc[-1]
    return float(selected["scalar"]), int(selected["_raw_row_id"])


def _scalar_events(
    data: pd.DataFrame, trigger: str, config: TripBuildConfig
) -> list[dict[str, Any]]:
    subset = data[data["trigger"].eq(trigger)]
    events: list[dict[str, Any]] = []
    for (vin, timestamp), group in subset.groupby(["vin", "timestamp"], sort=True):
        duplicates = Counter(group["signal"].tolist())
        latitude, latitude_row_id = _last_number_with_source(
            group.loc[group["signal"].eq(config.latitude_signal)]
        )
        longitude, longitude_row_id = _last_number_with_source(
            group.loc[group["signal"].eq(config.longitude_signal)]
        )
        altitude, altitude_row_id = _last_number_with_source(
            group.loc[group["signal"].eq(config.altitude_signal)]
        )
        events.append(
            {
                "vin": str(vin),
                "timestamp": pd.Timestamp(timestamp),
                "latitude": latitude,
                "longitude": longitude,
                "altitude": altitude,
                "source_row_ids": [
                    row_id
                    for row_id in (
                        latitude_row_id,
                        longitude_row_id,
                        altitude_row_id,
                    )
                    if row_id is not None
                ],
                "duplicate_signals": sorted(k for k, v in duplicates.items() if v > 1),
            }
        )
    return events


def _parse_series(value: Any) -> list[float] | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (list, tuple)):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON array: {text[:80]!r}") from exc
    if not isinstance(parsed, list):
        raise ValueError("timeSeriesValue must decode to a JSON array")
    result: list[float] = []
    for item in parsed:
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("timeSeriesValue contains a non-finite number")
        result.append(number)
    return result


def _trajectory_batches(data: pd.DataFrame, config: TripBuildConfig) -> list[dict[str, Any]]:
    subset = data[data["trigger"].eq(config.trajectory_trigger)]
    batches: list[dict[str, Any]] = []
    for (vin, timestamp), group in subset.groupby(["vin", "timestamp"], sort=True):
        batch: dict[str, Any] = {
            "vin": str(vin),
            "timestamp": pd.Timestamp(timestamp),
            "coordinates": [],
            "error": None,
            "truncated": False,
            "source_row_ids": [],
        }
        try:
            by_signal: dict[str, list[float] | None] = {}
            for signal_name, key in [
                (config.latitude_signal, "lat"),
                (config.longitude_signal, "lon"),
                (config.altitude_signal, "alt"),
            ]:
                rows = group.loc[
                    group["signal"].eq(signal_name) & group["series"].notna(),
                    ["series", "_raw_row_id"],
                ]
                if rows.empty:
                    by_signal[key] = None
                else:
                    selected = rows.iloc[-1]
                    by_signal[key] = _parse_series(selected["series"])
                    batch["source_row_ids"].append(int(selected["_raw_row_id"]))

            latitudes = by_signal["lat"]
            longitudes = by_signal["lon"]
            altitudes = by_signal["alt"]
            if not latitudes or not longitudes:
                raise ValueError("trajectory batch lacks latitude or longitude array")
            lengths = [len(latitudes), len(longitudes)]
            if altitudes is not None:
                lengths.append(len(altitudes))
            if len(set(lengths)) != 1:
                if config.series_length_policy == "reject":
                    raise ValueError(f"unaligned trajectory array lengths: {lengths}")
                n = min(lengths)
                batch["truncated"] = True
            else:
                n = lengths[0]
            batch["coordinates"] = [
                (
                    float(latitudes[i]),
                    float(longitudes[i]),
                    None if altitudes is None else float(altitudes[i]),
                )
                for i in range(n)
            ]
        except (TypeError, ValueError) as exc:
            batch["error"] = str(exc)
        batches.append(batch)
    return batches


def haversine_m(a: Sequence[float], b: Sequence[float]) -> float:
    """Great-circle distance between two (latitude, longitude) pairs."""
    lat1, lon1 = math.radians(float(a[0])), math.radians(float(a[1]))
    lat2, lon2 = math.radians(float(b[0])), math.radians(float(b[1]))
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(
        dlon / 2.0
    ) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(h)))


def _valid_coordinate(point: Sequence[float | None]) -> bool:
    lat, lon = point[0], point[1]
    return (
        lat is not None
        and lon is not None
        and math.isfinite(float(lat))
        and math.isfinite(float(lon))
        and -90.0 <= float(lat) <= 90.0
        and -180.0 <= float(lon) <= 180.0
    )


def _metadata_for_vin(
    data: pd.DataFrame, vin: str, metadata_columns: Iterable[str]
) -> tuple[dict[str, Any], list[str]]:
    metadata: dict[str, Any] = {}
    flags: list[str] = []
    vin_rows = data[data["vin"].eq(vin)]
    for column in metadata_columns:
        if column not in vin_rows.columns:
            metadata[column] = None
            continue
        values = vin_rows[column].dropna().astype(str).unique().tolist()
        metadata[column] = values[0] if values else None
        if len(values) > 1:
            flags.append(f"inconsistent_metadata:{column}")
    return metadata, flags


def _rejection(
    vin: str,
    start_time: pd.Timestamp | None,
    end_time: pd.Timestamp | None,
    reason: str,
    detail: str,
) -> dict[str, Any]:
    return {
        "VIN": vin,
        "StartTrip": None if start_time is None else start_time.isoformat(),
        "EndTrip": None if end_time is None else end_time.isoformat(),
        "reason": reason,
        "detail": detail,
    }


def _candidate_reasons(
    duration_s: float,
    point_count: int,
    distance_m: float,
    max_segment_speed_kmh: float,
    config: TripBuildConfig,
) -> list[tuple[str, str]]:
    reasons: list[tuple[str, str]] = []
    if duration_s < config.min_duration_seconds:
        reasons.append(("duration_too_short", f"duration_s={duration_s:.3f}"))
    if duration_s > config.max_duration_seconds:
        reasons.append(("duration_too_long", f"duration_s={duration_s:.3f}"))
    if point_count < config.min_points:
        reasons.append(("too_few_points", f"point_count={point_count}"))
    if distance_m < config.min_distance_m:
        reasons.append(("distance_too_short", f"distance_m={distance_m:.3f}"))
    if distance_m > config.max_distance_m:
        reasons.append(("distance_too_long", f"distance_m={distance_m:.3f}"))
    if max_segment_speed_kmh > config.max_segment_speed_kmh:
        reasons.append(
            (
                "implausible_segment_speed",
                f"max_segment_speed_kmh={max_segment_speed_kmh:.3f}",
            )
        )
    return reasons


def _cleaned_source_table(
    data: pd.DataFrame,
    source_usage: Mapping[int, Mapping[str, set[str]]],
    config: TripBuildConfig,
) -> pd.DataFrame:
    """Return only canonical raw rows that contributed to accepted trips."""
    base_columns = [
        "source_row_id",
        "trip_ids",
        "source_roles",
        "VIN",
        "source_timestamp",
        "source_trigger",
        "source_signal",
        "scalar_value",
        "series_value",
    ]
    optional_columns = ["ingestion_timestamp", *config.metadata_columns]
    if not source_usage:
        return pd.DataFrame(columns=[*base_columns, *optional_columns])

    used_ids = set(source_usage)
    source = data[data["_raw_row_id"].isin(used_ids)].copy()
    source["trip_ids"] = source["_raw_row_id"].map(
        lambda row_id: json.dumps(
            sorted(source_usage[int(row_id)]["trip_ids"]), separators=(",", ":")
        )
    )
    source["source_roles"] = source["_raw_row_id"].map(
        lambda row_id: json.dumps(
            sorted(source_usage[int(row_id)]["source_roles"]), separators=(",", ":")
        )
    )
    source["series"] = source["series"].map(
        lambda value: json.dumps(value, separators=(",", ":"))
        if isinstance(value, (list, tuple))
        else value
    )
    source = source.rename(
        columns={
            "_raw_row_id": "source_row_id",
            "vin": "VIN",
            "timestamp": "source_timestamp",
            "trigger": "source_trigger",
            "signal": "source_signal",
            "scalar": "scalar_value",
            "series": "series_value",
        }
    )
    selected_columns = [
        *base_columns,
        *[column for column in optional_columns if column in source.columns],
    ]
    return source[selected_columns].sort_values(
        ["VIN", "source_timestamp", "source_trigger", "source_signal", "source_row_id"]
    ).reset_index(drop=True)


def build_trips(
    raw: pd.DataFrame,
    config: TripBuildConfig | None = None,
    *,
    progress: TripProgressCallback | None = None,
) -> BuildResult:
    """Transform a raw telemetry DataFrame into trips, points, and an audit."""
    config = config or TripBuildConfig()
    config.validate()
    cleaning = clean_raw_data(raw, config)
    data = cleaning.data

    LOGGER.info("Extracting start, end, and periodic trajectory events")
    start_events = _scalar_events(data, config.start_trigger, config)
    end_events = _scalar_events(data, config.end_trigger, config)
    batches = _trajectory_batches(data, config)

    starts_by_vin: dict[str, list[dict[str, Any]]] = {}
    ends_by_vin: dict[str, list[dict[str, Any]]] = {}
    batches_by_vin: dict[str, list[dict[str, Any]]] = {}
    for event in start_events:
        starts_by_vin.setdefault(event["vin"], []).append(event)
    for event in end_events:
        ends_by_vin.setdefault(event["vin"], []).append(event)
    for batch in batches:
        batches_by_vin.setdefault(batch["vin"], []).append(batch)

    trip_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    accepted_by_vin: Counter[str] = Counter()
    source_usage: dict[int, dict[str, set[str]]] = {}

    def register_source_rows(
        row_ids: Iterable[int], trip_id: str, source_role: str
    ) -> None:
        for row_id in row_ids:
            usage = source_usage.setdefault(
                int(row_id), {"trip_ids": set(), "source_roles": set()}
            )
            usage["trip_ids"].add(trip_id)
            usage["source_roles"].add(source_role)

    all_vins = sorted(set(starts_by_vin) | set(ends_by_vin))
    LOGGER.info(
        "Reconstructing trips for %s VINs from %s starts, %s ends, and %s batches",
        f"{len(all_vins):,}",
        f"{len(start_events):,}",
        f"{len(end_events):,}",
        f"{len(batches):,}",
    )
    progress_interval = max(1, math.ceil(len(all_vins) / 20))
    if progress is not None:
        progress(
            TripBuildProgress(
                completed_vins=0,
                total_vins=len(all_vins),
                accepted_trips=0,
                rejected_trips=0,
            )
        )
    for vin_number, vin in enumerate(all_vins, start=1):
        starts = sorted(starts_by_vin.get(vin, []), key=lambda x: x["timestamp"])
        ends = sorted(ends_by_vin.get(vin, []), key=lambda x: x["timestamp"])
        vin_batches = sorted(
            batches_by_vin.get(vin, []), key=lambda x: x["timestamp"]
        )
        used_end_indices: set[int] = set()
        metadata, metadata_flags = _metadata_for_vin(data, vin, config.metadata_columns)

        for start_index, start in enumerate(starts):
            start_time = start["timestamp"]
            next_start = (
                starts[start_index + 1]["timestamp"]
                if start_index + 1 < len(starts)
                else None
            )
            end_index = next(
                (
                    i
                    for i, end in enumerate(ends)
                    if i not in used_end_indices
                    and end["timestamp"] >= start_time
                    and (next_start is None or end["timestamp"] < next_start)
                ),
                None,
            )
            if end_index is None:
                rejected_rows.append(
                    _rejection(vin, start_time, None, "missing_end_trigger", "No end trigger before the next journey start")
                )
                continue
            used_end_indices.add(end_index)
            end = ends[end_index]
            end_time = end["timestamp"]

            if start["latitude"] is None or start["longitude"] is None:
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "missing_start_coordinate", "Start trigger lacks latitude or longitude")
                )
                continue
            if end["latitude"] is None or end["longitude"] is None:
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "missing_end_coordinate", "End trigger lacks latitude or longitude")
                )
                continue

            start_point = (
                float(start["latitude"]),
                float(start["longitude"]),
                start["altitude"],
            )
            end_point = (
                float(end["latitude"]),
                float(end["longitude"]),
                end["altitude"],
            )
            if not _valid_coordinate(start_point) or not _valid_coordinate(end_point):
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "invalid_trigger_coordinate", "Start or end coordinate is outside geographic bounds")
                )
                continue

            selected_batches = [
                batch
                for batch in vin_batches
                if start_time <= batch["timestamp"] <= end_time
            ]
            failed_batches = [b for b in selected_batches if b["error"]]
            if failed_batches:
                detail = "; ".join(str(b["error"]) for b in failed_batches)
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "invalid_trajectory_array", detail)
                )
                continue

            coordinates: list[tuple[float, float, float | None]] = []
            quality_flags = list(metadata_flags)
            quality_flags.extend(f"duplicate_start_signal:{x}" for x in start["duplicate_signals"])
            quality_flags.extend(f"duplicate_end_signal:{x}" for x in end["duplicate_signals"])
            for batch in selected_batches:
                if batch["truncated"]:
                    quality_flags.append("trajectory_arrays_truncated")
                for coordinate in batch["coordinates"]:
                    if coordinates and haversine_m(coordinates[-1], coordinate) < 0.01:
                        continue
                    coordinates.append(coordinate)

            timestamp_method = "interpolated_between_trip_triggers"
            if not coordinates:
                if config.fallback_to_trigger_endpoints:
                    coordinates = [start_point, end_point]
                    quality_flags.append("trajectory_fallback_to_trigger_endpoints")
                else:
                    rejected_rows.append(
                        _rejection(vin, start_time, end_time, "missing_trajectory", "No usable periodic trajectory array")
                    )
                    continue

            if any(not _valid_coordinate(point) for point in coordinates):
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "invalid_trajectory_coordinate", "Trajectory contains latitude/longitude outside geographic bounds")
                )
                continue

            start_mismatch = haversine_m(start_point, coordinates[0])
            end_mismatch = haversine_m(coordinates[-1], end_point)
            if config.add_missing_trigger_endpoints:
                if start_mismatch > config.endpoint_match_tolerance_m:
                    coordinates.insert(0, start_point)
                    quality_flags.append("start_trigger_point_added")
                if end_mismatch > config.endpoint_match_tolerance_m:
                    coordinates.append(end_point)
                    quality_flags.append("end_trigger_point_added")
            else:
                if start_mismatch > config.endpoint_match_tolerance_m:
                    quality_flags.append("start_endpoint_mismatch")
                if end_mismatch > config.endpoint_match_tolerance_m:
                    quality_flags.append("end_endpoint_mismatch")

            duration_s = float((end_time - start_time).total_seconds())
            if duration_s <= 0:
                rejected_rows.append(
                    _rejection(vin, start_time, end_time, "non_positive_duration", f"duration_s={duration_s}")
                )
                continue

            point_times = pd.date_range(start_time, end_time, periods=len(coordinates))
            segment_distances = [0.0]
            for previous, current in zip(coordinates, coordinates[1:]):
                segment_distances.append(haversine_m(previous, current))
            cumulative_distances: list[float] = []
            running_distance = 0.0
            for distance in segment_distances:
                running_distance += distance
                cumulative_distances.append(running_distance)

            segment_speeds = [0.0]
            for i in range(1, len(point_times)):
                delta_s = float((point_times[i] - point_times[i - 1]).total_seconds())
                speed = 0.0 if delta_s <= 0 else segment_distances[i] / delta_s * 3.6
                segment_speeds.append(speed)
            distance_m = cumulative_distances[-1]
            max_speed = max(segment_speeds)
            reasons = _candidate_reasons(
                duration_s, len(coordinates), distance_m, max_speed, config
            )
            if reasons:
                for reason, detail in reasons:
                    rejected_rows.append(
                        _rejection(vin, start_time, end_time, reason, detail)
                    )
                continue

            accepted_by_vin[vin] += 1
            ordinal = accepted_by_vin[vin]
            trip_id = f"{vin}_{start_time.strftime('%Y%m%dT%H%M%SZ')}_{ordinal:04d}"
            register_source_rows(start["source_row_ids"], trip_id, "start_trigger")
            register_source_rows(end["source_row_ids"], trip_id, "end_trigger")
            for batch in selected_batches:
                register_source_rows(
                    batch["source_row_ids"], trip_id, "trajectory_batch"
                )
            decimals = config.coordinate_decimals
            trajectory = [
                [round(float(lat), decimals), round(float(lon), decimals)]
                for lat, lon, _ in coordinates
            ]
            altitude_trajectory = [
                None if altitude is None else round(float(altitude), 3)
                for _, _, altitude in coordinates
            ]
            flags = sorted(set(quality_flags))
            trip_row: dict[str, Any] = {
                "VIN": vin,
                "trip_id": trip_id,
                "StartTrip": start_time.isoformat(),
                "EndTrip": end_time.isoformat(),
                "Trajectory": json.dumps(trajectory, separators=(",", ":")),
                "AltitudeTrajectory": json.dumps(
                    altitude_trajectory, separators=(",", ":")
                ),
                "start_latitude": round(start_point[0], decimals),
                "start_longitude": round(start_point[1], decimals),
                "start_altitude_m": start_point[2],
                "end_latitude": round(end_point[0], decimals),
                "end_longitude": round(end_point[1], decimals),
                "end_altitude_m": end_point[2],
                "duration_seconds": round(duration_s, 3),
                "duration_minutes": round(duration_s / 60.0, 3),
                "distance_m": round(distance_m, 3),
                "distance_km": round(distance_m / 1000.0, 6),
                "straight_line_distance_m": round(
                    haversine_m(start_point, end_point), 3
                ),
                "average_speed_kmh": round(distance_m / duration_s * 3.6, 3),
                "max_segment_speed_kmh": round(max_speed, 3),
                "point_count": len(coordinates),
                "quality_status": "accepted_with_flags" if flags else "accepted",
                "quality_flags": json.dumps(flags, separators=(",", ":")),
                "trajectory_timestamp_method": timestamp_method,
            }
            trip_row.update(metadata)
            trip_rows.append(trip_row)

            for point_index, (coordinate, timestamp) in enumerate(
                zip(coordinates, point_times)
            ):
                lat, lon, altitude = coordinate
                if point_index == 0:
                    role = "start"
                elif point_index == len(coordinates) - 1:
                    role = "destination"
                else:
                    role = "trajectory"
                point_rows.append(
                    {
                        "trip_id": trip_id,
                        "VIN": vin,
                        "point_index": point_index,
                        "point_role": role,
                        "timestamp": timestamp.isoformat(),
                        "elapsed_seconds": round(
                            float((timestamp - start_time).total_seconds()), 3
                        ),
                        "latitude": round(float(lat), decimals),
                        "longitude": round(float(lon), decimals),
                        "altitude_m": None if altitude is None else round(float(altitude), 3),
                        "segment_distance_m": round(segment_distances[point_index], 3),
                        "cumulative_distance_m": round(
                            cumulative_distances[point_index], 3
                        ),
                        "segment_speed_kmh": round(segment_speeds[point_index], 3),
                        "timestamp_method": timestamp_method,
                    }
                )

        for index, end in enumerate(ends):
            if index not in used_end_indices:
                rejected_rows.append(
                    _rejection(vin, None, end["timestamp"], "unmatched_end_trigger", "End trigger has no preceding unmatched start trigger")
                )
        if progress is not None:
            progress(
                TripBuildProgress(
                    completed_vins=vin_number,
                    total_vins=len(all_vins),
                    accepted_trips=len(trip_rows),
                    rejected_trips=len(rejected_rows),
                )
            )
        elif vin_number % progress_interval == 0 or vin_number == len(all_vins):
            LOGGER.info(
                "Trip reconstruction progress: %s/%s VINs, %s accepted, %s rejected",
                f"{vin_number:,}",
                f"{len(all_vins):,}",
                f"{len(trip_rows):,}",
                f"{len(rejected_rows):,}",
            )

    trips = pd.DataFrame(trip_rows)
    if not trips.empty:
        trips = trips.sort_values(["VIN", "StartTrip", "trip_id"]).reset_index(drop=True)
    points = pd.DataFrame(point_rows)
    if not points.empty:
        points = points.sort_values(["VIN", "trip_id", "point_index"]).reset_index(drop=True)
    rejected_columns = ["VIN", "StartTrip", "EndTrip", "reason", "detail"]
    rejected = pd.DataFrame(rejected_rows, columns=rejected_columns)
    if not rejected.empty:
        rejected = rejected.sort_values(
            ["VIN", "StartTrip", "EndTrip", "reason"], na_position="last"
        ).reset_index(drop=True)

    cleaned_source = _cleaned_source_table(data, source_usage, config)
    cleaning_rows = [
        {
            "stage": "preconstruction",
            "reason": reason,
            "row_count": int(count),
        }
        for reason, count in sorted(
            cleaning.audit["preconstruction_discard_counts"].items()
        )
    ]
    unused_after_reconstruction = int(len(data) - len(cleaned_source))
    if unused_after_reconstruction:
        cleaning_rows.append(
            {
                "stage": "post_reconstruction",
                "reason": "not_used_by_accepted_trip",
                "row_count": unused_after_reconstruction,
            }
        )
    cleaning_audit = pd.DataFrame(
        cleaning_rows, columns=["stage", "reason", "row_count"]
    )

    reason_counts = (
        rejected["reason"].value_counts().sort_index().astype(int).to_dict()
        if not rejected.empty
        else {}
    )
    audit: dict[str, Any] = {
        "config_hash": config.reproducibility_hash,
        **cleaning.audit,
        "raw_rows_after_filters": int(len(data)),
        "exact_duplicates_removed": int(
            cleaning.audit["preconstruction_discard_counts"].get(
                "exact_duplicate", 0
            )
        ),
        "raw_rows_in_cleaned_trip_source": int(len(cleaned_source)),
        "raw_rows_unused_after_reconstruction": unused_after_reconstruction,
        "raw_rows_removed_from_final_source": int(len(raw) - len(cleaned_source)),
        "vins_after_filters": int(data["vin"].nunique()),
        "start_events": int(len(start_events)),
        "end_events": int(len(end_events)),
        "trajectory_batches": int(len(batches)),
        "accepted_trips": int(len(trips)),
        "accepted_points": int(len(points)),
        "rejection_records": int(len(rejected)),
        "rejection_counts": reason_counts,
        "timestamp_note": (
            "The raw trajectory arrays have no per-point timestamps; point timestamps "
            "are deterministically interpolated between StartTrip and EndTrip."
        ),
    }
    LOGGER.info(
        "Trip reconstruction complete: %s trips, %s points, %s rejections",
        f"{len(trips):,}",
        f"{len(points):,}",
        f"{len(rejected):,}",
    )
    return BuildResult(
        trips=trips,
        points=points,
        cleaned_source=cleaned_source,
        rejected=rejected,
        cleaning_audit=cleaning_audit,
        audit=audit,
    )


def build_trips_from_file(
    raw_path: str | Path,
    config: TripBuildConfig | None = None,
    *,
    progress: TripProgressCallback | None = None,
) -> BuildResult:
    """Read a raw CSV/Parquet file and reconstruct all eligible trips."""
    path = Path(raw_path).expanduser().resolve()
    result = build_trips(read_raw_table(path), config, progress=progress)
    LOGGER.info("Computing source fingerprint")
    result.audit.update(
        {
            "source_path": str(path),
            "source_size_bytes": path.stat().st_size,
            "source_sha256": _source_sha256(path),
        }
    )
    return result


def _write_table(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        table.to_csv(path, index=False)
    elif suffix in {".parquet", ".pq"}:
        try:
            table.to_parquet(path, index=False)
        except ImportError as exc:
            raise RuntimeError(
                "Parquet output requires pyarrow; choose a .csv output or install pyarrow"
            ) from exc
    else:
        raise ValueError(f"Unsupported output format {suffix!r}; use CSV or Parquet")


def write_result(
    result: BuildResult,
    output_dir: str | Path,
    *,
    trips_name: str = "trips.csv",
    points_name: str = "trip_points.csv",
    cleaned_source_name: str = "cleaned_trip_source.csv",
    rejected_name: str = "rejected_trips.csv",
    cleaning_audit_name: str = "raw_cleaning_audit.csv",
    audit_name: str = "trip_build_audit.json",
) -> dict[str, Path]:
    """Persist model tables, accepted-trip source rows, and audit artifacts."""
    directory = Path(output_dir).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Writing trip artifacts to %s", directory)
    paths = {
        "trips": directory / trips_name,
        "points": directory / points_name,
        "cleaned_source": directory / cleaned_source_name,
        "rejected": directory / rejected_name,
        "cleaning_audit": directory / cleaning_audit_name,
        "audit": directory / audit_name,
    }
    _write_table(result.trips, paths["trips"])
    _write_table(result.points, paths["points"])
    _write_table(result.cleaned_source, paths["cleaned_source"])
    _write_table(result.rejected, paths["rejected"])
    _write_table(result.cleaning_audit, paths["cleaning_audit"])
    with paths["audit"].open("w", encoding="utf-8") as handle:
        json.dump(result.audit, handle, indent=2, sort_keys=True)
        handle.write("\n")
    LOGGER.info("Finished writing %d trip artifact files", len(paths))
    return paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Transform long-format vehicle telemetry into model-ready trips."
    )
    parser.add_argument("raw_data", type=Path, help="Input CSV or Parquet file")
    parser.add_argument("--config", type=Path, help="Optional YAML configuration")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs"), help="Output directory"
    )
    parser.add_argument("--trips-name", default="trips.csv")
    parser.add_argument("--points-name", default="trip_points.csv")
    parser.add_argument("--cleaned-source-name", default="cleaned_trip_source.csv")
    parser.add_argument("--rejected-name", default="rejected_trips.csv")
    parser.add_argument("--cleaning-audit-name", default="raw_cleaning_audit.csv")
    parser.add_argument("--audit-name", default="trip_build_audit.json")
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show bounded cleaning and reconstruction progress",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.verbose:
        logging.basicConfig(level=logging.INFO, format="[vehicle-destination] %(message)s")
    config = load_config(args.config)
    progress = TerminalTripProgressBar() if args.verbose else None
    result = build_trips_from_file(args.raw_data, config, progress=progress)
    paths = write_result(
        result,
        args.output_dir,
        trips_name=args.trips_name,
        points_name=args.points_name,
        cleaned_source_name=args.cleaned_source_name,
        rejected_name=args.rejected_name,
        cleaning_audit_name=args.cleaning_audit_name,
        audit_name=args.audit_name,
    )
    print(json.dumps({"outputs": {k: str(v) for k, v in paths.items()}, **result.audit}, indent=2))
    return 0 if not result.trips.empty else 2


if __name__ == "__main__":
    raise SystemExit(main())
