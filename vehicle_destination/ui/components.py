"""Pure presentation helpers shared by all Streamlit pages."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import pandas as pd

from ..config import AppConfig
from ..dataset import FEATURE_COLUMNS


PAGE_SUBTITLES = {
    "Overview": "Monitor data readiness, model artifacts, and the complete prediction workflow.",
    "Data & Trips": "Clean raw telemetry, audit discarded rows, and reconstruct model-ready journeys.",
    "Training": "Train deterministic or sequence models with completely VIN-disjoint partitions.",
    "Evaluation": "Inspect geographic error, ranked destinations, prefix stages, and subgroup performance.",
    "Predict & Map": "Compare predicted destinations with the actual destination for a selected trip.",
    "Settings": "Review the resolved configuration, scientific safeguards, and runtime contract.",
}

TABLE_SAMPLE_SIZES = (10, 25, 50, 100)
DEFAULT_TABLE_SAMPLE_SIZE = 25


def format_file_size(size_bytes: int) -> str:
    """Return a compact, human-readable file size for UI captions."""

    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024.0 or unit == "GB":
            precision = 0 if unit == "B" else 1
            return f"{size:.{precision}f} {unit}"
        size /= 1024.0
    raise AssertionError("unreachable")


def artifact_paths(config: AppConfig) -> dict[str, Path]:
    work = config.path(config.data.work_dir)
    return {
        "work": work,
        "trips_dir": work / "trips",
        "trips": work / "trips" / config.data.trips_file,
        "points": work / "trips" / config.data.points_file,
        "cleaned": work / "trips" / "cleaned_trip_source.csv",
        "cleaning_audit": work / "trips" / "raw_cleaning_audit.csv",
        "rejected": work / "trips" / "rejected_trips.csv",
        "trip_audit": work / "trips" / "trip_build_audit.json",
        "baseline": work / "models" / "baseline",
        "keras": work / "models" / "keras",
    }


def render_page_header(st, title: str, config: AppConfig) -> None:
    subtitle = PAGE_SUBTITLES[title]
    st.markdown(
        f"""
        <header class="vdp-page-header">
          <h1 class="vdp-page-title">{html.escape(title)}</h1>
          <div class="vdp-page-subtitle">{html.escape(subtitle)}</div>
        </header>
        """,
        unsafe_allow_html=True,
    )


def render_section_title(st, text: str) -> None:
    st.markdown(f'<div class="vdp-section-title">{html.escape(text)}</div>', unsafe_allow_html=True)


def _css_value_class(tone: str | None) -> str:
    return "" if not tone else f" vdp-kpi-value--{tone}"


def render_kpi_stack(st, rows: Sequence[tuple[str, str, str | None]]) -> None:
    body = "".join(
        '<div class="vdp-kpi-row">'
        f'<div class="vdp-kpi-label">{html.escape(label)}</div>'
        f'<div class="vdp-kpi-value{_css_value_class(tone)}">{html.escape(value)}</div>'
        "</div>"
        for label, value, tone in rows
    )
    st.markdown(f'<div class="vdp-kpi-stack">{body}</div>', unsafe_allow_html=True)


def _detail_rows(rows: Iterable[tuple[str, Any]]) -> str:
    return "".join(
        f'<div class="vdp-detail-label">{html.escape(str(label))}</div>'
        f'<div class="vdp-detail-value">{html.escape(format_detail_value(value))}</div>'
        for label, value in rows
    )


def render_detail_panel(
    st,
    title: str,
    groups: Sequence[Sequence[tuple[str, Any]]],
) -> None:
    group_markup = "".join(
        f'<div class="vdp-detail-group">{_detail_rows(group)}</div>' for group in groups
    )
    st.markdown(
        f'<section class="vdp-detail-panel"><div class="vdp-panel-title">{html.escape(title)}</div>{group_markup}</section>',
        unsafe_allow_html=True,
    )


def format_detail_value(value: Any) -> str:
    if value is None:
        return "Unavailable"
    if isinstance(value, float) and pd.isna(value):
        return "Unavailable"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    return str(value)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def file_timestamp(path: Path) -> str:
    if not path.exists():
        return "Unavailable"
    stamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc)
    return stamp.strftime("%Y-%m-%d %H:%M UTC")


def model_label(engine: str) -> str:
    return "Tree ensemble · baseline" if engine == "baseline" else "TensorFlow/Keras sequence model"


def model_artifact_exists(paths: Mapping[str, Path], engine: str) -> bool:
    if engine == "baseline":
        return (paths[engine] / "baseline_model.joblib").exists()
    return (paths[engine] / "destination_model.keras").exists()


def available_engines(paths: Mapping[str, Path]) -> list[str]:
    return [engine for engine in ("baseline", "keras") if model_artifact_exists(paths, engine)]


def split_summary_from_metadata(metadata: Mapping[str, Any]) -> str:
    manifest = metadata.get("split_manifest", {})
    counts = [len(manifest.get(name, [])) for name in ("train_vins", "validation_vins", "test_vins")]
    return "/".join(str(value) for value in counts) + " VINs" if any(counts) else "Unavailable"


def prediction_candidates(prediction_row: Mapping[str, Any], payload: Mapping[str, Any]) -> pd.DataFrame:
    candidates = pd.DataFrame(json.loads(prediction_row["top_k_candidates"]))
    distances = {int(item["rank"]): float(item["error_to_actual_km"]) for item in payload["candidates"]}
    candidates["distance_km"] = candidates["rank"].map(distances)
    candidates = candidates.rename(
        columns={"rank": "Rank", "probability": "Probability", "distance_km": "Distance (km)", "cell": "Cell"}
    )
    candidates["Probability"] = candidates["Probability"].map(lambda value: f"{value:.1%}")
    candidates["Distance (km)"] = candidates["Distance (km)"].map(lambda value: f"{value:.2f}")
    return candidates[["Rank", "Probability", "Distance (km)", "Cell"]]


def prefix_label(fraction: float, duration_minutes: float | None = None) -> str:
    percent = int(round(float(fraction) * 100))
    if duration_minutes is None:
        return "Departure only" if percent == 0 else f"{percent}% of trip"
    observed = int(round(float(duration_minutes) * float(fraction)))
    return f"{observed} min ({percent}%)" if percent else "0 min (departure)"


def prefix_points_text(points: Sequence[Sequence[float]]) -> str:
    return json.dumps(
        [[round(float(point[0]), 7), round(float(point[1]), 7)] for point in points],
        indent=2,
    )


def parse_prefix_points_text(text: str) -> tuple[tuple[float, float], ...]:
    """Parse an editable JSON prefix with concise, user-facing validation."""
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Custom prefix is not valid JSON: {exc.msg}") from exc
    if isinstance(raw, Mapping):
        raw = raw.get("points")
    if not isinstance(raw, list) or not raw:
        raise ValueError("Custom prefix must be a non-empty JSON array of [lat, lon]")
    points: list[tuple[float, float]] = []
    for index, point in enumerate(raw, start=1):
        if not isinstance(point, Sequence) or isinstance(point, (str, bytes)) or len(point) < 2:
            raise ValueError(f"Prefix point {index} must be [latitude, longitude]")
        latitude, longitude = float(point[0]), float(point[1])
        if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError(f"Prefix point {index} is outside valid coordinate bounds")
        points.append((latitude, longitude))
    return tuple(points)


def feature_audit_table(
    sample: Mapping[str, Any], provenance: Mapping[str, Any]
) -> pd.DataFrame:
    """Return the exact model feature vector used for a prediction."""
    sources = provenance.get("input_sources", {})
    rows = []
    for feature in FEATURE_COLUMNS:
        if feature.startswith(("hour_", "weekday_", "month_", "weekend")):
            source = sources.get("departure_time", "derived")
        elif feature.startswith("history_") or "destination_" in feature:
            source = sources.get("history", "derived")
        elif feature.startswith("origin_"):
            source = sources.get("origin", "derived")
        else:
            source = sources.get("trajectory_prefix", "derived")
        rows.append(
            {
                "Feature": feature,
                "Value": float(sample[feature]),
                "Source": source,
            }
        )
    return pd.DataFrame(rows)


def safe_error_message(exc: Exception) -> str:
    message = str(exc).strip() or exc.__class__.__name__
    return message if len(message) <= 420 else message[:417] + "…"


def stage_status_rows(config: AppConfig) -> list[dict[str, str]]:
    paths = artifact_paths(config)
    return [
        {
            "stage": "Raw telemetry",
            "status": "Available" if config.path(config.data.raw_data).exists() else "Missing",
            "artifact": str(config.path(config.data.raw_data)),
        },
        {"stage": "Cleaned trips", "status": "Ready" if paths["trips"].exists() else "Not built", "artifact": str(paths["trips"])},
        {"stage": "Baseline model", "status": "Ready" if model_artifact_exists(paths, "baseline") else "Not trained", "artifact": str(paths["baseline"])},
        {"stage": "Keras model", "status": "Ready" if model_artifact_exists(paths, "keras") else "Not trained", "artifact": str(paths["keras"])},
    ]


def render_status_table(st, rows: Sequence[Mapping[str, str]]) -> None:
    body = "".join(
        "<tr>"
        f'<td><span class="vdp-status-dot{"" if row["status"] in {"Available", "Ready"} else " vdp-status-dot--muted"}"></span>{html.escape(row["stage"])}</td>'
        f'<td>{html.escape(row["status"])}</td>'
        f'<td>{html.escape(row["artifact"])}</td>'
        "</tr>"
        for row in rows
    )
    st.markdown(
        '<div style="overflow-x:auto;border:1px solid var(--vdp-line);border-radius:var(--vdp-radius)">'
        '<table class="vdp-status-table"><thead><tr><th>Stage</th><th>Status</th><th>Artifact</th></tr></thead>'
        f"<tbody>{body}</tbody></table></div>",
        unsafe_allow_html=True,
    )
