"""Reusable service layer for CLI, notebooks, tests, and Streamlit."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from .config import AppConfig, save_resolved_config
from .dataset import prepare_samples
from .evaluation import metrics_by_group, prediction_metrics, serialize_metrics
from .inference import PreparedInference, VinInferenceRequest, prepare_vin_inference
from .models.baseline import BaselineDestinationModel, load_baseline, train_baseline
from .trip_builder import (
    TripProgressCallback,
    build_trips_from_file,
    clean_raw_data,
    read_raw_table,
    write_result,
)


Engine = Literal["baseline", "keras"]
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StageResult:
    outputs: dict[str, str]
    summary: dict[str, Any]
    model: Any | None = field(default=None, repr=False, compare=False)


@dataclass(frozen=True)
class InferenceResult:
    prediction: pd.DataFrame
    reference_trip: pd.Series
    sample: pd.DataFrame
    model: Any
    provenance: dict[str, Any]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def build_trip_artifacts(
    config: AppConfig,
    *,
    raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    progress: TripProgressCallback | None = None,
) -> StageResult:
    source = config.path(raw_path or config.data.raw_data)
    destination = config.path(output_dir or config.data.work_dir) / "trips"
    LOGGER.info("Starting trip build from %s", source)
    result = build_trips_from_file(
        source,
        config.trip_builder,
        progress=progress,
    )
    paths = write_result(
        result,
        destination,
        trips_name=config.data.trips_file,
        points_name=config.data.points_file,
    )
    save_resolved_config(config, destination / "resolved_config.yaml")
    LOGGER.info("Trip artifacts are ready in %s", destination)
    return StageResult(
        outputs={name: str(path) for name, path in paths.items()},
        summary=result.audit,
    )


def clean_raw_artifacts(
    config: AppConfig,
    *,
    raw_path: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> StageResult:
    """Run the preconstruction cleaning stage without building trips."""
    source = config.path(raw_path or config.data.raw_data)
    destination = config.path(output_dir or config.data.work_dir) / "cleaning"
    destination.mkdir(parents=True, exist_ok=True)
    LOGGER.info("Starting raw telemetry cleaning from %s", source)
    cleaned = clean_raw_data(read_raw_table(source), config.trip_builder)
    data_path = destination / "cleaned_raw.csv"
    discarded_path = destination / "discarded_raw_rows.csv"
    audit_path = destination / "raw_cleaning_audit.json"
    _atomic_csv(cleaned.data, data_path)
    _atomic_csv(cleaned.discarded, discarded_path)
    _atomic_text(
        audit_path,
        json.dumps(cleaned.audit, indent=2, sort_keys=True) + "\n",
    )
    resolved_path = save_resolved_config(
        config, destination / "resolved_config.yaml"
    )
    LOGGER.info("Cleaned telemetry artifacts are ready in %s", destination)
    return StageResult(
        outputs={
            "cleaned_raw": str(data_path),
            "discarded_rows": str(discarded_path),
            "cleaning_audit": str(audit_path),
            "resolved_config": str(resolved_path),
        },
        summary=cleaned.audit,
    )


def train_model(
    config: AppConfig,
    *,
    engine: Engine = "baseline",
    trips_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    verbose: bool = False,
) -> StageResult:
    """Train and persist a model, optionally enabling engine-level progress."""
    work_dir = config.path(config.data.work_dir)
    source = config.path(trips_path) if trips_path else work_dir / "trips" / config.data.trips_file
    LOGGER.info("Reading reconstructed trips from %s", source)
    trips = pd.read_csv(source)
    model_dir = config.path(output_dir) if output_dir else work_dir / "models" / engine
    model_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.info(
        "Starting %s training from %s trips",
        engine,
        f"{len(trips):,}",
    )

    if engine == "baseline":
        model, predictions, metrics, grouped = train_baseline(
            trips, config, verbose=verbose
        )
        model_paths = model.save(model_dir)
        history_path = None
    elif engine == "keras":
        from .models.keras_model import train_keras

        model, predictions, metrics, history = train_keras(
            trips, config, model_dir, verbose=verbose
        )
        model_paths = model.save(model_dir)
        grouped = metrics_by_group(predictions)
        history_path = model_dir / "training_history.csv"
        _atomic_csv(pd.DataFrame(history.history), history_path)
    else:  # pragma: no cover - guarded by type/CLI
        raise ValueError(f"Unknown model engine: {engine}")

    predictions_path = model_dir / "predictions.csv"
    metrics_path = model_dir / "test_metrics.json"
    grouped_path = model_dir / "metrics_by_group.csv"
    _atomic_csv(predictions, predictions_path)
    _atomic_csv(grouped, grouped_path)
    _atomic_text(metrics_path, serialize_metrics(metrics))
    config_path = save_resolved_config(config, model_dir / "resolved_config.yaml")
    outputs = {
        **{name: str(path) for name, path in model_paths.items()},
        "predictions": str(predictions_path),
        "metrics": str(metrics_path),
        "metrics_by_group": str(grouped_path),
        "resolved_config": str(config_path),
    }
    if history_path:
        outputs["training_history"] = str(history_path)
    LOGGER.info("Model artifacts are ready in %s", model_dir)
    return StageResult(outputs=outputs, summary=metrics, model=model)


def load_model(engine: Engine, model_dir: str | Path) -> Any:
    directory = Path(model_dir).expanduser().resolve()
    if engine == "baseline":
        return load_baseline(directory / "baseline_model.joblib")
    if engine == "keras":
        from .models.keras_model import load_keras

        return load_keras(directory)
    raise ValueError(f"Unknown model engine: {engine}")


def predict_existing_trip(
    config: AppConfig,
    *,
    trip_id: str,
    prefix_fraction: float,
    engine: Engine = "baseline",
    trips_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    top_k: int = 5,
    loaded_model: Any | None = None,
) -> tuple[pd.DataFrame, pd.Series, Any]:
    work_dir = config.path(config.data.work_dir)
    trips_source = (
        config.path(trips_path)
        if trips_path
        else work_dir / "trips" / config.data.trips_file
    )
    trips = pd.read_csv(trips_source)
    trip = trips.loc[trips["trip_id"].astype(str).eq(str(trip_id))]
    if trip.empty:
        raise KeyError(f"Unknown trip_id: {trip_id}")
    request = VinInferenceRequest(
        vin=str(trip.iloc[0]["VIN"]),
        reference_trip_id=str(trip_id),
        prefix_fraction=float(prefix_fraction),
        top_k=int(top_k),
    )
    result = predict_vin_scenario(
        config,
        request=request,
        engine=engine,
        trips_path=trips_source,
        model_dir=model_dir,
        loaded_model=loaded_model,
    )
    return result.prediction, result.reference_trip, result.model


def predict_vin_scenario(
    config: AppConfig,
    *,
    request: VinInferenceRequest,
    engine: Engine = "baseline",
    trips_path: str | Path | None = None,
    model_dir: str | Path | None = None,
    loaded_model: Any | None = None,
) -> InferenceResult:
    """Predict one VIN scenario with a complete input provenance record."""
    work_dir = config.path(config.data.work_dir)
    trips_source = (
        config.path(trips_path)
        if trips_path
        else work_dir / "trips" / config.data.trips_file
    )
    models = config.path(model_dir) if model_dir else work_dir / "models" / engine
    trips = pd.read_csv(trips_source)
    # Interactive callers can pass a fingerprint-cached model. CLI and notebook
    # workflows remain self-contained when the argument is omitted.
    model = loaded_model if loaded_model is not None else load_model(engine, models)
    manifest = (
        model.split_manifest
        if isinstance(model, BaselineDestinationModel)
        else model.preprocessor.split_manifest
    )
    prepared: PreparedInference = prepare_vin_inference(
        trips, config.dataset, request, manifest
    )
    prediction = model.predict(prepared.sample, top_k=request.top_k)
    prediction["scenario_id"] = request.reproducibility_hash
    prediction["input_provenance"] = json.dumps(
        prepared.provenance, sort_keys=True, separators=(",", ":")
    )
    return InferenceResult(
        prediction=prediction,
        reference_trip=prepared.reference_trip,
        sample=prepared.sample,
        model=model,
        provenance=prepared.provenance,
    )


def evaluate_saved_predictions(
    predictions_path: str | Path, config: AppConfig, split: str = "test"
) -> dict[str, object]:
    predictions = pd.read_csv(predictions_path)
    selected = predictions.loc[predictions["split"].eq(split)]
    if selected.empty:
        raise ValueError(f"No predictions exist for split {split!r}")
    return prediction_metrics(
        selected,
        top_k=config.evaluation.top_k,
        recall_distances_km=config.evaluation.recall_distance_km,
    )


def run_all(
    config: AppConfig,
    *,
    engine: Engine = "baseline",
    verbose: bool = False,
    trip_progress: TripProgressCallback | None = None,
) -> StageResult:
    """Build trips, train a model, and persist an end-to-end run summary."""
    LOGGER.info("Starting complete pipeline with the %s engine", engine)
    build = build_trip_artifacts(config, progress=trip_progress)
    train = train_model(
        config,
        engine=engine,
        trips_path=build.outputs["trips"],
        verbose=verbose,
    )
    summary = {
        "trip_build": build.summary,
        "model_test": train.summary,
        "scientific_note": (
            "The bundled sample validates execution only; production claims require a "
            "larger representative dataset and untouched external validation."
        ),
    }
    work_dir = config.path(config.data.work_dir)
    manifest_path = work_dir / "run_summary.json"
    _atomic_text(manifest_path, json.dumps(summary, indent=2, sort_keys=True) + "\n")
    latest_path = work_dir / "LATEST"
    _atomic_text(latest_path, manifest_path.name + "\n")
    LOGGER.info("Complete pipeline finished; summary written to %s", manifest_path)
    return StageResult(
        outputs={**build.outputs, **train.outputs, "run_summary": str(manifest_path)},
        summary=summary,
    )


def copy_demo_data(config: AppConfig, destination: str | Path) -> Path:
    """Explicit helper used by notebooks or demos; never mutates the source fixture."""
    source = config.path(config.data.raw_data)
    target = Path(destination).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return target
