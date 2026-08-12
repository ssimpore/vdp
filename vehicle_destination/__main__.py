"""Command-line interface for the complete VDP workflow."""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml

from .config import load_config
from .inference import VinInferenceRequest, request_from_mapping
from .pipeline import (
    build_trip_artifacts,
    clean_raw_artifacts,
    evaluate_saved_predictions,
    predict_existing_trip,
    predict_vin_scenario,
    run_all,
    train_model,
)
from .trip_builder import TerminalTripProgressBar
from .ui.maps import build_leaflet_html, build_map_payload


def _add_verbose_option(parser: argparse.ArgumentParser) -> None:
    """Add the shared opt-in progress flag to a long-running command."""
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Show progress on stderr while preserving JSON output on stdout",
    )


def _configure_verbose_logging() -> None:
    """Enable concise progress logs for this package without noisy dependencies."""
    package_logger = logging.getLogger("vehicle_destination")
    package_logger.setLevel(logging.INFO)
    if not package_logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("[vehicle-destination] %(message)s"))
        package_logger.addHandler(handler)
    package_logger.propagate = False


def _write_text(path: str | Path, text: str) -> Path:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(target)
    return target


def _load_prefix_points(path: str | None):
    if not path:
        return None
    source = Path(path).expanduser().resolve()
    if source.suffix.lower() == ".json":
        raw = json.loads(source.read_text(encoding="utf-8"))
        points = raw.get("points") if isinstance(raw, dict) else raw
        return tuple(tuple(point[:2]) for point in points)
    frame = pd.read_csv(source)
    missing = sorted({"latitude", "longitude"} - set(frame.columns))
    if missing:
        raise ValueError(f"Prefix CSV is missing columns: {missing}")
    return tuple(
        (float(row.latitude), float(row.longitude))
        for row in frame.itertuples(index=False)
    )


def _request_from_args(arguments) -> VinInferenceRequest:
    if arguments.request_file:
        source = Path(arguments.request_file).expanduser().resolve()
        loaded = yaml.safe_load(source.read_text(encoding="utf-8")) or {}
        if not isinstance(loaded, dict):
            raise ValueError("Inference request file must contain a mapping")
        return request_from_mapping(loaded)
    if not arguments.vin or not arguments.reference_trip_id:
        raise ValueError(
            "Provide --vin and --reference-trip-id, or use --request-file"
        )
    history = None
    if arguments.history_trip_ids is not None:
        history = tuple(
            value.strip()
            for value in arguments.history_trip_ids.split(",")
            if value.strip()
        )
    return VinInferenceRequest(
        vin=arguments.vin,
        reference_trip_id=arguments.reference_trip_id,
        departure_time=arguments.departure_time,
        origin_latitude=arguments.origin_latitude,
        origin_longitude=arguments.origin_longitude,
        prefix_fraction=arguments.prefix_fraction,
        prefix_points=_load_prefix_points(arguments.prefix_file),
        history_trip_ids=history,
        history_limit=arguments.history_limit,
        top_k=arguments.top_k,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vehicle-destination",
        description="Raw telemetry to destination prediction and mapped results.",
    )
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    clean = subparsers.add_parser(
        "clean-data", help="Validate and clean raw telemetry without building trips"
    )
    clean.add_argument("--raw-data")
    clean.add_argument("--output-dir")
    _add_verbose_option(clean)

    build = subparsers.add_parser("build-trips", help="Clean raw data and reconstruct trips")
    build.add_argument("--raw-data")
    build.add_argument("--output-dir")
    _add_verbose_option(build)

    train = subparsers.add_parser("train", help="Train and evaluate a model")
    train.add_argument("--engine", choices=["baseline", "keras"], default="baseline")
    train.add_argument("--trips")
    train.add_argument("--output-dir")
    _add_verbose_option(train)

    evaluate = subparsers.add_parser("evaluate", help="Recompute metrics from saved predictions")
    evaluate.add_argument("--engine", choices=["baseline", "keras"], default="baseline")
    evaluate.add_argument("--split", choices=["train", "validation", "test"], default="test")
    evaluate.add_argument("--predictions")

    predict = subparsers.add_parser("predict", help="Predict one reconstructed trip")
    predict.add_argument("--engine", choices=["baseline", "keras"], default="baseline")
    predict.add_argument("--trip-id", required=True)
    predict.add_argument("--prefix-fraction", type=float, default=0.25)
    predict.add_argument("--top-k", type=int, default=5)
    predict.add_argument("--trips")
    predict.add_argument("--model-dir")
    predict.add_argument("--map-json", help="Optional path for a map-ready JSON payload")

    vin_predict = subparsers.add_parser(
        "predict-vin",
        help="Predict from a selected VIN with editable, provenance-tracked inputs",
    )
    vin_predict.add_argument("--engine", choices=["baseline", "keras"], default="baseline")
    vin_predict.add_argument("--vin")
    vin_predict.add_argument("--reference-trip-id")
    vin_predict.add_argument("--departure-time")
    vin_predict.add_argument("--origin-latitude", type=float)
    vin_predict.add_argument("--origin-longitude", type=float)
    vin_predict.add_argument("--prefix-fraction", type=float, default=0.25)
    vin_predict.add_argument(
        "--prefix-file",
        help="JSON points array or CSV with latitude/longitude columns",
    )
    vin_predict.add_argument(
        "--history-trip-ids",
        help="Comma-separated completed trip IDs; omit for automatic history",
    )
    vin_predict.add_argument("--history-limit", type=int)
    vin_predict.add_argument("--top-k", type=int, default=5)
    vin_predict.add_argument("--request-file", help="YAML/JSON request; replaces input flags")
    vin_predict.add_argument("--trips")
    vin_predict.add_argument("--model-dir")
    vin_predict.add_argument("--output-json")
    vin_predict.add_argument("--map-json")
    vin_predict.add_argument("--map-html")

    list_vins = subparsers.add_parser(
        "list-vins", help="List VINs and available reference trips"
    )
    list_vins.add_argument("--trips")

    all_command = subparsers.add_parser("run-all", help="Build trips, train, and evaluate")
    all_command.add_argument("--engine", choices=["baseline", "keras"], default="baseline")
    _add_verbose_option(all_command)

    serve = subparsers.add_parser("serve", help="Launch the Streamlit application")
    serve.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if getattr(arguments, "verbose", False):
        _configure_verbose_logging()
    config = load_config(arguments.config)
    if arguments.command == "clean-data":
        result = clean_raw_artifacts(
            config, raw_path=arguments.raw_data, output_dir=arguments.output_dir
        )
        print(json.dumps({"outputs": result.outputs, "summary": result.summary}, indent=2))
        return 0
    if arguments.command == "build-trips":
        progress = TerminalTripProgressBar() if arguments.verbose else None
        result = build_trip_artifacts(
            config,
            raw_path=arguments.raw_data,
            output_dir=arguments.output_dir,
            progress=progress,
        )
        print(json.dumps({"outputs": result.outputs, "summary": result.summary}, indent=2))
        return 0
    if arguments.command == "train":
        result = train_model(
            config,
            engine=arguments.engine,
            trips_path=arguments.trips,
            output_dir=arguments.output_dir,
            verbose=arguments.verbose,
        )
        print(json.dumps({"outputs": result.outputs, "summary": result.summary}, indent=2))
        return 0
    if arguments.command == "evaluate":
        predictions = (
            config.path(arguments.predictions)
            if arguments.predictions
            else config.path(config.data.work_dir)
            / "models"
            / arguments.engine
            / "predictions.csv"
        )
        print(
            json.dumps(
                evaluate_saved_predictions(predictions, config, arguments.split), indent=2
            )
        )
        return 0
    if arguments.command == "predict":
        prediction, trip, model = predict_existing_trip(
            config,
            trip_id=arguments.trip_id,
            prefix_fraction=arguments.prefix_fraction,
            engine=arguments.engine,
            trips_path=arguments.trips,
            model_dir=arguments.model_dir,
            top_k=arguments.top_k,
        )
        print(prediction.to_json(orient="records", indent=2))
        if arguments.map_json:
            grid = model.grid if arguments.engine == "baseline" else model.preprocessor.grid
            payload = build_map_payload(trip, prediction.iloc[0], grid)
            path = Path(arguments.map_json).expanduser().resolve()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0
    if arguments.command == "predict-vin":
        request = _request_from_args(arguments)
        result = predict_vin_scenario(
            config,
            request=request,
            engine=arguments.engine,
            trips_path=arguments.trips,
            model_dir=arguments.model_dir,
        )
        grid = (
            result.model.grid
            if arguments.engine == "baseline"
            else result.model.preprocessor.grid
        )
        payload = build_map_payload(
            result.reference_trip,
            result.prediction.iloc[0],
            grid,
            observed_prefix=result.sample.iloc[0]["trajectory_prefix"],
        )
        bundle = {
            "request": request.to_dict(),
            "provenance": result.provenance,
            "prediction": json.loads(result.prediction.to_json(orient="records"))[0],
            "map": payload,
        }
        text = json.dumps(bundle, indent=2, allow_nan=False) + "\n"
        if arguments.output_json:
            _write_text(arguments.output_json, text)
        if arguments.map_json:
            _write_text(
                arguments.map_json,
                json.dumps(payload, indent=2, allow_nan=False) + "\n",
            )
        if arguments.map_html:
            visible = {
                "Full trajectory",
                "Observed prefix",
                "Candidates",
                "Destinations",
                "Candidate cells",
                "Connection lines",
            }
            _write_text(arguments.map_html, build_leaflet_html(payload, visible))
        print(text, end="")
        return 0
    if arguments.command == "list-vins":
        trips_path = (
            config.path(arguments.trips)
            if arguments.trips
            else config.path(config.data.work_dir) / "trips" / config.data.trips_file
        )
        trips = pd.read_csv(trips_path)
        rows = (
            trips.groupby("VIN", as_index=False)
            .agg(
                trip_count=("trip_id", "size"),
                first_trip=("StartTrip", "min"),
                last_trip=("StartTrip", "max"),
            )
            .to_dict(orient="records")
        )
        print(json.dumps(rows, indent=2))
        return 0
    if arguments.command == "run-all":
        progress = TerminalTripProgressBar() if arguments.verbose else None
        result = run_all(
            config,
            engine=arguments.engine,
            verbose=arguments.verbose,
            trip_progress=progress,
        )
        print(json.dumps({"outputs": result.outputs, "summary": result.summary}, indent=2))
        return 0
    if arguments.command == "serve":
        app_path = config.project_root / "streamlit_app.py"
        return subprocess.call(
            [
                sys.executable,
                "-m",
                "streamlit",
                "run",
                str(app_path),
                "--server.port",
                str(arguments.port),
                "--",
                "--config",
                str(Path(arguments.config).expanduser().resolve()),
            ]
        )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
