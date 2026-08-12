"""Fast, reference-faithful Streamlit workspace for the destination pipeline."""

from __future__ import annotations

import argparse
import json
import math
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd
import yaml

from .. import __version__
from ..config import AppConfig, load_config
from ..dataset import make_vin_split
from ..inference import VinInferenceRequest, inference_reference
from ..pipeline import (
    build_trip_artifacts,
    load_model,
    predict_vin_scenario,
    train_model,
)
from ..trip_builder import TripBuildProgress
from .components import (
    DEFAULT_TABLE_SAMPLE_SIZE,
    PAGE_SUBTITLES,
    TABLE_SAMPLE_SIZES,
    artifact_paths,
    available_engines,
    file_timestamp,
    format_file_size,
    feature_audit_table,
    model_label,
    parse_prefix_points_text,
    prediction_candidates,
    prefix_label,
    prefix_points_text,
    read_json,
    render_detail_panel,
    render_kpi_stack,
    render_page_header,
    render_section_title,
    render_status_table,
    safe_error_message,
    split_summary_from_metadata,
    stage_status_rows,
)
from .maps import (
    build_leaflet_html,
    build_map_payload,
    build_trip_overview_html,
    build_trip_overview_payload,
)
from .theme import BRAND_HTML, inject_theme


_CACHED_CSV: Callable[..., pd.DataFrame] | None = None
_CACHED_JSON: Callable[..., dict[str, Any]] | None = None
_CACHED_MODEL: Callable[..., Any] | None = None


def _streamlit():
    try:
        import streamlit as st
    except ImportError as exc:  # pragma: no cover - depends on app extra
        raise RuntimeError(
            "Streamlit is not installed. Run `python -m pip install -e '.[app]'`."
        ) from exc
    return st


def _initial_config_path() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", default="configs/default.yaml")
    arguments, _ = parser.parse_known_args()
    return Path(arguments.config).expanduser().resolve()


def _read_csv_file(
    path: str,
    fingerprint: int,
    columns: tuple[str, ...] | None = None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    del fingerprint
    return pd.read_csv(
        path,
        usecols=list(columns) if columns is not None else None,
        nrows=row_limit,
    )


def _read_json_file(path: str, fingerprint: int) -> dict[str, Any]:
    del fingerprint
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _cached_csv(
    st,
    path: Path,
    *,
    columns: list[str] | tuple[str, ...] | None = None,
    row_limit: int | None = None,
) -> pd.DataFrame:
    global _CACHED_CSV
    if _CACHED_CSV is None:
        _CACHED_CSV = st.cache_data(show_spinner=False, max_entries=24)(_read_csv_file)
    column_key = tuple(columns) if columns is not None else None
    return _CACHED_CSV(
        str(path),
        path.stat().st_mtime_ns,
        column_key,
        row_limit,
    ).copy()


def _cached_json(st, path: Path) -> dict[str, Any]:
    global _CACHED_JSON
    if _CACHED_JSON is None:
        _CACHED_JSON = st.cache_data(show_spinner=False, max_entries=24)(_read_json_file)
    return dict(_CACHED_JSON(str(path), path.stat().st_mtime_ns))


def _clear_data_caches() -> None:
    if _CACHED_CSV is not None:
        _CACHED_CSV.clear()
    if _CACHED_JSON is not None:
        _CACHED_JSON.clear()


def _clear_prediction_cache(st) -> None:
    """Discard session predictions after data or model artifacts change."""

    st.session_state.pop("prediction_cache", None)
    st.session_state.pop("active_prediction", None)


def _load_model_file(engine: str, model_dir: str, fingerprint: tuple[int, int]):
    del fingerprint
    return load_model(engine, model_dir)


def _cached_model(st, engine: str, model_dir: Path):
    """Load once; invalidate only when the trained artifact changes."""

    global _CACHED_MODEL
    artifact = model_dir / (
        "baseline_model.joblib" if engine == "baseline" else "destination_model.keras"
    )
    fingerprint = (artifact.stat().st_mtime_ns, artifact.stat().st_size)
    if _CACHED_MODEL is None:
        _CACHED_MODEL = st.cache_resource(show_spinner=False, max_entries=4)(_load_model_file)
    return _CACHED_MODEL(engine, str(model_dir), fingerprint)


def _format_count(value: int | float) -> str:
    return f"{int(value):,}".replace(",", " ")


def _navigate(st, page: str) -> None:
    st.session_state["active_page"] = page
    st.rerun()


def _sidebar(st, config_path: str, config: AppConfig | None) -> str:
    pages = ["Overview", "Data & Trips", "Training", "Evaluation", "Predict & Map", "Settings"]
    labels = {
        "Overview": "⌂  Overview",
        "Data & Trips": "▤  Data & Trips",
        "Training": "▥  Training",
        "Evaluation": "⌁  Evaluation",
        "Predict & Map": "⌾  Predict & Map",
        "Settings": "⚙  Settings",
    }
    active = st.session_state.setdefault("active_page", "Predict & Map")
    if active not in pages:
        active = "Overview"
        st.session_state["active_page"] = active
    with st.sidebar:
        st.markdown(BRAND_HTML, unsafe_allow_html=True)
        for page in pages:
            if st.button(
                labels[page],
                key=f"nav::{page}",
                type="primary" if page == active else "secondary",
                width="stretch",
            ):
                st.session_state["active_page"] = page
                st.rerun()
        hash_text = config.reproducibility_hash[:12] if config else "invalid config"
        st.markdown(
            f"""
            <div class="vdp-sidebar-footer">
              <strong>Active configuration</strong><br/>
              {hash_text}<br/>
              VDP {__version__} · local workspace
            </div>
            """,
            unsafe_allow_html=True,
        )
    return active


def _overview(st, config: AppConfig) -> None:
    render_page_header(st, "Overview", config)
    paths = artifact_paths(config)
    trips = _cached_csv(st, paths["trips"]) if paths["trips"].exists() else pd.DataFrame()
    points = int(trips["point_count"].sum()) if not trips.empty else 0
    metrics_path = paths["baseline"] / "test_metrics.json"
    metrics = _cached_json(st, metrics_path) if metrics_path.exists() else {}

    columns = st.columns(4)
    columns[0].metric("Accepted trips", _format_count(len(trips)))
    columns[1].metric("Vehicles", _format_count(trips["VIN"].nunique() if not trips.empty else 0))
    columns[2].metric("Trajectory points", _format_count(points))
    median_error = metrics.get("median_haversine_km")
    columns[3].metric("Median test error", "Not evaluated" if median_error is None else f"{median_error:.2f} km")

    render_section_title(st, "Trips used by the model")
    if trips.empty:
        st.info("Build trips to see the complete model dataset on the map.")
    else:
        manifest = make_vin_split(trips["VIN"], config.dataset)
        coverage = build_trip_overview_payload(trips, manifest)
        split_counts = coverage["splits"]
        st.caption(
            f"{coverage['trip_count']} accepted trips across {coverage['vehicle_count']} vehicles · "
            f"{split_counts['train']['trip_count']} training · "
            f"{split_counts['validation']['trip_count']} validation · "
            f"{split_counts['test']['trip_count']} final test. "
            "Each vehicle belongs to one split only. Hover a route for trip details."
        )
        st.iframe(
            build_trip_overview_html(coverage),
            width="stretch",
            height=480,
            tab_index=0,
        )

    left, right = st.columns([2.25, 1], gap="large")
    with left:
        render_section_title(st, "Workflow readiness")
        render_status_table(st, stage_status_rows(config))
        st.markdown(
            '<div class="vdp-science-note"><b>Scientific scope.</b> The bundled 70-trip dataset is a deterministic execution fixture. It validates data contracts, leakage controls, persistence, and visualization—not production predictive performance.</div>',
            unsafe_allow_html=True,
        )
    with right:
        render_section_title(st, "Quick actions")
        with st.container(border=True):
            st.caption("Continue the workflow from its next useful stage.")
            if st.button("Process raw telemetry", width="stretch"):
                _navigate(st, "Data & Trips")
            if st.button("Train a model", width="stretch"):
                _navigate(st, "Training")
            if st.button("Review evaluation", width="stretch"):
                _navigate(st, "Evaluation")
            if st.button("Predict a destination", type="primary", width="stretch"):
                _navigate(st, "Predict & Map")


def _data_and_trips(st, config: AppConfig) -> None:
    render_page_header(st, "Data & Trips", config)
    paths = artifact_paths(config)
    source = config.path(config.data.raw_data)
    source_column, action_column = st.columns([3.5, 1.2], vertical_alignment="bottom")
    with source_column:
        upload = st.file_uploader(
            "Raw telemetry source",
            type=["csv"],
            help="Upload a long-format telemetry CSV or use the configured source.",
        )
        if upload is not None:
            upload_dir = paths["work"] / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            source = upload_dir / "raw_data.csv"
            source.write_bytes(upload.getbuffer())
            st.session_state["uploaded_raw_name"] = upload.name
    with action_column:
        build_clicked = st.button(
            "Clean & build trips",
            type="primary",
            width="stretch",
            disabled=not source.exists(),
        )

    if build_clicked:
        try:
            with st.status("Cleaning telemetry and reconstructing trips…", expanded=True) as status:
                st.write("Validating timestamps, triggers, signals, and duplicate rows")
                progress_bar = st.progress(
                    0.0,
                    text="Preparing VIN-level trip reconstruction…",
                )

                def update_progress(progress: TripBuildProgress) -> None:
                    progress_bar.progress(
                        progress.fraction,
                        text=(
                            f"Processed {progress.completed_vins:,}/{progress.total_vins:,} "
                            f"VINs · {progress.accepted_trips:,} accepted · "
                            f"{progress.rejected_trips:,} rejected"
                        ),
                    )

                result = build_trip_artifacts(
                    config,
                    raw_path=source,
                    progress=update_progress,
                )
                progress_bar.progress(
                    1.0,
                    text=f"Completed {result.summary['accepted_trips']:,} accepted trips",
                )
                st.write("Writing accepted trips, points, provenance, and rejection audits")
                status.update(label="Trip artifacts are ready", state="complete", expanded=False)
            st.session_state["last_build"] = result.summary
            _clear_data_caches()
            _clear_prediction_cache(st)
            st.success(f"Built {result.summary['accepted_trips']} accepted trips.")
        except Exception as exc:
            st.error(safe_error_message(exc))
            with st.expander("Technical details"):
                st.exception(exc)

    if not source.exists():
        st.error(f"Configured raw source does not exist: {source}")
        return

    preview_control, preview_note = st.columns([1, 3], vertical_alignment="bottom")
    with preview_control:
        raw_row_limit = st.selectbox(
            "Raw rows shown",
            TABLE_SAMPLE_SIZES,
            index=TABLE_SAMPLE_SIZES.index(DEFAULT_TABLE_SAMPLE_SIZE),
            help="Only this many rows are read for the browser preview.",
        )
    with preview_note:
        st.caption(
            "Sample-only preview · the complete telemetry file stays on disk until you run "
            "Clean & build trips."
        )

    raw_preview = _cached_csv(st, source, row_limit=raw_row_limit)
    metrics = st.columns(4)
    metrics[0].metric("Sample rows", _format_count(len(raw_preview)))
    metrics[1].metric("Columns", _format_count(len(raw_preview.columns)))
    vin_column = config.trip_builder.vin_column
    metrics[2].metric("VINs in sample", _format_count(raw_preview[vin_column].nunique() if vin_column in raw_preview else 0))
    metrics[3].metric("Trip artifact", "Ready" if paths["trips"].exists() else "Not built")

    tabs = st.tabs(["Raw preview", "Reconstructed trips", "Cleaning audit", "Artifacts"])
    with tabs[0]:
        st.caption(
            f"Showing the first {len(raw_preview):,} rows · "
            f"{format_file_size(source.stat().st_size)} source · {source}"
        )
        st.dataframe(raw_preview, width="stretch", hide_index=True, height=360)
    with tabs[1]:
        if paths["trips"].exists():
            columns = [
                "trip_id", "StartTrip", "duration_minutes", "distance_km", "point_count", "average_speed_kmh", "quality_status"
            ]
            trip_row_limit = st.selectbox(
                "Trip rows shown",
                TABLE_SAMPLE_SIZES,
                index=TABLE_SAMPLE_SIZES.index(DEFAULT_TABLE_SAMPLE_SIZE),
                help="The complete trips artifact is not loaded into the browser.",
            )
            trips = _cached_csv(
                st,
                paths["trips"],
                columns=columns,
                row_limit=trip_row_limit,
            )
            trip_audit = _cached_json(st, paths["trip_audit"]) if paths["trip_audit"].exists() else {}
            accepted_trips = trip_audit.get("accepted_trips")
            total_label = f" of {accepted_trips:,}" if isinstance(accepted_trips, int) else ""
            st.caption(
                f"Showing the first {len(trips):,}{total_label} model-ready trips · "
                f"{format_file_size(paths['trips'].stat().st_size)} artifact."
            )
            st.dataframe(trips, width="stretch", hide_index=True, height=390)
            if paths["trips"].stat().st_size <= 25 * 1024 * 1024:
                st.download_button(
                    "↓  Export complete trips CSV",
                    data=paths["trips"].read_bytes(),
                    file_name="trips.csv",
                    mime="text/csv",
                )
            else:
                st.info(
                    "The full CSV is too large to preload safely in the browser. "
                    f"Use the artifact directly at {paths['trips']}."
                )
        else:
            st.info("Run Clean & build trips to create the model-ready table.")
    with tabs[2]:
        if paths["trip_audit"].exists():
            audit = _cached_json(st, paths["trip_audit"])
            audit_metrics = st.columns(4)
            audit_metrics[0].metric("Input rows", _format_count(audit.get("raw_rows_received", 0)))
            audit_metrics[1].metric("Accepted trips", _format_count(audit.get("accepted_trips", 0)))
            audit_metrics[2].metric("Rejections", _format_count(audit.get("rejection_records", 0)))
            audit_metrics[3].metric("Rows removed", _format_count(audit.get("raw_rows_removed_from_final_source", 0)))
            if paths["cleaning_audit"].exists():
                st.dataframe(_cached_csv(st, paths["cleaning_audit"]), width="stretch", hide_index=True)
            with st.expander("Complete build audit"):
                st.json(audit)
        else:
            st.info("No cleaning audit is available yet.")
    with tabs[3]:
        artifact_rows = [
            {"Artifact": name, "Available": path.exists(), "Path": str(path), "Size (KB)": round(path.stat().st_size / 1024, 1) if path.exists() else None}
            for name, path in paths.items()
            if name not in {"work", "trips_dir", "baseline", "keras"}
        ]
        st.dataframe(pd.DataFrame(artifact_rows), width="stretch", hide_index=True)


def _training(st, config: AppConfig) -> None:
    render_page_header(st, "Training", config)
    paths = artifact_paths(config)
    if not paths["trips"].exists():
        st.warning("Build trips first on Data & Trips.")
        if st.button("Open Data & Trips →"):
            _navigate(st, "Data & Trips")
        return

    controls = st.columns([1.35, 1, 1, 1.25], vertical_alignment="bottom")
    engine = controls[0].selectbox(
        "Model engine",
        ["baseline", "keras"],
        format_func=model_label,
    )
    controls[1].number_input(
        "Prefix stages",
        value=len(config.dataset.prefix_fractions),
        disabled=True,
    )
    controls[2].number_input(
        "Split seed",
        value=config.dataset.split_seed,
        disabled=True,
    )
    start = controls[3].button(
        "▶  Start training",
        type="primary",
        width="stretch",
    )

    if start:
        try:
            with st.status(f"Training {model_label(engine)}…", expanded=True) as status:
                st.write("Creating VIN-disjoint train, validation, and final-test partitions")
                st.write("Fitting preprocessing and the model on training VINs only")
                result = train_model(config, engine=engine)
                st.write("Persisting model, metadata, predictions, and untouched-test metrics")
                status.update(label="Training and evaluation completed", state="complete", expanded=False)
            st.session_state["last_training"] = result.summary
            _clear_data_caches()
            _clear_prediction_cache(st)
            st.success("The model and its reproducibility artifacts are ready.")
        except Exception as exc:
            st.error(safe_error_message(exc))
            with st.expander("Technical details"):
                st.exception(exc)

    model_dir = paths[engine]
    metadata = read_json(model_dir / "model_metadata.json")
    metrics = read_json(model_dir / "test_metrics.json")
    left, right = st.columns([1.65, 1], gap="large")
    with left:
        render_section_title(st, "Training contract")
        rows = [
            {"Setting": "Engine", "Value": model_label(engine)},
            {"Setting": "Trajectory prefixes", "Value": ", ".join(f"{value:.0%}" for value in config.dataset.prefix_fractions)},
            {"Setting": "Validation VIN fraction", "Value": f"{config.dataset.validation_vin_fraction:.0%}"},
            {"Setting": "Final-test VIN fraction", "Value": f"{config.dataset.test_vin_fraction:.0%}"},
            {"Setting": "VIN as neural feature", "Value": "Never"},
            {"Setting": "Destination/final point leakage", "Value": "Blocked"},
        ]
        st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True, height=250)
    with right:
        render_section_title(st, "Latest artifact")
        render_detail_panel(
            st,
            "Model status",
            [
                [("Status", "Ready" if metadata else "Not trained"), ("Saved", file_timestamp(model_dir / "model_metadata.json"))],
                [("VIN split", split_summary_from_metadata(metadata)), ("Config", str(metadata.get("config_hash", "Unavailable"))[:12])],
                [("Median test error", "Unavailable" if metrics.get("median_haversine_km") is None else f"{metrics['median_haversine_km']:.2f} km")],
            ],
        )


def _evaluation(st, config: AppConfig) -> None:
    render_page_header(st, "Evaluation", config)
    paths = artifact_paths(config)
    engines = available_engines(paths)
    if not engines:
        st.warning("Train a model before opening the evaluation workspace.")
        return
    engine = st.selectbox("Model", engines, format_func=model_label)
    model_dir = paths[engine]
    metrics_path = model_dir / "test_metrics.json"
    predictions_path = model_dir / "predictions.csv"
    if not metrics_path.exists() or not predictions_path.exists():
        st.warning(f"No saved {engine} evaluation is available.")
        return
    metrics = _cached_json(st, metrics_path)
    evaluation_columns = [
        "VIN",
        "vehicle_id",
        "trip_id",
        "split",
        "prefix_fraction",
        "error_km",
        "actual_rank",
        "predicted_cell_probability",
    ]
    predictions = _cached_csv(st, predictions_path, columns=evaluation_columns)
    splits = [value for value in ("test", "validation", "train") if value in set(predictions["split"])]
    split = st.selectbox("Dataset partition", splits)
    selected = predictions.loc[predictions["split"].eq(split)].copy()

    metric_columns = st.columns(4)
    metric_columns[0].metric("Median error", f"{selected['error_km'].median():.2f} km")
    metric_columns[1].metric("P90 error", f"{selected['error_km'].quantile(.9):.2f} km")
    ranks = pd.to_numeric(selected["actual_rank"], errors="coerce")
    top3 = (ranks.dropna() <= 3).mean() if ranks.notna().any() else None
    metric_columns[2].metric("Top-3 cell", "Undefined" if top3 is None else f"{top3:.1%}")
    metric_columns[3].metric("Vehicles", _format_count(selected["VIN"].nunique()))

    chart_column, distribution_column = st.columns(2, gap="large")
    curve = selected.groupby("prefix_fraction", as_index=False)["error_km"].median().rename(columns={"error_km": "Median error (km)"})
    with chart_column:
        render_section_title(st, "Median error by observed prefix")
        st.line_chart(curve, x="prefix_fraction", y="Median error (km)", height=260)
    with distribution_column:
        render_section_title(st, "Error distribution")
        bins = pd.cut(selected["error_km"], bins=min(12, max(4, int(math.sqrt(len(selected))))), duplicates="drop")
        distribution = bins.value_counts(sort=False).rename_axis("Error range").reset_index(name="Predictions")
        distribution["Error range"] = distribution["Error range"].astype(str)
        st.bar_chart(distribution, x="Error range", y="Predictions", height=260)

    audit_title, audit_control = st.columns([3, 1], vertical_alignment="bottom")
    with audit_title:
        render_section_title(st, "Prediction audit")
    with audit_control:
        audit_row_limit = st.selectbox(
            "Audit rows shown",
            TABLE_SAMPLE_SIZES,
            index=TABLE_SAMPLE_SIZES.index(DEFAULT_TABLE_SAMPLE_SIZE),
            help="Only the worst-error sample is sent to the browser table.",
        )
    audit_columns = ["vehicle_id", "trip_id", "prefix_fraction", "error_km", "actual_rank", "predicted_cell_probability"]
    audit = selected[audit_columns].sort_values("error_km", ascending=False)
    visible_audit = audit.head(audit_row_limit)
    st.caption(
        f"Showing {len(visible_audit):,} of {len(audit):,} predictions, sorted by largest error. "
        "The complete artifact remains on disk."
    )
    st.dataframe(visible_audit, width="stretch", hide_index=True, height=340)
    st.download_button(
        "↓  Export visible sample (CSV)",
        data=visible_audit.to_csv(index=False).encode("utf-8"),
        file_name=f"evaluation_sample_{engine}_{split}.csv",
        mime="text/csv",
    )
    if split == "test" and metrics.get("cell_metrics_defined_samples", 0) == 0:
        st.info("Held-out destinations are absent from the fitted vocabulary. Cell metrics are correctly undefined; geographic metrics remain valid.")


def _render_map(st, payload: Mapping[str, Any], visible: set[str]) -> None:
    map_html = build_leaflet_html(payload, visible)
    st.iframe(map_html, width="stretch", height=444, tab_index=0)
    st.markdown(
        """
        <div class="vdp-map-legend" aria-label="Map legend">
          <span class="vdp-legend-item" style="color:#1f2937"><span class="vdp-legend-dot"></span>Origin</span>
          <span class="vdp-legend-item" style="color:#10a05a"><span class="vdp-legend-dot"></span>Actual destination</span>
          <span class="vdp-legend-item" style="color:#ef4444"><span class="vdp-legend-dot"></span>Refined prediction</span>
          <span class="vdp-legend-item" style="color:#0968d8"><span class="vdp-legend-dot"></span>Candidate</span>
          <span class="vdp-legend-item" style="color:#0968d8"><span class="vdp-legend-line"></span>Observed trajectory</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _run_prediction(
    st,
    config: AppConfig,
    engine: str,
    request: VinInferenceRequest,
) -> bool:
    model_dir = artifact_paths(config)[engine]
    artifact = model_dir / (
        "baseline_model.joblib" if engine == "baseline" else "destination_model.keras"
    )
    cache_key = (
        config.reproducibility_hash,
        engine,
        artifact.stat().st_mtime_ns,
        artifact.stat().st_size,
        request.reproducibility_hash,
    )
    cache = st.session_state.setdefault("prediction_cache", {})
    if cache_key in cache:
        st.session_state["active_prediction"] = cache[cache_key]
        return True

    model = _cached_model(st, engine, artifact_paths(config)[engine])
    inference = predict_vin_scenario(
        config,
        request=request,
        engine=engine,
        loaded_model=model,
    )
    grid = model.grid if engine == "baseline" else model.preprocessor.grid
    payload = build_map_payload(
        inference.reference_trip,
        inference.prediction.iloc[0],
        grid,
        observed_prefix=inference.sample.iloc[0]["trajectory_prefix"],
    )
    result = {
        "signature": (engine, request.reproducibility_hash),
        "request": request.to_dict(),
        "prediction": inference.prediction,
        "trip": inference.reference_trip.to_dict(),
        "sample": inference.sample,
        "provenance": inference.provenance,
        "payload": payload,
        "ran_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    cache[cache_key] = result
    # Keep memory use predictable during long exploratory sessions.
    while len(cache) > 12:
        cache.pop(next(iter(cache)))
    st.session_state["active_prediction"] = result
    return False


def _predict_and_map(st, config: AppConfig) -> None:
    render_page_header(st, "Predict & Map", config)
    paths = artifact_paths(config)
    if not paths["trips"].exists():
        st.warning("Build trips first.")
        return
    trips = _cached_csv(st, paths["trips"])
    engines = available_engines(paths)
    if not engines:
        st.warning("Train a model first.")
        return

    controls = st.columns([1.25, .9, 1.55, .58, 1.05], vertical_alignment="bottom")
    engine = controls[0].selectbox("Model", engines, format_func=model_label)
    engine_metadata = read_json(paths[engine] / "model_metadata.json")
    test_vins = set(engine_metadata.get("split_manifest", {}).get("test_vins", []))
    vin_options = sorted(trips["VIN"].astype(str).unique())
    default_vin_index = next(
        (index for index, vin in enumerate(vin_options) if vin in test_vins),
        0,
    )
    vin = controls[1].selectbox("VIN", vin_options, index=default_vin_index)
    vin_trips = trips.loc[trips["VIN"].astype(str).eq(str(vin))].copy()
    trip_options = vin_trips["trip_id"].astype(str).tolist()
    trip_id = controls[2].selectbox("Reference trip", trip_options)
    top_k_options = sorted(set(config.evaluation.top_k) | {10})
    top_k = controls[3].selectbox("Top-K", top_k_options, index=len(top_k_options) - 1)
    run = controls[4].button(
        "▶  Run prediction",
        type="primary",
        width="stretch",
    )

    default_prefix = float(config.dataset.prefix_fractions[-1])
    reference = inference_reference(
        trips,
        vin=str(vin),
        reference_trip_id=str(trip_id),
        prefix_fraction=default_prefix,
        history_limit=config.dataset.max_history_trips,
    )
    reference_start = pd.Timestamp(reference["departure_time"])
    with st.expander(
        "Scenario inputs — departure, origin, observed trajectory and history",
        expanded=False,
    ):
        st.caption(
            "Inputs are prefilled from the selected VIN and reference trip. Edit any value before running inference; the actual destination remains comparison-only."
        )
        timing = st.columns([1, 1, .78, .78], vertical_alignment="bottom")
        departure_date = timing[0].date_input(
            "Departure date (UTC)",
            value=reference_start.date(),
            key=f"scenario-date::{trip_id}",
        )
        departure_clock = timing[1].time_input(
            "Departure time (UTC)",
            value=reference_start.time().replace(tzinfo=None),
            key=f"scenario-time::{trip_id}",
        )
        origin_latitude = timing[2].number_input(
            "Origin latitude",
            min_value=-90.0,
            max_value=90.0,
            value=float(reference["origin_latitude"]),
            format="%.7f",
            key=f"scenario-lat::{trip_id}",
        )
        origin_longitude = timing[3].number_input(
            "Origin longitude",
            min_value=-180.0,
            max_value=180.0,
            value=float(reference["origin_longitude"]),
            format="%.7f",
            key=f"scenario-lon::{trip_id}",
        )
        departure_timestamp = pd.Timestamp(
            datetime.combine(departure_date, departure_clock), tz="UTC"
        )
        observation = st.columns([1, 1.2, 2.25], vertical_alignment="bottom")
        prefix_percent = observation[0].slider(
            "Observed trip fraction",
            min_value=0,
            max_value=90,
            value=int(round(default_prefix * 100)),
            step=5,
            format="%d%%",
            key=f"scenario-prefix::{trip_id}",
        )
        prefix = float(prefix_percent) / 100.0
        trajectory_mode = observation[1].selectbox(
            "Observed trajectory",
            ["Reference trip", "Custom JSON points"],
            key=f"scenario-trajectory-mode::{trip_id}",
        )
        eligible_history = vin_trips.loc[
            pd.to_datetime(vin_trips["EndTrip"], utc=True).lt(departure_timestamp)
            & ~vin_trips["trip_id"].astype(str).eq(str(trip_id))
        ]["trip_id"].astype(str).tolist()
        default_history = eligible_history[-config.dataset.max_history_trips :]
        history_trip_ids = observation[2].multiselect(
            "Completed history trips",
            options=eligible_history,
            default=default_history,
            key=f"scenario-history::{trip_id}::{departure_timestamp.isoformat()}",
            help="Only trips completed before the edited departure time are eligible.",
        )
        custom_prefix = None
        if trajectory_mode == "Custom JSON points":
            dynamic_reference = inference_reference(
                trips,
                vin=str(vin),
                reference_trip_id=str(trip_id),
                prefix_fraction=float(prefix),
                history_limit=config.dataset.max_history_trips,
            )
            custom_text = st.text_area(
                "Observed prefix points as [[latitude, longitude], ...]",
                value=prefix_points_text(dynamic_reference["prefix_points"]),
                height=145,
                key=f"scenario-prefix-json::{trip_id}::{float(prefix):.2f}",
            )
            try:
                custom_prefix = parse_prefix_points_text(custom_text)
            except ValueError as exc:
                st.error(safe_error_message(exc))
                return

    request = VinInferenceRequest(
        vin=str(vin),
        reference_trip_id=str(trip_id),
        departure_time=(
            None
            if departure_timestamp == reference_start
            else departure_timestamp.isoformat()
        ),
        origin_latitude=(
            None
            if math.isclose(float(origin_latitude), float(reference["origin_latitude"]), abs_tol=1e-9)
            and math.isclose(float(origin_longitude), float(reference["origin_longitude"]), abs_tol=1e-9)
            else float(origin_latitude)
        ),
        origin_longitude=(
            None
            if math.isclose(float(origin_latitude), float(reference["origin_latitude"]), abs_tol=1e-9)
            and math.isclose(float(origin_longitude), float(reference["origin_longitude"]), abs_tol=1e-9)
            else float(origin_longitude)
        ),
        prefix_fraction=float(prefix),
        prefix_points=custom_prefix,
        history_trip_ids=(
            None
            if list(history_trip_ids) == default_history
            else tuple(history_trip_ids)
        ),
        top_k=int(top_k),
    )
    signature = (engine, request.reproducibility_hash)
    active = st.session_state.get("active_prediction")
    should_seed = active is None
    if run or should_seed:
        try:
            with st.spinner("Running leakage-safe destination inference…"):
                cache_hit = _run_prediction(st, config, engine, request)
            active = st.session_state["active_prediction"]
            if run and cache_hit:
                st.toast("Loaded instantly from the session prediction cache.", icon="⚡")
        except Exception as exc:
            st.error(safe_error_message(exc))
            with st.expander("Technical details"):
                st.exception(exc)
            return
    if active is None:
        return
    if active["signature"] != signature:
        st.info("Controls changed. Select Run prediction to refresh the map and ranked candidates.")

    prediction: pd.DataFrame = active["prediction"]
    trip: Mapping[str, Any] = active["trip"]
    payload: Mapping[str, Any] = active["payload"]
    row = prediction.iloc[0]
    model_dir = paths[str(active["signature"][0])]
    metadata = read_json(model_dir / "model_metadata.json")
    provenance: Mapping[str, Any] = active["provenance"]
    sample: pd.DataFrame = active["sample"]

    map_column, detail_column = st.columns([4.65, 1], gap="small")
    with map_column:
        visible = {"Full trajectory", "Observed prefix", "Destinations", "Candidates", "Connection lines"}
        _render_map(st, payload, visible)
    with detail_column:
        render_detail_panel(
            st,
            "Active configuration",
            [
                [
                    ("Model", model_label(str(active["signature"][0]))),
                    ("Trained", file_timestamp(model_dir / "model_metadata.json")),
                    ("Features", len(metadata.get("feature_columns", [])) or "Dynamic sequence inputs"),
                    ("VIN split", split_summary_from_metadata(metadata)),
                ],
                [
                    ("VIN", str(row["VIN"])),
                    ("Reference trip", trip["trip_id"]),
                    ("Scenario time", provenance["departure_time"]),
                    ("Origin", f"{float(row['origin_latitude']):.5f}, {float(row['origin_longitude']):.5f}"),
                    ("Duration", f"{float(trip['duration_minutes']):.0f} min"),
                    ("Prefix", prefix_label(float(row["prefix_fraction"]), float(trip["duration_minutes"]))),
                    ("History", f"{len(provenance['history_trip_ids'])} completed trips"),
                    ("Top-K", len(payload["candidates"])),
                ],
            ],
        )

    metrics_column, table_column = st.columns([1.28, 2.72], gap="small")
    actual_rank = row["actual_rank"]
    rank_text = "Outside Top-K" if pd.isna(actual_rank) else f"{int(actual_rank)} / {len(payload['candidates'])}"
    with metrics_column:
        render_kpi_stack(
            st,
            [
                ("Geographic error", f"{float(row['error_km']):.2f} km", "bad" if float(row["error_km"]) > 5 else None),
                ("Actual rank", rank_text, "good" if not pd.isna(actual_rank) and int(actual_rank) <= 3 else None),
                ("Confidence (Top-1)", f"{float(row['predicted_cell_probability']):.1%}", None),
                ("Dataset split", str(row["split"]).title(), None),
            ],
        )
    with table_column:
        candidates = prediction_candidates(row, payload)
        st.dataframe(candidates, width="stretch", hide_index=True, height=244)

    with st.expander("Prediction input audit", expanded=False):
        audit_left, audit_right = st.columns([1.8, 1], gap="large")
        with audit_left:
            st.caption("Exact 21-value feature vector supplied to the selected model.")
            st.dataframe(
                feature_audit_table(sample.iloc[0], provenance),
                width="stretch",
                hide_index=True,
                height=330,
            )
        with audit_right:
            st.caption("Input origin and leakage-safeguard provenance.")
            st.json(provenance, expanded=2)

    footer_columns = st.columns([2.6, .95, 1.05], vertical_alignment="center")
    footer_columns[0].markdown(
        f'<div class="vdp-run-footer"><span>Prediction run: {active["ran_at"]}</span><span class="vdp-run-footer__separator">|</span><span>Inference: 1 trip</span><span class="vdp-run-footer__separator">|</span><span>Config: {config.reproducibility_hash[:12]}</span></div>',
        unsafe_allow_html=True,
    )
    footer_columns[1].download_button(
        "↓  Export results (CSV)",
        data=prediction.to_csv(index=False).encode("utf-8"),
        file_name=f"prediction_{trip['trip_id']}.csv",
        mime="text/csv",
        width="stretch",
    )
    request_export = {
        "request": active["request"],
        "provenance": provenance,
        "prediction": json.loads(prediction.to_json(orient="records"))[0],
    }
    footer_columns[2].download_button(
        "↓  Export scenario (JSON)",
        data=(json.dumps(request_export, indent=2, allow_nan=False) + "\n").encode("utf-8"),
        file_name=f"scenario_{str(row['VIN'])}_{provenance['request_hash'][:10]}.json",
        mime="application/json",
        width="stretch",
    )


def _settings(st, config: AppConfig) -> None:
    render_page_header(st, "Settings", config)
    config_tab, safeguards_tab, runtime_tab = st.tabs(["Resolved configuration", "Scientific safeguards", "Runtime"])
    with config_tab:
        left, right = st.columns([2.3, 1], gap="large")
        with left:
            st.json(config.to_dict(include_runtime=True), expanded=2)
        with right:
            render_detail_panel(
                st,
                "Configuration identity",
                [
                    [("Project", config.project_name), ("Version", config.config_version)],
                    [("Hash", config.reproducibility_hash), ("Root", config.project_root)],
                ],
            )
            resolved = config.to_dict(include_runtime=True)
            resolved["reproducibility_hash"] = config.reproducibility_hash
            st.download_button(
                "↓  Export resolved YAML",
                data=yaml.safe_dump(resolved, sort_keys=False).encode("utf-8"),
                file_name="resolved_config.yaml",
                mime="application/yaml",
                width="stretch",
            )
    with safeguards_tab:
        safeguards = pd.DataFrame(
            [
                ("VIN feature exclusion", "Pass", "VIN is used only for grouping, history, and dataset partitions."),
                ("VIN-disjoint partitions", "Pass", "A VIN belongs to exactly one train, validation, or final-test partition."),
                ("Future-trip leakage", "Pass", "History contains only trips completed before target departure."),
                ("Destination leakage", "Pass", "The destination and final GPS point are excluded from trajectory prefixes."),
                ("Training-fitted preprocessing", "Pass", "Saved preprocessing and spatial metadata are reused for inference."),
                ("Undefined metrics", "Pass", "Unsupported destination-cell metrics remain undefined rather than zero."),
            ],
            columns=["Safeguard", "Status", "Contract"],
        )
        st.dataframe(safeguards, width="stretch", hide_index=True, height=300)
    with runtime_tab:
        runtime = pd.DataFrame(
            [
                ("Application", __version__),
                ("Python", platform.python_version()),
                ("pandas", pd.__version__),
                ("Platform", platform.platform()),
                ("Project root", str(config.project_root)),
                ("Configuration", st.session_state.get("config_path", "Unavailable")),
            ],
            columns=["Component", "Value"],
        )
        st.dataframe(runtime, width="stretch", hide_index=True)


def main() -> None:
    st = _streamlit()
    st.set_page_config(
        page_title="Vehicle Destination Lab",
        page_icon="🧪",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={"About": "Vehicle Destination Lab — reproducible destination prediction."},
    )
    inject_theme(st)
    default_path = _initial_config_path()
    config_text = st.session_state.setdefault("config_path", str(default_path))
    try:
        config = load_config(config_text)
    except Exception as exc:
        _sidebar(st, config_text, None)
        st.error(f"Invalid configuration: {safe_error_message(exc)}")
        st.code(config_text, language=None)
        return
    page = _sidebar(st, config_text, config)
    pages = {
        "Overview": _overview,
        "Data & Trips": _data_and_trips,
        "Training": _training,
        "Evaluation": _evaluation,
        "Predict & Map": _predict_and_map,
        "Settings": _settings,
    }
    pages[page](st, config)
