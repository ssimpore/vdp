# Repository Guidelines

## Project Structure & Module Organization

`vehicle_destination/` is the shared implementation used by every interface. Raw cleaning and reconstruction live in `trip_builder.py`; leakage-safe prefixes, history, and VIN-disjoint splits live in `dataset.py`; training implementations are under `models/`; and `pipeline.py` orchestrates persisted stages. Keep the CLI (`__main__.py`), Streamlit UI (`ui/` and `streamlit_app.py`), scripts, and notebook as clients of those services rather than creating parallel data or model logic.

`configs/default.yaml` is the main runtime contract. `data/sample/` is the deterministic execution fixture. `artifacts/demo/` and `artifacts/notebook_run/` contain generated trips, predictions, models, audits, and resolved configuration snapshots. Tests are standard-library `unittest` modules under `tests/`.

## Build, Test, and Development Commands

- `make setup` installs the core editable package; `make setup-app` adds Streamlit; `make setup-all` includes notebook, TensorFlow, and Parquet extras.
- `make clean` performs raw-data cleaning and writes audits; it does not delete files.
- `make trips` reconstructs model-ready trips. `make train ENGINE=baseline` trains the default model; use `ENGINE=keras` only with TensorFlow installed.
- `make run-all` runs trip building, training, and evaluation. `make app PORT=8501` launches the Streamlit workspace.
- `make test` runs the complete suite. Run one test with `python -m unittest tests.test_trip_builder.TripBuilderTests.test_sample_reconstructs_expected_trips_and_points -v`.
- `make check` compiles Python, runs tests, and smoke-tests CLI inference. Use it before handoff when the change affects executable code.

## Coding Style & Scientific Invariants

Match the existing typed Python style: four-space indentation, `snake_case` functions, `PascalCase` dataclasses/classes, module docstrings, and type annotations. No formatter, linter, or static type checker is configured; `make check` is the repository’s executable validation. Keep TensorFlow and Streamlit optional imports lazy.

Preserve immutable raw inputs, configuration hashes, atomic artifact writes, and explicit rejection audits. VIN may group history and define splits but must never become a model feature. Train, validation, and test VINs must remain disjoint; destinations, final trajectory points, and future trips must not enter features. Preserve `null` for undefined ranked-cell metrics instead of converting it to zero.

## Testing Guidelines

Add regression coverage to the closest existing module: trip reconstruction, pipeline/model behavior, VIN inference, or UI/maps. Tests use the bundled synthetic fixture and temporary directories. TensorFlow-specific execution is optional; the normal suite must remain usable without a live TensorFlow training run. For Streamlit changes, use `streamlit.testing.v1.AppTest` and verify bounded table rendering and page exceptions.
