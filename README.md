# Vehicle Destination Lab

Complete, reproducible pipeline from long-format raw telemetry to mapped
vehicle-destination predictions.

The archive is immediately runnable with the bundled synthetic fixture and a
pretrained baseline artifact. It also contains an optional TensorFlow/Keras
multi-input model for larger datasets.

## What is included

- Raw-row cleaning with explicit discard reasons and immutable source data.
- Start/end trigger pairing and trajectory-array reconstruction.
- Model-ready trip and ordered point tables.
- Leakage-safe trajectory prefixes and prior-trip history.
- Deterministic, completely VIN-disjoint train/validation/test partitions.
- Fast tree-ensemble baseline with destination-cell ranking and coordinate
  regression.
- Optional GRU, LSTM, Bidirectional GRU, or Bidirectional LSTM model with
  trajectory, context, and history branches.
- Geographic, ranked-cell, cold/warm-history, and prefix-stage outputs.
- Reference-grade Streamlit workspace with a compact navy workflow rail,
  responsive research controls, cached artifact reads, data processing,
  training, evaluation, inference, and predicted-versus-actual maps.
- Fingerprint-aware model resource caching, so repeated prefix, trip, and
  Top-K predictions do not reload trained artifacts.
- VIN-first scenario inference: select a vehicle and reference trip, then edit
  departure time, origin, observed prefix points, prior-trip history, and Top-K.
- A request hash and per-input provenance record for every interactive, CLI,
  or notebook prediction.
- A bounded session prediction cache: reopening a previously viewed
  model/trip/prefix/Top-K combination is immediate and the cache is
  automatically invalidated after data reconstruction or retraining.
- CLI, Python service layer, tests, configuration, sample data, resolved run
  metadata, and generated demonstration artifacts.

## Project structure

```text
vehicle_destination_lab/
├── configs/default.yaml            # Single source of project settings
├── configs/inference_example.yaml  # Complete editable inference request
├── data/sample/raw_data.csv        # Deterministic 14-VIN execution fixture
├── artifacts/demo/                 # Prebuilt trips, baseline model, results
├── vehicle_destination/
│   ├── config.py                   # Typed validation and reproducibility hash
│   ├── trip_builder.py             # Raw cleaning and trip reconstruction
│   ├── dataset.py                  # Features, history, prefixes, VIN splits
│   ├── geo.py                      # Grid cells and Haversine calculations
│   ├── evaluation.py               # Scientific metrics
│   ├── inference.py                # VIN scenarios, validation and provenance
│   ├── pipeline.py                 # Reusable orchestration/service layer
│   ├── models/
│   │   ├── baseline.py             # Immediately runnable model
│   │   └── keras_model.py          # Multi-input sequence model
│   └── ui/
│       ├── app.py                  # Complete Streamlit workflow
│       ├── components.py           # Reusable panels, status and tables
│       ├── theme.py                # Visual tokens and responsive UI system
│       └── maps.py                 # Map-ready scientific layers
├── tests/                          # Regression, leakage, split, model, map tests
├── Makefile                        # Short commands for every workflow stage
├── scripts/run_demo.py             # One-command demonstration
├── notebooks/
│   └── end_to_end_destination_prediction.ipynb # Source-driven workflow
├── streamlit_app.py                # App entry point
└── pyproject.toml                  # Installable package and dependencies
```

## Quick start

Python 3.10–3.12 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the complete baseline pipeline:

```bash
python -m vehicle_destination --config configs/default.yaml run-all --engine baseline
```

Launch the application:

```bash
streamlit run streamlit_app.py -- --config configs/default.yaml
```

Or use the concise Make interface:

```bash
make setup-app
make clean       # data cleaning; raw source is never deleted
make trips
make train
make predict
make app
```

Long-running cleaning, trip reconstruction, and training commands support
bounded progress output. Verbose trip reconstruction includes a VIN-level
progress bar with accepted/rejected counts. Progress is written to stderr so
the final JSON on stdout remains usable by scripts:

```bash
make trips VERBOSE=1
make train ENGINE=baseline VERBOSE=1

# Equivalent CLI flags
python -m vehicle_destination --config configs/default.yaml build-trips --verbose
python -m vehicle_destination --config configs/default.yaml train --engine baseline --verbose
```

The application opens at `http://localhost:8501`. Because `artifacts/demo/`
is included, the sample trips, baseline evaluation, and map workflow are
available immediately.

### Jupyter notebook

Install and launch the notebook environment:

```bash
python -m pip install -r requirements-notebook.txt
jupyter lab notebooks/end_to_end_destination_prediction.ipynb
```

Run the cells in order. The notebook imports the public functions from the
`vehicle_destination` package and executes cleaning, trip reconstruction,
VIN-disjoint sample preparation, baseline training, evaluation, selected-trip
inference, and the interactive prediction-versus-actual map. Its generated
artifacts are isolated under `artifacts/notebook_run/`.

### Application workflow

The interface is organized in the same order as the research process:

1. **Overview** — readiness, artifacts, key metrics, and quick actions.
2. **Data & Trips** — upload or select telemetry, clean it, reconstruct trips,
   inspect rejection audits, and export model-ready CSV files.
3. **Training** — choose the baseline or Keras engine, confirm leakage-safe
   settings, run training, and inspect persisted model metadata.
4. **Evaluation** — select a partition, compare prefix stages, inspect error
   distributions, audit individual predictions, and export results.
5. **Predict & Map** — select the model, VIN and reference trip; optionally
   edit UTC departure, origin, observed trajectory, eligible completed history
   and Top-K; inspect exact model inputs and provenance; compare actual,
   refined and candidate destinations; export the result and scenario JSON.
6. **Settings** — inspect or export the resolved configuration, review leakage
   safeguards, and verify runtime versions.

CSV and JSON previews are cached using their file modification fingerprints,
so navigation and repeated predictions remain fast while changed artifacts are
invalidated automatically. Trained baseline and Keras models use the same
artifact-fingerprint policy and remain in memory across Streamlit reruns.

## Run each stage

```bash
# 1. Clean raw data and reconstruct trips
python -m vehicle_destination --config configs/default.yaml build-trips

# 2. Train the immediately runnable baseline
python -m vehicle_destination --config configs/default.yaml train --engine baseline

# 3. Evaluate saved untouched-test predictions
python -m vehicle_destination --config configs/default.yaml evaluate \
  --engine baseline --split test

# 4. Predict a selected trip and export map layers
python -m vehicle_destination --config configs/default.yaml predict \
  --engine baseline \
  --trip-id VIN000_20260213T070000Z_0001 \
  --prefix-fraction 0.25 \
  --top-k 5 \
  --map-json artifacts/demo/map_payload.json
```

### VIN-driven editable inference

The professional inference path begins with a VIN. A reference trip supplies
safe defaults and an optional actual destination for comparison, but the
destination is never part of the model input. Every model-relevant field can be
changed:

- departure timestamp in UTC;
- origin latitude and longitude;
- observed trajectory fraction;
- custom observed prefix points from JSON or CSV;
- completed history trips for the selected VIN;
- Top-K destination count.

Run a counterfactual scenario from flags:

```bash
python -m vehicle_destination --config configs/default.yaml predict-vin \
  --engine baseline \
  --vin VIN006 \
  --reference-trip-id VIN006_20260213T070000Z_0001 \
  --departure-time 2026-02-13T07:30:00+00:00 \
  --origin-latitude 48.9805 \
  --origin-longitude 2.3205 \
  --prefix-fraction 0.25 \
  --top-k 10 \
  --output-json artifacts/demo/inference/prediction.json \
  --map-html artifacts/demo/inference/prediction_map.html
```

Or use a validated request file:

```bash
python -m vehicle_destination --config configs/default.yaml predict-vin \
  --engine baseline \
  --request-file configs/inference_example.yaml \
  --output-json artifacts/demo/inference/prediction.json
```

Equivalent short commands:

```bash
make list-vins
make predict VIN=VIN006 \
  REFERENCE_TRIP=VIN006_20260213T070000Z_0001 \
  PREFIX=.25 TOP_K=10

make predict VIN=VIN006 \
  REFERENCE_TRIP=VIN006_20260213T070000Z_0001 \
  DEPARTURE=2026-02-13T07:30:00+00:00 \
  ORIGIN_LAT=48.9805 ORIGIN_LON=2.3205
```

The output bundle contains the resolved request, request hash, exact input
provenance, prediction, ranked candidates, reference-only actual destination,
and complete map payload.

## Make command reference

```text
make help        Show all commands and overrides
make setup       Install the core package
make setup-app   Install the package and Streamlit
make clean       Clean raw rows and write a preconstruction audit
make trips       Build accepted trips and ordered trajectory points
make train       Train ENGINE=baseline or ENGINE=keras
make evaluate    Evaluate the saved test partition
make run-all     Run trips, training and evaluation
make list-vins   List available VINs
make predict     Run VIN-driven inference and create JSON + HTML map
make app         Launch the Streamlit workspace
make notebook    Launch the source-driven Jupyter notebook
make test        Run the complete test suite
make check       Compile, test and smoke-test the CLI
make package     Build a clean, reproducible ZIP in dist/
```

The equivalent installed command is `vehicle-destination`.

## TensorFlow/Keras model

Install the optional sequence-model dependencies:

```bash
python -m pip install -r requirements-tensorflow.txt
python -m vehicle_destination --config configs/default.yaml train --engine keras
```

Architecture plotting installs `pydot` with the TensorFlow extra and also
requires the Graphviz `dot` executable. On Debian/Ubuntu:

```bash
sudo apt-get install graphviz
```

The notebook training result retains its in-memory model, so the text and
graphical summaries are available directly:

```python
model = keras_training.model
model.summary()
display(model.plot("artifacts/notebook_run/models/keras/model_architecture.png"))
```

Set `keras.encoder` to `GRU`, `LSTM`, `BidirectionalGRU`, or
`BidirectionalLSTM`. TensorFlow is imported only when this engine is selected;
the baseline and data workflow remain usable without it.

The sequence model uses:

1. A masked trajectory-prefix GRU/LSTM branch.
2. A normalized departure-context branch.
3. A masked prior-destination history branch.
4. Shared fusion layers.
5. A destination-cell softmax head and coordinate-residual head.

## Expected raw schema

The default input is one signal per row:

| Column | Purpose |
| --- | --- |
| `vin` | Grouping and splitting only; never a model input |
| `triggerOrContext` | Start, end, or periodic trajectory trigger |
| `name` | Latitude, longitude, or altitude signal name |
| `scalarValue` | Start/end trigger value |
| `timeSeriesValue` | JSON coordinate array for periodic data |
| `vehicleCollectionTime` | UTC event time |
| `inCdpTime` | Optional ingestion time |
| `fuel`, `hyb`, `model` | Optional vehicle metadata |

Adapt all names, trigger values, signals, limits, paths, model sizes, and
evaluation thresholds in `configs/default.yaml`.

## Data outputs

`artifacts/demo/trips/` contains:

- `trips.csv`: one row per accepted trip.
- `trip_points.csv`: ordered points with reconstructed timestamps.
- `cleaned_trip_source.csv`: only source rows used by accepted trips.
- `raw_cleaning_audit.csv`: row-removal counts by reason.
- `rejected_trips.csv`: rejected candidates and explanations.
- `trip_build_audit.json`: input/configuration fingerprints and reconciled
  counts.

The raw source is never modified.

`make clean` additionally writes the isolated preconstruction stage under
`artifacts/demo/cleaning/`: canonical cleaned rows, discarded row IDs/reasons,
the cleaning audit, and resolved configuration. In this project, `make clean`
means **clean data**; it never deletes files or raw telemetry.

## Leakage protections

- VIN is used only for grouping, history construction, and splitting.
- Every VIN belongs to exactly one of train, validation, or final test.
- The final destination and final GPS point never enter trajectory features.
- Editable inference history is restricted to completed trips belonging to the
  selected VIN and ending before the edited departure time.
- Custom prefix points must start at the configured origin and cannot include
  a destination supplied by the application.
- A target trip can use only trips completed before its departure.
- Prefix fractions are strictly below 100%.
- Preprocessing statistics and spatial vocabulary are fitted on training data.
- Destination-cell metrics remain `null` when a held-out cell is unseen; they
  are not silently reported as zero.

## Tests

```bash
python -m unittest discover -s tests -v
```

The normal suite mocks no scientific result and runs the full baseline path.
TensorFlow is optional; a clear installation message is tested when it is not
available.

## Important scientific limitations

- The bundled data has 70 deterministic synthetic trips. It is suitable for
  validating execution, schema, leakage controls, persistence, and maps—not
  for claiming real-world model accuracy.
- Raw coordinate arrays have no point-level timestamps. The trip builder
  interpolates them evenly between trip triggers and records that assumption.
- The portable geographic grid avoids a required H3 dependency. Replace or
  extend `GridEncoder` if exact H3 compatibility is required.
- Use a large, representative fleet, temporal monitoring, external validation,
  and operational privacy controls before production deployment.
