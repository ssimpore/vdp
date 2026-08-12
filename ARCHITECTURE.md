# Architecture and scientific contract

```mermaid
flowchart TD
    A[Raw signal rows] --> B[Cleaning and audit]
    B --> C[Trip reconstruction]
    C --> D[VIN-disjoint samples]
    D --> E[Baseline or Keras model]
    E --> F[Geographic evaluation]
    F --> G[VIN scenario builder]
    G --> H[Prediction and map]
```

## Stage contracts

| Stage | Input | Output | Main safeguard |
| --- | --- | --- | --- |
| Cleaning | Long telemetry table | Canonical usable rows | Original data unchanged |
| Reconstruction | Start/end/scalar/array signals | Trips and points | Every discard audited |
| Dataset | Accepted trips | Prefix/history samples | Destination and future hidden |
| Split | VIN list | Train/validation/test VINs | Complete group disjointness |
| Model | Training samples | Ranked cells and coordinates | VIN never a feature |
| Evaluation | Untouched predictions | Geographic/ranking metrics | Undefined remains undefined |
| VIN scenario | VIN, reference trip, editable known inputs | One exact feature row + provenance | History ends before departure; actual is comparison-only |
| Map | Trip and prediction | Trajectory/candidates/actual layers | Actual and predicted are distinct |

## Entry-point architecture

The CLI, Makefile, notebook and Streamlit application all call the same
`vehicle_destination.pipeline` and `vehicle_destination.inference` services.
They do not contain parallel implementations of cleaning, feature engineering,
training or prediction.

```mermaid
flowchart LR
    A[Make or CLI] --> D[Pipeline services]
    B[Jupyter] --> D
    C[Streamlit] --> D
    D --> E[Artifacts and models]
```

`VinInferenceRequest` is the single scenario contract. It validates all editable
inputs, produces a deterministic request hash, and records whether departure,
origin, prefix and history were inherited or overridden. The selected VIN is
used to locate eligible history and the saved split only; it is never added to
`FEATURE_COLUMNS`.

## Baseline

The baseline trains a random-forest destination-cell classifier and an
extra-trees latitude/longitude regressor on departure-time, partial-trajectory,
calendar, origin, and earlier-history features. It is the default because it is
fast, inspectable, and reliable on small execution fixtures.

## Sequence model

```mermaid
flowchart TD
    A[Observed trajectory prefix] --> D[GRU or LSTM]
    B[Departure context] --> E[Dense encoder]
    C[Earlier completed trips] --> F[History GRU]
    D --> G[Fusion]
    E --> G
    F --> G
    G --> H[Destination cell]
    G --> I[Coordinate residual]
```

The classification vocabulary is learned from training VINs only. Unknown
validation/test cells receive zero classification sample weight while their
coordinate errors remain evaluable.
