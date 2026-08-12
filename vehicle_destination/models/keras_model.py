"""TensorFlow/Keras multi-input sequence model with lazy TensorFlow imports."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import AppConfig
from ..dataset import FEATURE_COLUMNS, SplitManifest, feature_matrix, prepare_samples
from ..evaluation import attach_geographic_error, prediction_metrics
from ..geo import GridEncoder, bearing_degrees, haversine_km


LOGGER = logging.getLogger(__name__)


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "TensorFlow is not installed. Install the ML extras with "
            "`python -m pip install -e '.[tensorflow]'`, or use --engine baseline."
        ) from exc
    return tf


def _require_plot_dependencies() -> None:
    """Validate optional model-plotting dependencies when plotting is requested."""
    try:
        __import__("pydot")
    except ImportError as exc:
        raise RuntimeError(
            "Keras model plotting requires pydot. Install the TensorFlow extra with "
            "`python -m pip install -e '.[tensorflow]'`."
        ) from exc
    if shutil.which("dot") is None:
        raise RuntimeError(
            "Keras model plotting requires the Graphviz `dot` executable on PATH. "
            "Install Graphviz with your operating system package manager."
        )


@dataclass
class KerasPreprocessor:
    context_mean: np.ndarray
    context_scale: np.ndarray
    vocabulary: tuple[str, ...]
    grid: GridEncoder
    max_points: int
    max_history: int
    split_manifest: SplitManifest
    config_hash: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_mean": self.context_mean.tolist(),
            "context_scale": self.context_scale.tolist(),
            "vocabulary": list(self.vocabulary),
            "grid": self.grid.to_dict(),
            "max_points": self.max_points,
            "max_history": self.max_history,
            "split_manifest": self.split_manifest.to_dict(),
            "config_hash": self.config_hash,
            "feature_columns": list(FEATURE_COLUMNS),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "KerasPreprocessor":
        if tuple(raw["feature_columns"]) != FEATURE_COLUMNS:
            raise ValueError("Saved Keras feature contract is incompatible")
        return cls(
            context_mean=np.asarray(raw["context_mean"], dtype=np.float32),
            context_scale=np.asarray(raw["context_scale"], dtype=np.float32),
            vocabulary=tuple(str(value) for value in raw["vocabulary"]),
            grid=GridEncoder(**raw["grid"]),
            max_points=int(raw["max_points"]),
            max_history=int(raw["max_history"]),
            split_manifest=SplitManifest.from_dict(raw["split_manifest"]),
            config_hash=str(raw["config_hash"]),
        )


def fit_preprocessor(
    samples: pd.DataFrame, config: AppConfig, manifest: SplitManifest
) -> KerasPreprocessor:
    training = samples.loc[samples["split"].eq("train")]
    matrix = feature_matrix(training)
    mean = matrix.mean(axis=0)
    scale = matrix.std(axis=0)
    scale[scale < 1e-8] = 1.0
    vocabulary = tuple(sorted(training["actual_cell"].astype(str).unique()))
    if len(vocabulary) < 2:
        raise ValueError("Keras classification requires at least two training cells")
    return KerasPreprocessor(
        context_mean=mean.astype(np.float32),
        context_scale=scale.astype(np.float32),
        vocabulary=vocabulary,
        grid=GridEncoder(config.dataset.grid_cell_size_degrees),
        max_points=config.dataset.max_trajectory_points,
        max_history=config.dataset.max_history_trips,
        split_manifest=manifest,
        config_hash=config.reproducibility_hash,
    )


def _trajectory_tensor(samples: pd.DataFrame, maximum: int) -> np.ndarray:
    tensor = np.zeros((len(samples), maximum, 6), dtype=np.float32)
    for row_index, (_, row) in enumerate(samples.iterrows()):
        points = row["trajectory_prefix"][-maximum:]
        origin = points[0]
        for point_index, point in enumerate(points):
            if point_index:
                distance = haversine_km(
                    points[point_index - 1][0],
                    points[point_index - 1][1],
                    point[0],
                    point[1],
                )
                bearing = bearing_degrees(points[point_index - 1], point)
            else:
                distance, bearing = 0.0, 0.0
            tensor[row_index, point_index] = [
                (point[0] - origin[0]) * 100.0,
                (point[1] - origin[1]) * 100.0,
                distance,
                np.sin(np.radians(bearing)),
                np.cos(np.radians(bearing)),
                (point_index + 1) / max(len(points), 1),
            ]
    return tensor


def _history_tensor(samples: pd.DataFrame, maximum: int) -> np.ndarray:
    tensor = np.zeros((len(samples), maximum, 4), dtype=np.float32)
    if maximum == 0:
        return tensor
    for row_index, (_, row) in enumerate(samples.iterrows()):
        history = row["history_sequence"][-maximum:]
        origin_latitude = float(row["origin_latitude"])
        origin_longitude = float(row["origin_longitude"])
        start_timestamp = pd.Timestamp(row["StartTrip"]).timestamp()
        offset = maximum - len(history)
        for position, item in enumerate(history):
            age_days = max(0.0, (start_timestamp - item["completed_at"]) / 86400.0)
            tensor[row_index, offset + position] = [
                (item["latitude"] - origin_latitude) * 100.0,
                (item["longitude"] - origin_longitude) * 100.0,
                np.log1p(age_days),
                (position + 1) / max(len(history), 1),
            ]
    return tensor


def keras_arrays(
    samples: pd.DataFrame, preprocessor: KerasPreprocessor
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    context = (feature_matrix(samples) - preprocessor.context_mean) / preprocessor.context_scale
    inputs = {
        "trajectory": _trajectory_tensor(samples, preprocessor.max_points),
        "context": context.astype(np.float32),
        "history": _history_tensor(samples, preprocessor.max_history),
    }
    vocabulary = {cell: index for index, cell in enumerate(preprocessor.vocabulary)}
    labels = np.asarray(
        [vocabulary.get(str(cell), 0) for cell in samples["actual_cell"]], dtype=np.int32
    )
    known = np.asarray(
        [float(str(cell) in vocabulary) for cell in samples["actual_cell"]],
        dtype=np.float32,
    )
    origins = samples[["origin_latitude", "origin_longitude"]].to_numpy(dtype=np.float32)
    destinations = samples[["actual_latitude", "actual_longitude"]].to_numpy(
        dtype=np.float32
    )
    targets = {
        "cell": labels,
        "coordinate": (destinations - origins) * 100.0,
    }
    weights = {"cell": known, "coordinate": np.ones(len(samples), dtype=np.float32)}
    return inputs, targets, weights


def build_model(config: AppConfig, number_of_cells: int):
    tf = _tensorflow()
    keras = tf.keras
    trajectory_input = keras.Input(
        shape=(config.dataset.max_trajectory_points, 6), name="trajectory"
    )
    context_input = keras.Input(shape=(len(FEATURE_COLUMNS),), name="context")
    history_input = keras.Input(
        shape=(config.dataset.max_history_trips, 4), name="history"
    )

    recurrent_name = config.keras.encoder.removeprefix("Bidirectional")
    recurrent_class = keras.layers.GRU if recurrent_name == "GRU" else keras.layers.LSTM

    trajectory = keras.layers.Masking()(trajectory_input)
    for layer_index in range(config.keras.recurrent_layers):
        recurrent = recurrent_class(
            config.keras.recurrent_units,
            return_sequences=layer_index < config.keras.recurrent_layers - 1,
            dropout=config.keras.dropout,
            name=f"trajectory_{recurrent_name.lower()}_{layer_index + 1}",
        )
        if config.keras.encoder.startswith("Bidirectional"):
            recurrent = keras.layers.Bidirectional(recurrent)
        trajectory = recurrent(trajectory)
    trajectory = keras.layers.LayerNormalization()(trajectory)

    context = keras.layers.Dense(64, activation="gelu")(context_input)
    context = keras.layers.Dropout(config.keras.dropout)(context)

    history = keras.layers.Masking()(history_input)
    history = keras.layers.GRU(
        config.keras.recurrent_units,
        dropout=config.keras.dropout,
        name="history_gru",
    )(history)
    history = keras.layers.LayerNormalization()(history)

    fused = keras.layers.Concatenate()([trajectory, context, history])
    for units in config.keras.dense_units:
        residual = fused
        fused = keras.layers.Dense(units, activation="gelu")(fused)
        fused = keras.layers.LayerNormalization()(fused)
        fused = keras.layers.Dropout(config.keras.dropout)(fused)
        if residual.shape[-1] == units:
            fused = keras.layers.Add()([residual, fused])
    cell = keras.layers.Dense(number_of_cells, activation="softmax", name="cell")(fused)
    coordinate = keras.layers.Dense(2, name="coordinate")(fused)
    model = keras.Model(
        inputs={
            "trajectory": trajectory_input,
            "context": context_input,
            "history": history_input,
        },
        outputs={"cell": cell, "coordinate": coordinate},
        name="vehicle_destination_multi_input",
    )
    optimizer = keras.optimizers.AdamW(
        learning_rate=config.keras.learning_rate, clipnorm=1.0
    )
    model.compile(
        optimizer=optimizer,
        loss={
            "cell": keras.losses.SparseCategoricalCrossentropy(),
            "coordinate": keras.losses.Huber(delta=0.1),
        },
        loss_weights={"cell": 0.5, "coordinate": 0.5},
        weighted_metrics={
            "cell": [keras.metrics.SparseTopKCategoricalAccuracy(k=3)]
        },
    )
    return model


class KerasDestinationModel:
    def __init__(self, model: Any, preprocessor: KerasPreprocessor):
        self.model = model
        self.preprocessor = preprocessor

    def summary(self, *args: Any, **kwargs: Any) -> None:
        """Display the wrapped native Keras model summary."""
        self.model.summary(*args, **kwargs)

    def plot(
        self,
        to_file: str | Path = "model.png",
        *,
        show_shapes: bool = True,
        show_dtype: bool = False,
        show_layer_names: bool = True,
        rankdir: str = "TB",
        expand_nested: bool = True,
        dpi: int = 160,
        show_layer_activations: bool = False,
        show_trainable: bool = True,
        splines: str = "ortho",
    ) -> Any:
        """Render the wrapped Keras architecture and return its notebook image."""
        _require_plot_dependencies()
        output = Path(to_file).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        tf = _tensorflow()
        return tf.keras.utils.plot_model(
            self.model,
            to_file=str(output),
            show_shapes=show_shapes,
            show_dtype=show_dtype,
            show_layer_names=show_layer_names,
            rankdir=rankdir,
            expand_nested=expand_nested,
            dpi=dpi,
            show_layer_activations=show_layer_activations,
            show_trainable=show_trainable,
            splines=splines,
        )

    def predict(self, samples: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        inputs, _, _ = keras_arrays(samples, self.preprocessor)
        output = self.model.predict(inputs, verbose=0)
        cell_probabilities = np.asarray(output["cell"])
        residuals = np.asarray(output["coordinate"]) / 100.0
        origins = samples[["origin_latitude", "origin_longitude"]].to_numpy(dtype=float)
        coordinates = origins + residuals
        k = min(max(1, int(top_k)), len(self.preprocessor.vocabulary))
        ordered = np.argsort(cell_probabilities, axis=1)[:, ::-1][:, :k]
        rows: list[dict[str, object]] = []
        for index, sample in samples.reset_index(drop=True).iterrows():
            candidates = []
            actual_rank = None
            for rank, class_index in enumerate(ordered[index], start=1):
                cell = self.preprocessor.vocabulary[class_index]
                latitude, longitude = self.preprocessor.grid.decode(cell)
                candidates.append(
                    {
                        "rank": rank,
                        "cell": cell,
                        "probability": float(cell_probabilities[index, class_index]),
                        "latitude": latitude,
                        "longitude": longitude,
                    }
                )
                if cell == str(sample["actual_cell"]):
                    actual_rank = rank
            rows.append(
                {
                    "sample_id": sample["sample_id"],
                    "VIN": sample["VIN"],
                    "vehicle_id": sample["vehicle_id"],
                    "trip_id": sample["trip_id"],
                    "split": sample["split"],
                    "prefix_fraction": float(sample["prefix_fraction"]),
                    "origin_latitude": float(sample["origin_latitude"]),
                    "origin_longitude": float(sample["origin_longitude"]),
                    "actual_latitude": float(sample["actual_latitude"]),
                    "actual_longitude": float(sample["actual_longitude"]),
                    "actual_cell": str(sample["actual_cell"]),
                    "predicted_latitude": float(coordinates[index, 0]),
                    "predicted_longitude": float(coordinates[index, 1]),
                    "predicted_cell": candidates[0]["cell"],
                    "predicted_cell_probability": candidates[0]["probability"],
                    "actual_rank": actual_rank,
                    "top_k_candidates": json.dumps(candidates, separators=(",", ":")),
                }
            )
        return attach_geographic_error(pd.DataFrame(rows))

    def save(self, directory: str | Path) -> dict[str, Path]:
        output = Path(directory).expanduser().resolve()
        output.mkdir(parents=True, exist_ok=True)
        model_path = output / "destination_model.keras"
        metadata_path = output / "keras_preprocessor.json"
        self.model.save(model_path)
        metadata_path.write_text(
            json.dumps(self.preprocessor.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        return {"model": model_path, "metadata": metadata_path}


def train_keras(
    trips: pd.DataFrame,
    config: AppConfig,
    output_dir: str | Path | None = None,
    *,
    verbose: bool = False,
) -> tuple[KerasDestinationModel, pd.DataFrame, dict[str, object], Any]:
    """Fit and evaluate the Keras model with optional per-epoch progress."""
    tf = _tensorflow()
    tf.keras.utils.set_random_seed(config.random_seed)
    LOGGER.info("Preparing model samples from %s reconstructed trips", f"{len(trips):,}")
    samples, manifest = prepare_samples(trips, config.dataset)
    preprocessor = fit_preprocessor(samples, config, manifest)
    training = samples.loc[samples["split"].eq("train")].reset_index(drop=True)
    validation = samples.loc[samples["split"].eq("validation")].reset_index(drop=True)
    test_count = int(samples["split"].eq("test").sum())
    LOGGER.info(
        "Prepared %s samples: %s train, %s validation, %s test",
        f"{len(samples):,}",
        f"{len(training):,}",
        f"{len(validation):,}",
        f"{test_count:,}",
    )
    LOGGER.info("Creating Keras tensors and preprocessing state")
    train_inputs, train_targets, train_weights = keras_arrays(training, preprocessor)
    val_inputs, val_targets, val_weights = keras_arrays(validation, preprocessor)
    model = build_model(config, len(preprocessor.vocabulary))
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=config.keras.early_stopping_patience,
            restore_best_weights=True,
        )
    ]
    if output_dir is not None:
        checkpoint = Path(output_dir).expanduser().resolve() / "best_model.keras"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            tf.keras.callbacks.ModelCheckpoint(
                checkpoint, monitor="val_loss", save_best_only=True
            )
        )
    if verbose:
        def log_epoch(epoch: int, logs: dict[str, float] | None = None) -> None:
            values = ", ".join(
                f"{name}={float(value):.4g}"
                for name, value in sorted((logs or {}).items())
            )
            LOGGER.info(
                "Keras epoch %d/%d%s",
                epoch + 1,
                config.keras.epochs,
                f": {values}" if values else "",
            )

        callbacks.append(tf.keras.callbacks.LambdaCallback(on_epoch_end=log_epoch))
    LOGGER.info(
        "Fitting %s for up to %d epochs",
        config.keras.encoder,
        config.keras.epochs,
    )
    history = model.fit(
        train_inputs,
        train_targets,
        sample_weight=train_weights,
        validation_data=(val_inputs, val_targets, val_weights),
        batch_size=config.keras.batch_size,
        epochs=config.keras.epochs,
        callbacks=callbacks,
        verbose=0,
    )
    bundle = KerasDestinationModel(model, preprocessor)
    LOGGER.info("Generating predictions for %s samples", f"{len(samples):,}")
    predictions = bundle.predict(samples, top_k=max(config.evaluation.top_k))
    test = predictions.loc[predictions["split"].eq("test")]
    metrics = prediction_metrics(
        test,
        top_k=config.evaluation.top_k,
        recall_distances_km=config.evaluation.recall_distance_km,
    )
    LOGGER.info("Keras training and evaluation complete")
    return bundle, predictions, metrics, history


def load_keras(directory: str | Path) -> KerasDestinationModel:
    tf = _tensorflow()
    root = Path(directory).expanduser().resolve()
    model = tf.keras.models.load_model(root / "destination_model.keras")
    metadata = json.loads((root / "keras_preprocessor.json").read_text(encoding="utf-8"))
    return KerasDestinationModel(model, KerasPreprocessor.from_dict(metadata))
