"""Immediately runnable tree ensemble for destination-cell and coordinate prediction."""

from __future__ import annotations

import json
import logging
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import ExtraTreesRegressor, RandomForestClassifier

from ..config import AppConfig
from ..dataset import (
    FEATURE_COLUMNS,
    SplitManifest,
    coordinate_targets,
    feature_matrix,
    prepare_samples,
)
from ..evaluation import attach_geographic_error, metrics_by_group, prediction_metrics
from ..geo import GridEncoder


LOGGER = logging.getLogger(__name__)


@dataclass
class BaselineDestinationModel:
    classifier: RandomForestClassifier
    regressor: ExtraTreesRegressor
    grid: GridEncoder
    feature_columns: tuple[str, ...]
    split_manifest: SplitManifest
    config_hash: str

    def predict(self, samples: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
        matrix = feature_matrix(samples)
        coordinates = self.regressor.predict(matrix)
        probabilities = self.classifier.predict_proba(matrix)
        classes = np.asarray(self.classifier.classes_, dtype=object)
        k = max(1, min(int(top_k), len(classes)))
        ordered = np.argsort(probabilities, axis=1)[:, ::-1][:, :k]
        rows: list[dict[str, object]] = []
        for index, sample in samples.reset_index(drop=True).iterrows():
            candidates: list[dict[str, object]] = []
            actual_rank: int | None = None
            for rank, class_index in enumerate(ordered[index], start=1):
                cell = str(classes[class_index])
                latitude, longitude = self.grid.decode(cell)
                probability = float(probabilities[index, class_index])
                candidates.append(
                    {
                        "rank": rank,
                        "cell": cell,
                        "probability": probability,
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
        model_path = output / "baseline_model.joblib"
        temporary = output / "baseline_model.joblib.tmp"
        joblib.dump(self, temporary)
        temporary.replace(model_path)
        metadata = {
            "engine": "baseline",
            "config_hash": self.config_hash,
            "feature_columns": list(self.feature_columns),
            "grid": self.grid.to_dict(),
            "split_manifest": self.split_manifest.to_dict(),
            "python_version": platform.python_version(),
            "scikit_learn_version": sklearn.__version__,
        }
        metadata_path = output / "model_metadata.json"
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
        return {"model": model_path, "metadata": metadata_path}


def train_baseline(
    trips: pd.DataFrame,
    config: AppConfig,
    *,
    verbose: bool = False,
) -> tuple[BaselineDestinationModel, pd.DataFrame, dict[str, object], pd.DataFrame]:
    """Fit and evaluate the baseline model, optionally exposing estimator progress."""
    LOGGER.info("Preparing model samples from %s reconstructed trips", f"{len(trips):,}")
    samples, manifest = prepare_samples(trips, config.dataset)
    train = samples.loc[samples["split"].eq("train")].reset_index(drop=True)
    validation_count = int(samples["split"].eq("validation").sum())
    test_count = int(samples["split"].eq("test").sum())
    LOGGER.info(
        "Prepared %s samples: %s train, %s validation, %s test",
        f"{len(samples):,}",
        f"{len(train):,}",
        f"{validation_count:,}",
        f"{test_count:,}",
    )
    if train["actual_cell"].nunique() < 2:
        raise ValueError("Baseline classification requires at least two training cells")
    classifier = RandomForestClassifier(
        n_estimators=config.baseline.n_estimators,
        min_samples_leaf=config.baseline.min_samples_leaf,
        max_depth=config.baseline.max_depth,
        n_jobs=config.baseline.n_jobs,
        random_state=config.random_seed,
        class_weight="balanced_subsample",
        verbose=int(verbose),
    )
    regressor = ExtraTreesRegressor(
        n_estimators=config.baseline.n_estimators,
        min_samples_leaf=config.baseline.min_samples_leaf,
        max_depth=config.baseline.max_depth,
        n_jobs=config.baseline.n_jobs,
        random_state=config.random_seed,
        verbose=int(verbose),
    )
    matrix = feature_matrix(train)
    LOGGER.info(
        "Fitting baseline destination classifier with %d trees",
        config.baseline.n_estimators,
    )
    classifier.fit(matrix, train["actual_cell"].astype(str))
    LOGGER.info(
        "Fitting baseline coordinate regressor with %d trees",
        config.baseline.n_estimators,
    )
    regressor.fit(matrix, coordinate_targets(train))
    model = BaselineDestinationModel(
        classifier=classifier,
        regressor=regressor,
        grid=GridEncoder(config.dataset.grid_cell_size_degrees),
        feature_columns=FEATURE_COLUMNS,
        split_manifest=manifest,
        config_hash=config.reproducibility_hash,
    )
    LOGGER.info("Generating predictions for %s samples", f"{len(samples):,}")
    predictions = model.predict(samples, top_k=max(config.evaluation.top_k))
    test_predictions = predictions.loc[predictions["split"].eq("test")].copy()
    metrics = prediction_metrics(
        test_predictions,
        top_k=config.evaluation.top_k,
        recall_distances_km=config.evaluation.recall_distance_km,
    )
    grouped = metrics_by_group(predictions)
    LOGGER.info("Baseline training and evaluation complete")
    return model, predictions, metrics, grouped


def load_baseline(path: str | Path) -> BaselineDestinationModel:
    model = joblib.load(Path(path).expanduser().resolve())
    if not isinstance(model, BaselineDestinationModel):
        raise TypeError("The artifact is not a BaselineDestinationModel")
    if tuple(model.feature_columns) != FEATURE_COLUMNS:
        raise ValueError("The saved feature contract is incompatible with this code")
    return model
