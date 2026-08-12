"""Geographic and ranked-cell evaluation with explicit undefined metrics."""

from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .geo import haversine_km


def prediction_metrics(
    predictions: pd.DataFrame,
    *,
    top_k: Sequence[int] = (1, 3, 5),
    recall_distances_km: Iterable[float] = (1.0, 5.0, 10.0, 25.0),
) -> dict[str, float | int | None]:
    if predictions.empty:
        raise ValueError("Cannot evaluate an empty prediction table")
    errors = predictions["error_km"].to_numpy(dtype=float)
    result: dict[str, float | int | None] = {
        "sample_count": int(len(predictions)),
        "trip_count": int(predictions["trip_id"].nunique()),
        "vin_count": int(predictions["VIN"].nunique()),
        "mean_haversine_km": float(np.mean(errors)),
        "median_haversine_km": float(np.median(errors)),
        "p90_haversine_km": float(np.quantile(errors, 0.9)),
    }
    for distance in recall_distances_km:
        result[f"recall_within_{distance:g}km"] = float(np.mean(errors <= distance))
    ranks = pd.to_numeric(predictions["actual_rank"], errors="coerce")
    defined = ranks.notna()
    result["cell_metrics_defined_samples"] = int(defined.sum())
    result["cell_metrics_undefined_samples"] = int((~defined).sum())
    for k in top_k:
        result[f"top_{k}_cell_accuracy"] = (
            float(np.mean(ranks.loc[defined] <= k)) if defined.any() else None
        )
    result["mean_reciprocal_rank"] = (
        float(np.mean(1.0 / ranks.loc[defined])) if defined.any() else None
    )
    return result


def metrics_by_group(
    predictions: pd.DataFrame,
    group_columns: Sequence[str] = ("split", "prefix_fraction"),
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for group_column in group_columns:
        if group_column not in predictions.columns:
            continue
        for group_value, group in predictions.groupby(group_column, dropna=False):
            rows.append(
                {
                    "group": group_column,
                    "value": group_value,
                    **prediction_metrics(group),
                }
            )
    return pd.DataFrame(rows)


def attach_geographic_error(predictions: pd.DataFrame) -> pd.DataFrame:
    frame = predictions.copy()
    frame["error_km"] = haversine_km(
        frame["actual_latitude"].to_numpy(),
        frame["actual_longitude"].to_numpy(),
        frame["predicted_latitude"].to_numpy(),
        frame["predicted_longitude"].to_numpy(),
    )
    return frame


def serialize_metrics(metrics: dict[str, object]) -> str:
    return json.dumps(metrics, indent=2, sort_keys=True, allow_nan=False) + "\n"
