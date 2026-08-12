"""Geographic primitives with no paid map or geocoding dependency."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Iterable, Sequence

import numpy as np


EARTH_RADIUS_KM = 6371.0088


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    """Vectorized great-circle distance in kilometres."""
    a_lat = np.radians(lat1)
    b_lat = np.radians(lat2)
    delta_lat = np.radians(np.asarray(lat2) - np.asarray(lat1))
    delta_lon = np.radians(np.asarray(lon2) - np.asarray(lon1))
    value = np.sin(delta_lat / 2) ** 2 + np.cos(a_lat) * np.cos(b_lat) * np.sin(
        delta_lon / 2
    ) ** 2
    distance = 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.clip(value, 0, 1)))
    return float(distance) if np.ndim(distance) == 0 else distance


def bearing_degrees(a: Sequence[float], b: Sequence[float]) -> float:
    lat1, lon1 = map(math.radians, a)
    lat2, lon2 = map(math.radians, b)
    y = math.sin(lon2 - lon1) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(
        lat2
    ) * math.cos(lon2 - lon1)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def path_distance_km(points: Iterable[Sequence[float]]) -> float:
    values = list(points)
    return float(
        sum(
            haversine_km(a[0], a[1], b[0], b[1])
            for a, b in zip(values, values[1:])
        )
    )


@dataclass(frozen=True)
class GridEncoder:
    """Portable spatial-cell encoder used when H3 is not installed."""

    cell_size_degrees: float = 0.01

    def encode(self, latitude: float, longitude: float) -> str:
        row = math.floor((float(latitude) + 90.0) / self.cell_size_degrees)
        column = math.floor((float(longitude) + 180.0) / self.cell_size_degrees)
        return f"g{row}:{column}"

    def decode(self, cell: str) -> tuple[float, float]:
        row_text, column_text = cell.removeprefix("g").split(":", 1)
        row, column = int(row_text), int(column_text)
        return (
            row * self.cell_size_degrees - 90.0 + self.cell_size_degrees / 2,
            column * self.cell_size_degrees - 180.0 + self.cell_size_degrees / 2,
        )

    def polygon(self, cell: str) -> list[list[float]]:
        latitude, longitude = self.decode(cell)
        half = self.cell_size_degrees / 2
        return [
            [longitude - half, latitude - half],
            [longitude + half, latitude - half],
            [longitude + half, latitude + half],
            [longitude - half, latitude + half],
            [longitude - half, latitude - half],
        ]

    def to_dict(self) -> dict[str, float]:
        return asdict(self)
