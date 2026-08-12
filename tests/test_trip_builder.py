import json
from pathlib import Path
import unittest

import pandas as pd

from vehicle_destination.config import load_config
from vehicle_destination.trip_builder import build_trips, clean_raw_data


ROOT = Path(__file__).resolve().parents[1]


class TripBuilderTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "default.yaml")
        cls.raw = pd.read_csv(ROOT / "data" / "sample" / "raw_data.csv")

    def test_sample_reconstructs_expected_trips_and_points(self):
        result = build_trips(self.raw, self.config.trip_builder)
        self.assertEqual(len(result.trips), 70)
        self.assertEqual(len(result.points), 406)
        self.assertEqual(result.trips["VIN"].nunique(), 14)
        self.assertTrue(result.rejected.empty)
        self.assertEqual(result.audit["raw_rows_removed_from_final_source"], 0)

    def test_cleaning_removes_irrelevant_and_duplicate_rows(self):
        extra = self.raw.iloc[[0]].copy()
        irrelevant = self.raw.iloc[[0]].copy()
        irrelevant["triggerOrContext"] = "Vehicle/Context/Unused"
        irrelevant["name"] = "ambientTemperature"
        dirty = pd.concat([self.raw, extra, irrelevant], ignore_index=True)
        cleaned = clean_raw_data(dirty, self.config.trip_builder)
        self.assertEqual(
            cleaned.audit["preconstruction_discard_counts"],
            {"exact_duplicate": 1, "irrelevant_trigger_or_signal": 1},
        )

    def test_trip_build_reports_monotonic_vin_progress(self):
        updates = []

        result = build_trips(
            self.raw,
            self.config.trip_builder,
            progress=updates.append,
        )

        self.assertEqual(len(updates), 15)
        self.assertEqual(updates[0].completed_vins, 0)
        self.assertEqual(updates[0].total_vins, 14)
        self.assertEqual(
            [update.completed_vins for update in updates],
            list(range(15)),
        )
        self.assertEqual(updates[-1].fraction, 1.0)
        self.assertEqual(updates[-1].accepted_trips, len(result.trips))
        self.assertEqual(updates[-1].rejected_trips, len(result.rejected))

    def test_trajectory_contract_is_ordered_and_model_ready(self):
        result = build_trips(self.raw, self.config.trip_builder)
        trip = result.trips.iloc[0]
        trajectory = json.loads(trip["Trajectory"])
        self.assertEqual(trajectory[0], [trip["start_latitude"], trip["start_longitude"]])
        self.assertEqual(trajectory[-1], [trip["end_latitude"], trip["end_longitude"]])
        self.assertEqual(trip["trajectory_timestamp_method"], "interpolated_between_trip_triggers")


if __name__ == "__main__":
    unittest.main()
