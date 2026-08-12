from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from vehicle_destination.config import load_config
from vehicle_destination.inference import (
    VinInferenceRequest,
    inference_reference,
    prepare_vin_inference,
)
from vehicle_destination.models.baseline import train_baseline
from vehicle_destination.pipeline import predict_vin_scenario
from vehicle_destination.trip_builder import build_trips


ROOT = Path(__file__).resolve().parents[1]


class VinInferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = load_config(ROOT / "configs" / "default.yaml")
        cls.config = replace(
            original,
            baseline=replace(original.baseline, n_estimators=24, n_jobs=1),
        )
        raw = pd.read_csv(ROOT / "data" / "sample" / "raw_data.csv")
        cls.trips = build_trips(raw, cls.config.trip_builder).trips
        cls.model, _, _, _ = train_baseline(cls.trips, cls.config)
        cls.vin = "VIN006"
        cls.trip_id = "VIN006_20260213T130000Z_0003"

    def test_reference_prefills_vin_inputs_and_prior_history(self):
        reference = inference_reference(
            self.trips,
            vin=self.vin,
            reference_trip_id=self.trip_id,
            prefix_fraction=0.25,
            history_limit=20,
        )
        self.assertEqual(reference["vin"], self.vin)
        self.assertEqual(len(reference["history_trip_ids"]), 2)
        self.assertGreaterEqual(len(reference["prefix_points"]), 1)

    def test_editable_scenario_builds_exact_feature_contract_and_provenance(self):
        request = VinInferenceRequest(
            vin=self.vin,
            reference_trip_id=self.trip_id,
            departure_time="2026-02-13T14:00:00+00:00",
            origin_latitude=48.987,
            origin_longitude=2.329,
            prefix_fraction=0.25,
            history_trip_ids=("VIN006_20260213T070000Z_0001",),
            top_k=3,
        )
        prepared = prepare_vin_inference(
            self.trips, self.config.dataset, request, self.model.split_manifest
        )
        sample = prepared.sample.iloc[0]
        self.assertAlmostEqual(sample["origin_latitude"], 48.987)
        self.assertAlmostEqual(sample["origin_longitude"], 2.329)
        self.assertEqual(sample["history_count"], 1.0)
        self.assertEqual(
            prepared.provenance["input_sources"]["origin"], "user_override"
        )
        self.assertEqual(
            prepared.provenance["input_sources"]["trajectory_prefix"],
            "translated_reference_trip",
        )
        self.assertFalse(
            prepared.provenance["scientific_contract"]["vin_is_model_feature"]
        )

    def test_future_or_current_trip_cannot_be_selected_as_history(self):
        request = VinInferenceRequest(
            vin=self.vin,
            reference_trip_id=self.trip_id,
            history_trip_ids=("VIN006_20260213T160000Z_0004",),
        )
        with self.assertRaisesRegex(ValueError, "end before departure"):
            prepare_vin_inference(
                self.trips, self.config.dataset, request, self.model.split_manifest
            )

    def test_custom_prefix_must_start_at_origin(self):
        request = VinInferenceRequest(
            vin=self.vin,
            reference_trip_id=self.trip_id,
            origin_latitude=48.0,
            origin_longitude=2.0,
            prefix_fraction=0.25,
            prefix_points=((49.0, 3.0), (49.01, 3.01)),
        )
        with self.assertRaisesRegex(ValueError, "start at the configured origin"):
            prepare_vin_inference(
                self.trips, self.config.dataset, request, self.model.split_manifest
            )

    def test_request_hash_is_stable_and_changes_with_inputs(self):
        first = VinInferenceRequest(self.vin, self.trip_id, top_k=5)
        same = VinInferenceRequest(self.vin, self.trip_id, top_k=5)
        changed = VinInferenceRequest(self.vin, self.trip_id, top_k=10)
        self.assertEqual(first.reproducibility_hash, same.reproducibility_hash)
        self.assertNotEqual(first.reproducibility_hash, changed.reproducibility_hash)

    def test_pipeline_predicts_scenario_and_persists_provenance_column(self):
        request = VinInferenceRequest(
            self.vin,
            self.trip_id,
            prefix_fraction=0.5,
            top_k=3,
        )
        with tempfile.TemporaryDirectory() as directory:
            trips_path = Path(directory) / "trips.csv"
            self.trips.to_csv(trips_path, index=False)
            result = predict_vin_scenario(
                self.config,
                request=request,
                engine="baseline",
                trips_path=trips_path,
                loaded_model=self.model,
            )
        self.assertEqual(len(result.prediction), 1)
        self.assertEqual(result.prediction.iloc[0]["scenario_id"], request.reproducibility_hash)
        self.assertIn("input_provenance", result.prediction.columns)
        self.assertEqual(len(result.sample.iloc[0]["trajectory_prefix"]), 3)


if __name__ == "__main__":
    unittest.main()
