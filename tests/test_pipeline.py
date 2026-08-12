from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import pandas as pd

from vehicle_destination.config import load_config, save_resolved_config
from vehicle_destination.dataset import FEATURE_COLUMNS, make_vin_split, prepare_samples
from vehicle_destination.evaluation import prediction_metrics
from vehicle_destination.models.baseline import (
    BaselineDestinationModel,
    load_baseline,
    train_baseline,
)
from vehicle_destination.models.keras_model import KerasDestinationModel, _tensorflow
from vehicle_destination.pipeline import train_model
from vehicle_destination.trip_builder import build_trips
from vehicle_destination.ui.maps import build_leaflet_html, build_map_payload


ROOT = Path(__file__).resolve().parents[1]


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        original = load_config(ROOT / "configs" / "default.yaml")
        cls.config = replace(
            original,
            baseline=replace(original.baseline, n_estimators=24, n_jobs=1),
        )
        raw = pd.read_csv(ROOT / "data" / "sample" / "raw_data.csv")
        cls.trips = build_trips(raw, cls.config.trip_builder).trips
        cls.model, cls.predictions, cls.metrics, cls.grouped = train_baseline(
            cls.trips, cls.config
        )

    def test_configuration_hash_and_resolved_snapshot(self):
        self.assertEqual(len(self.config.reproducibility_hash), 64)
        with tempfile.TemporaryDirectory() as directory:
            path = save_resolved_config(self.config, Path(directory) / "resolved.yaml")
            text = path.read_text(encoding="utf-8")
            self.assertIn(self.config.reproducibility_hash, text)

    def test_vin_split_is_deterministic_and_disjoint(self):
        first = make_vin_split(self.trips["VIN"], self.config.dataset)
        second = make_vin_split(self.trips["VIN"], self.config.dataset)
        self.assertEqual(first, second)
        self.assertFalse(set(first.train_vins) & set(first.validation_vins))
        self.assertFalse(set(first.train_vins) & set(first.test_vins))
        self.assertFalse(set(first.validation_vins) & set(first.test_vins))

    def test_features_exclude_vin_and_destinations(self):
        self.assertNotIn("VIN", FEATURE_COLUMNS)
        self.assertNotIn("actual_latitude", FEATURE_COLUMNS)
        samples, _ = prepare_samples(self.trips, self.config.dataset)
        self.assertGreater(len(samples), len(self.trips))

    def test_prefix_never_contains_final_destination(self):
        samples, _ = prepare_samples(self.trips, self.config.dataset)
        for _, row in samples.iterrows():
            self.assertLess(len(row["trajectory_prefix"]), len(row["trajectory"]))

    def test_history_contains_only_completed_past_trips(self):
        samples, _ = prepare_samples(self.trips, self.config.dataset)
        for _, row in samples.iterrows():
            start = pd.Timestamp(row["StartTrip"]).timestamp()
            self.assertTrue(all(item["completed_at"] < start for item in row["history_sequence"]))

    def test_baseline_end_to_end_and_test_metrics(self):
        self.assertEqual(self.metrics["sample_count"], 30)
        self.assertEqual(self.metrics["vin_count"], 2)
        self.assertTrue((self.predictions["error_km"] >= 0).all())
        self.assertIn("prefix_fraction", self.grouped["group"].tolist())
        self.assertEqual(self.model.classifier.verbose, 0)
        self.assertEqual(self.model.regressor.verbose, 0)

    def test_baseline_verbose_enables_estimator_progress(self):
        model, _, _, _ = train_baseline(self.trips, self.config, verbose=True)

        self.assertEqual(model.classifier.verbose, 1)
        self.assertEqual(model.regressor.verbose, 1)

    def test_training_stage_exposes_in_memory_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            trips_path = root / "trips.csv"
            self.trips.to_csv(trips_path, index=False)
            result = train_model(
                self.config,
                engine="baseline",
                trips_path=trips_path,
                output_dir=root / "model",
            )

        self.assertIsInstance(result.model, BaselineDestinationModel)

    def test_keras_wrapper_exposes_native_model_summary(self):
        native_model = Mock()
        model = KerasDestinationModel(native_model, Mock())

        model.summary(expand_nested=True, show_trainable=True)

        native_model.summary.assert_called_once_with(
            expand_nested=True,
            show_trainable=True,
        )

    def test_keras_wrapper_plots_native_model_for_notebook(self):
        native_model = Mock()
        model = KerasDestinationModel(native_model, Mock())
        image = object()
        tensorflow = Mock()
        tensorflow.keras.utils.plot_model.return_value = image

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vehicle_destination.models.keras_model._require_plot_dependencies"
            ),
            patch(
                "vehicle_destination.models.keras_model._tensorflow",
                return_value=tensorflow,
            ),
        ):
            output = Path(directory) / "plots" / "architecture.png"
            result = model.plot(output, show_dtype=True, rankdir="LR")

        self.assertIs(result, image)
        tensorflow.keras.utils.plot_model.assert_called_once_with(
            native_model,
            to_file=str(output.resolve()),
            show_shapes=True,
            show_dtype=True,
            show_layer_names=True,
            rankdir="LR",
            expand_nested=True,
            dpi=160,
            show_layer_activations=False,
            show_trainable=True,
            splines="ortho",
        )

    def test_model_save_load_and_prediction_parity(self):
        samples, _ = prepare_samples(
            self.trips,
            self.config.dataset,
            manifest=self.model.split_manifest,
            prefix_fractions=(0.25,),
        )
        selected = samples.iloc[[0]]
        expected = self.model.predict(selected)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.model.save(directory)
            loaded = load_baseline(paths["model"])
            actual = loaded.predict(selected)
        self.assertAlmostEqual(
            expected.iloc[0]["predicted_latitude"],
            actual.iloc[0]["predicted_latitude"],
            places=9,
        )

    def test_map_payload_contains_all_scientific_layers(self):
        samples, _ = prepare_samples(
            self.trips,
            self.config.dataset,
            manifest=self.model.split_manifest,
            prefix_fractions=(0.25,),
        )
        prediction = self.model.predict(samples.iloc[[0]])
        trip = self.trips.loc[self.trips["trip_id"].eq(samples.iloc[0]["trip_id"])].iloc[0]
        payload = build_map_payload(trip, prediction.iloc[0], self.model.grid)
        self.assertEqual({point["kind"] for point in payload["points"]}, {"origin", "actual", "refined"})
        self.assertGreaterEqual(len(payload["candidates"]), 1)
        self.assertGreaterEqual(len(payload["trajectory"]), 2)
        self.assertTrue(all(candidate["polygon"] for candidate in payload["candidates"]))

    def test_leaflet_map_is_self_contained_and_layer_aware(self):
        samples, _ = prepare_samples(
            self.trips,
            self.config.dataset,
            manifest=self.model.split_manifest,
            prefix_fractions=(0.25,),
        )
        prediction = self.model.predict(samples.iloc[[0]])
        trip = self.trips.loc[self.trips["trip_id"].eq(samples.iloc[0]["trip_id"])].iloc[0]
        payload = build_map_payload(trip, prediction.iloc[0], self.model.grid)
        page = build_leaflet_html(
            payload,
            {"Full trajectory", "Observed prefix", "Destinations", "Candidates"},
        )
        self.assertIn("leaflet", page.lower())
        self.assertIn("Observed trajectory prefix", page)
        self.assertIn(str(payload["candidates"][0]["cell"]), page)
        self.assertNotIn("unpkg.com/leaflet", page)

    def test_undefined_cell_metrics_remain_none(self):
        test = self.predictions.loc[self.predictions["split"].eq("test")]
        metrics = prediction_metrics(test)
        if metrics["cell_metrics_defined_samples"] == 0:
            self.assertIsNone(metrics["top_1_cell_accuracy"])

    def test_tensorflow_missing_has_actionable_message(self):
        try:
            _tensorflow()
        except RuntimeError as exc:
            self.assertIn("pip install", str(exc))
        else:
            self.skipTest("TensorFlow is installed; live model tests belong to the optional suite")


if __name__ == "__main__":
    unittest.main()
