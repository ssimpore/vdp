import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from vehicle_destination.config import load_config
from vehicle_destination.dataset import make_vin_split
from vehicle_destination.ui.components import (
    artifact_paths,
    feature_audit_table,
    parse_prefix_points_text,
    prediction_candidates,
    prefix_label,
    stage_status_rows,
)
from vehicle_destination.ui.maps import (
    build_trip_overview_html,
    build_trip_overview_payload,
)
from vehicle_destination.ui.app import _read_csv_file


ROOT = Path(__file__).resolve().parents[1]


class PresentationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config(ROOT / "configs" / "default.yaml")

    def test_artifact_paths_are_project_relative(self):
        paths = artifact_paths(self.config)
        self.assertEqual(paths["work"], ROOT / "artifacts" / "demo")
        self.assertEqual(paths["trips"].name, "trips.csv")

    def test_prefix_labels_include_minutes_and_percentage(self):
        self.assertEqual(prefix_label(0.0, 62), "0 min (departure)")
        self.assertEqual(prefix_label(0.25, 60), "15 min (25%)")

    def test_candidate_table_is_ranked_and_export_ready(self):
        prediction = {
            "top_k_candidates": json.dumps(
                [
                    {"rank": 1, "cell": "g1", "probability": 0.75, "latitude": 48.8, "longitude": 2.2},
                    {"rank": 2, "cell": "g2", "probability": 0.25, "latitude": 48.9, "longitude": 2.3},
                ]
            )
        }
        payload = {
            "candidates": [
                {"rank": 1, "error_to_actual_km": 1.234},
                {"rank": 2, "error_to_actual_km": 2.5},
            ]
        }
        table = prediction_candidates(prediction, payload)
        self.assertEqual(table.columns.tolist(), ["Rank", "Probability", "Distance (km)", "Cell"])
        self.assertEqual(table.iloc[0].to_dict(), {"Rank": 1, "Probability": "75.0%", "Distance (km)": "1.23", "Cell": "g1"})

    def test_stage_status_contract_has_every_pipeline_stage(self):
        rows = stage_status_rows(self.config)
        self.assertEqual([row["stage"] for row in rows], ["Raw telemetry", "Cleaned trips", "Baseline model", "Keras model"])

    def test_custom_prefix_parser_validates_coordinates(self):
        points = parse_prefix_points_text("[[48.8, 2.2], [48.9, 2.3]]")
        self.assertEqual(points[0], (48.8, 2.2))
        with self.assertRaisesRegex(ValueError, "outside valid coordinate"):
            parse_prefix_points_text("[[148.8, 2.2]]")

    def test_feature_audit_exposes_all_model_inputs_and_sources(self):
        from vehicle_destination.dataset import FEATURE_COLUMNS

        sample = {name: float(index) for index, name in enumerate(FEATURE_COLUMNS)}
        provenance = {
            "input_sources": {
                "departure_time": "reference_trip",
                "origin": "user_override",
                "trajectory_prefix": "reference_trip",
                "history": "automatic_previous_trips",
            }
        }
        audit = feature_audit_table(sample, provenance)
        self.assertEqual(len(audit), len(FEATURE_COLUMNS))
        self.assertEqual(
            audit.loc[audit["Feature"].eq("origin_latitude"), "Source"].iloc[0],
            "user_override",
        )

    def test_overview_map_contains_every_model_trip_and_split(self):
        trips = pd.read_csv(ROOT / "artifacts" / "demo" / "trips" / "trips.csv")
        manifest = make_vin_split(trips["VIN"], self.config.dataset)
        payload = build_trip_overview_payload(trips, manifest)

        self.assertEqual(payload["trip_count"], len(trips))
        self.assertEqual(len(payload["routes"]), len(trips))
        self.assertEqual(
            sum(item["trip_count"] for item in payload["splits"].values()),
            len(trips),
        )
        self.assertEqual(
            {route["split"] for route in payload["routes"]},
            {"train", "validation", "test"},
        )
        self.assertTrue(all(len(route["path"]) >= 2 for route in payload["routes"]))

        page = build_trip_overview_html(payload)
        self.assertIn("All accepted model trips by dataset split", page)
        self.assertIn("Model trip coverage", page)
        self.assertIn("Final test", page)
        self.assertNotIn("unpkg.com/leaflet", page)

    def test_csv_reader_can_limit_rows_and_columns(self):
        frame = pd.DataFrame(
            {
                "keep": range(1_000),
                "discard": ["large-value"] * 1_000,
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "large.csv"
            frame.to_csv(path, index=False)
            sample = _read_csv_file(
                str(path),
                fingerprint=0,
                columns=("keep",),
                row_limit=25,
            )

        self.assertEqual(sample.columns.tolist(), ["keep"])
        self.assertEqual(len(sample), 25)
        self.assertEqual(sample["keep"].iloc[-1], 24)


@unittest.skipUnless(importlib.util.find_spec("streamlit"), "Streamlit app extra is not installed")
class StreamlitSmokeTests(unittest.TestCase):
    def test_data_tables_render_bounded_samples(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
        next(button for button in app.button if "Data & Trips" in button.label).click().run()

        self.assertFalse(list(app.exception))
        self.assertEqual(
            [control.label for control in app.selectbox],
            ["Raw rows shown", "Trip rows shown"],
        )
        self.assertEqual(len(app.dataframe[0].value), 25)
        self.assertEqual(len(app.dataframe[1].value), 25)

        app.selectbox[0].set_value(10).run()
        app.selectbox[1].set_value(50).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.dataframe[0].value), 10)
        self.assertEqual(len(app.dataframe[1].value), 50)

    def test_evaluation_audit_renders_only_selected_sample(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
        next(button for button in app.button if "Evaluation" in button.label).click().run()

        self.assertFalse(list(app.exception))
        audit_control = next(
            control for control in app.selectbox if control.label == "Audit rows shown"
        )
        self.assertEqual(len(app.dataframe[0].value), 25)
        audit_control.set_value(10).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.dataframe[0].value), 10)

    def test_overview_renders_complete_trip_map(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
        next(button for button in app.button if "Overview" in button.label).click().run()

        self.assertFalse(list(app.exception))
        self.assertEqual(len(app.get("iframe")), 1)
        self.assertTrue(
            any(
                "70 accepted trips across 14 vehicles" in caption.value
                for caption in app.caption
            )
        )

    def test_default_prediction_workspace_renders_without_exception(self):
        from streamlit.testing.v1 import AppTest

        app = AppTest.from_file(ROOT / "streamlit_app.py", default_timeout=60).run()
        self.assertFalse(list(app.exception))
        self.assertEqual(app.selectbox[0].label, "Model")
        self.assertEqual(app.selectbox[1].label, "VIN")
        self.assertEqual(app.selectbox[2].label, "Reference trip")
        self.assertTrue(any("Run prediction" in button.label for button in app.button))
        self.assertGreaterEqual(len(app.dataframe), 2)
        app.number_input[0].set_value(48.981).run()
        app.button[0].click().run()
        self.assertFalse(list(app.exception))
        feature_audit = app.dataframe[1].value
        source = feature_audit.loc[
            feature_audit["Feature"].eq("origin_latitude"), "Source"
        ].iloc[0]
        self.assertEqual(source, "user_override")


if __name__ == "__main__":
    unittest.main()
