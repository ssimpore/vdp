import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from vehicle_destination.__main__ import main
from vehicle_destination.pipeline import StageResult


ROOT = Path(__file__).resolve().parents[1]


class CliVerboseTests(unittest.TestCase):
    def _run_trip_build(self, *, verbose: bool) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as directory:
            command = [
                sys.executable,
                "-m",
                "vehicle_destination",
                "--config",
                str(ROOT / "configs" / "default.yaml"),
                "build-trips",
                "--raw-data",
                str(ROOT / "data" / "sample" / "raw_data.csv"),
                "--output-dir",
                directory,
            ]
            if verbose:
                command.append("--verbose")
            return subprocess.run(
                command,
                cwd=ROOT,
                check=True,
                capture_output=True,
                text=True,
            )

    def test_trip_build_verbose_progress_stays_out_of_json_stdout(self):
        completed = self._run_trip_build(verbose=True)
        payload = json.loads(completed.stdout)

        self.assertEqual(payload["summary"]["accepted_trips"], 70)
        self.assertIn("[vehicle-destination] Starting trip build", completed.stderr)
        self.assertIn("Trip reconstruction progress", completed.stderr)
        self.assertRegex(
            completed.stderr,
            r"Trip reconstruction progress \[[#-]+\]\s+100%",
        )
        self.assertIn("Trip artifacts are ready", completed.stderr)

    def test_trip_build_is_quiet_without_verbose_flag(self):
        completed = self._run_trip_build(verbose=False)

        json.loads(completed.stdout)
        self.assertNotIn("[vehicle-destination]", completed.stderr)

    def test_train_verbose_flag_reaches_training_service(self):
        result = StageResult(outputs={"model": "model.joblib"}, summary={})
        stdout = io.StringIO()
        with (
            patch("vehicle_destination.__main__._configure_verbose_logging") as configure,
            patch("vehicle_destination.__main__.train_model", return_value=result) as train,
            contextlib.redirect_stdout(stdout),
        ):
            exit_code = main(
                [
                    "--config",
                    str(ROOT / "configs" / "default.yaml"),
                    "train",
                    "--engine",
                    "baseline",
                    "--verbose",
                ]
            )

        self.assertEqual(exit_code, 0)
        configure.assert_called_once_with()
        self.assertTrue(train.call_args.kwargs["verbose"])
        json.loads(stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
