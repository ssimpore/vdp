"""One-command sample run without relying on shell-specific syntax."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicle_destination.config import load_config  # noqa: E402
from vehicle_destination.pipeline import run_all  # noqa: E402


if __name__ == "__main__":
    result = run_all(load_config(ROOT / "configs" / "default.yaml"))
    print("Pipeline completed")
    for name, path in result.outputs.items():
        print(f"{name}: {path}")
