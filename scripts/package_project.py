"""Create a deterministic, clean source-and-demo ZIP for handoff."""

from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_PARTS = {
    ".git",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "vehicle_destination_lab.egg-info",
    "notebook_validation",
    "notebook_run",
}
EXCLUDED_SUFFIXES = {".pyc", ".pyo", ".tmp"}


def included_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*")
        if path.is_file()
        and not EXCLUDED_PARTS.intersection(path.relative_to(ROOT).parts)
        and path.suffix.lower() not in EXCLUDED_SUFFIXES
    )


def build_archive(output: Path) -> tuple[Path, int]:
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    files = included_files()
    with ZipFile(temporary, "w", ZIP_DEFLATED, compresslevel=9) as archive:
        for source in files:
            relative = Path(ROOT.name) / source.relative_to(ROOT)
            info = ZipInfo.from_file(source, arcname=relative.as_posix())
            # Stable timestamps make repeated packaging byte-reproducible when
            # project contents are unchanged.
            info.date_time = (2026, 1, 1, 0, 0, 0)
            info.compress_type = ZIP_DEFLATED
            with source.open("rb") as handle:
                archive.writestr(info, handle.read(), compresslevel=9)
    temporary.replace(output)
    with ZipFile(output) as archive:
        bad = archive.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity check failed at {bad}")
    return output, len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", default="dist/vehicle_destination_lab_v1_6_ready.zip"
    )
    arguments = parser.parse_args()
    output, count = build_archive(Path(arguments.output))
    print(f"Created {output} with {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
