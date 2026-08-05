from __future__ import annotations

import json
import zipfile
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "docker" / "espocrm-extension"
DIST_ROOT = SOURCE_ROOT / "dist"
PACKAGE_PATH = DIST_ROOT / "mva-insurance-data-model-1.0.0.zip"


def build_package() -> Path:
    manifest_path = SOURCE_ROOT / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    required_manifest_keys = {"name", "version", "acceptableVersions", "description"}
    missing = required_manifest_keys.difference(manifest)
    if missing:
        raise RuntimeError(f"Extension manifest is missing: {sorted(missing)}")

    files_root = SOURCE_ROOT / "files"
    if not files_root.is_dir():
        raise RuntimeError(f"Extension files directory not found: {files_root}")

    DIST_ROOT.mkdir(parents=True, exist_ok=True)
    temporary_path = PACKAGE_PATH.with_suffix(".tmp")

    with zipfile.ZipFile(
        temporary_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for path in sorted(SOURCE_ROOT.rglob("*")):
            if not path.is_file() or DIST_ROOT in path.parents:
                continue
            archive.write(path, path.relative_to(SOURCE_ROOT).as_posix())

    temporary_path.replace(PACKAGE_PATH)
    return PACKAGE_PATH


def main() -> int:
    package_path = build_package()
    print(package_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
