"""Package final images into competition submission ZIP.

Usage:
    python package.py                          # all scenes
    python package.py --scenes bonsai chair     # specific scenes
    python package.py --source final            # use OUTPUT_DIR/final/
    python package.py --source ensemble         # use OUTPUT_DIR/ensemble/
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
import config as _cfg
from config import OUTPUT_DIR, SUBMISSION_DIR, SUBMISSION_NAME, SCENES, set_data_dir


def package(scenes: list[str] | None = None, source: str = "final",
            output_name: str | None = None) -> Path:
    """Create submission.zip from source directory."""
    if scenes is None:
        scenes = SCENES
    if output_name is None:
        output_name = SUBMISSION_NAME

    SUBMISSION_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = SUBMISSION_DIR / output_name
    report = {"scenes": {}, "total": 0, "missing": []}

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene in scenes:
            src_dir = OUTPUT_DIR / source / scene
            if not src_dir.exists() or not any(src_dir.iterdir()):
                found = False
                for alt_src in ["final", "ensemble"]:
                    alt = OUTPUT_DIR / alt_src / scene
                    if alt.exists() and any(alt.iterdir()):
                        src_dir = alt
                        found = True
                        break
                if not found:
                    # Fallback: use first variant dir under renders/
                    renders_scene = OUTPUT_DIR / "renders" / scene
                    if renders_scene.exists():
                        for d in sorted(renders_scene.iterdir()):
                            if d.is_dir() and any(d.iterdir()):
                                src_dir = d
                                found = True
                                break
                if not found:
                    print(f"  [SKIP] {scene}: no images at {src_dir}")
                    report["scenes"][scene] = {"expected": 0, "added": 0, "missing": 0}
                    continue

            test_csv = _cfg.DATA_DIR / scene / "test" / "test_poses.csv"
            if not test_csv.exists():
                test_csv = _cfg.DATA_DIR / scene / "test_poses.csv"
            with open(test_csv) as f:
                expected = [r["image_name"] for r in csv.DictReader(f)]

            added = 0
            for name in expected:
                p = src_dir / name
                if p.exists():
                    zf.write(str(p), f"{scene}/{name}")
                    added += 1

            missing = len(expected) - added
            report["scenes"][scene] = {"expected": len(expected), "added": added, "missing": missing}
            report["total"] += added
            if missing:
                report["missing"].append(scene)
            print(f"  {'✅' if missing == 0 else '⚠️'} {scene}: {added}/{len(expected)}")

    # Summary
    print(f"\n{'='*60}")
    print(f"PACKAGE: {zip_path.name}")
    print(f"{'='*60}")
    print(f"  Total: {report['total']} images")
    print(f"  Size:  {zip_path.stat().st_size / (1024*1024):.1f} MB")
    if report["missing"]:
        print(f"  ⚠️  Missing images in: {report['missing']}")
    else:
        print(f"  ✅ All images present — Ready to submit!")

    (SUBMISSION_DIR / f"{output_name}.report.json").write_text(
        json.dumps(report, indent=2, default=str)
    )
    return zip_path


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--source", default="final")
    p.add_argument("--output", default=None)
    p.add_argument("--data-dir", default=None, help="Override data directory")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    package(args.scenes, args.source, args.output)
