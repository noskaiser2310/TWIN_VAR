#!/usr/bin/env python
"""VAR 2026 — Digital Twin BTS: One-click Auto Pipeline.

Usage:
    # Full pipeline (all scenes, all phases):
    python run_pipeline.py

    # Specific scenes only:
    python run_pipeline.py --scenes HCM0421 HCM0539

    # Dry run (build bundles, don't push):
    python run_pipeline.py --dry-run

    # Skip depth map generation (already done):
    python run_pipeline.py --skip-depth

    # Validate data only:
    python run_pipeline.py --validate-only

    # Package submission from existing outputs:
    python run_pipeline.py --package-only --variant full_combo

Pipeline Phases:
    1. VALIDATE   — Check all scenes have required data
    2. DEPTH      — Generate depth maps (Depth-Anything v2)
    3. UPLOAD     — Upload scenes as Kaggle private datasets
    4. BUNDLE     — Build kernel bundles (3DGS code + metadata)
    5. PUSH       — Push kernels to Kaggle GPU
    6. POLL       — Monitor kernel execution
    7. DOWNLOAD   — Download rendered outputs
    8. PACKAGE    — Create submission_round1.zip

Architecture:
    ┌──────────┐     ┌──────────────┐     ┌───────────────┐
    │  Local   │────▶│ Kaggle       │────▶│ Kaggle GPU    │
    │  Prep    │     │ Datasets     │     │ Kernels (x7)  │
    │  Depth   │     │ (private)    │     │ 3DGS ×6 vars  │
    └──────────┘     └──────────────┘     └───────────────┘
                                                │
    ┌──────────┐     ┌──────────────┐           │
    │ Submit   │◀────│ Package      │◀──────────┘
    │ ZIP      │     │ Selection    │    Download renders
    └──────────┘     └──────────────┘
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure pipeline modules are importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import SCENES, OUTPUT_DIR  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="VAR 2026 Auto Pipeline — Digital Twin BTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py                          # Full auto pipeline
  python run_pipeline.py --dry-run                # Test without pushing
  python run_pipeline.py --scenes HCM0421         # Single scene
  python run_pipeline.py --validate-only          # Check data only
  python run_pipeline.py --depth-only             # Generate depth maps only
  python run_pipeline.py --package-only           # Package existing outputs
  python run_pipeline.py --skip-depth --skip-upload  # Resume from bundles
        """,
    )

    # Scene selection
    parser.add_argument("--scenes", nargs="*", default=None,
                        help=f"Scenes to process (default: all {len(SCENES)} scenes)")

    # Phase toggles
    parser.add_argument("--validate-only", action="store_true",
                        help="Only validate data, then exit")
    parser.add_argument("--depth-only", action="store_true",
                        help="Only generate depth maps, then exit")
    parser.add_argument("--upload-only", action="store_true",
                        help="Only upload datasets, then exit")
    parser.add_argument("--package-only", action="store_true",
                        help="Only package submission from existing outputs")

    # Skip flags
    parser.add_argument("--skip-depth", action="store_true",
                        help="Skip depth map generation")
    parser.add_argument("--skip-upload", action="store_true",
                        help="Skip Kaggle dataset upload")
    parser.add_argument("--skip-push", action="store_true",
                        help="Skip kernel push (already pushed)")
    parser.add_argument("--skip-poll", action="store_true",
                        help="Skip polling (monitor manually)")

    # Options
    parser.add_argument("--variant", default="full_combo",
                        help="Primary variant for submission (default: full_combo)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build bundles but don't push to Kaggle")
    parser.add_argument("--output", default=None,
                        help="Submission ZIP filename")

    args = parser.parse_args()

    scenes = args.scenes if args.scenes else SCENES
    print(f"VAR 2026 AUTO PIPELINE")
    print(f"Scenes: {', '.join(scenes)}")
    print(f"Output: {OUTPUT_DIR}")
    print()

    # ── Phase: Validate only ──
    if args.validate_only:
        from pipeline.orchestrator import validate_all_scenes
        validate_all_scenes()
        return

    # ── Phase: Depth only ──
    if args.depth_only:
        from pipeline.prepare_depth import prepare_all_depths
        prepare_all_depths(scenes)
        return

    # ── Phase: Upload only ──
    if args.upload_only:
        from pipeline.kaggle_uploader import upload_all_scenes
        upload_all_scenes(scenes, dry_run=args.dry_run)
        return

    # ── Phase: Package only ──
    if args.package_only:
        from pipeline.package_submission import package_submission, validate_submission_zip
        zip_path = package_submission(
            scenes=scenes,
            variant=args.variant,
            output_name=args.output,
        )
        validate_submission_zip(zip_path)
        return

    # ══════════════════════════════════════════════════════════
    # FULL PIPELINE
    # ══════════════════════════════════════════════════════════

    # ── Phase 1: Validate data ──
    print("=" * 60)
    print("PHASE 1/5: DATA VALIDATION")
    print("=" * 60)
    from pipeline.orchestrator import validate_all_scenes
    reports = validate_all_scenes()
    invalid = [r["scene"] for r in reports if not r["valid"]]
    if invalid:
        print(f"\n❌ Cannot continue. Fix issues in: {invalid}")
        return

    # ── Phase 2: Depth maps ──
    if not args.skip_depth:
        print(f"\n{'='*60}")
        print("PHASE 2/5: DEPTH MAPS (Depth-Anything v2)")
        print("=" * 60)
        from pipeline.prepare_depth import prepare_all_depths
        prepare_all_depths(scenes)
    else:
        print("\n[SKIP] Depth map generation")

    # ── Phase 3: Upload datasets ──
    if not args.skip_upload:
        print(f"\n{'='*60}")
        print("PHASE 3/5: KAGGLE DATASET UPLOAD")
        print("=" * 60)
        from pipeline.kaggle_uploader import upload_all_scenes
        results = upload_all_scenes(scenes, dry_run=args.dry_run)
        failed = [s for s, r in results.items() if not r["success"]]
        if failed and not args.dry_run:
            print(f"\n⚠️  Failed uploads: {failed}")
            print("You can retry with: python run_pipeline.py --upload-only")
    else:
        print("\n[SKIP] Kaggle dataset upload")

    # ── Phase 4: Build + Push + Poll + Download ──
    print(f"\n{'='*60}")
    print("PHASE 4/5: KAGGLE GPU TRAINING")
    print("=" * 60)
    from pipeline.orchestrator import run_full_pipeline
    report = run_full_pipeline(
        scenes=scenes,
        skip_push=args.skip_push or args.dry_run,
        skip_poll=args.skip_poll or args.dry_run,
        dry_run=args.dry_run,
    )

    if args.dry_run:
        print("\n[DRY RUN] Bundles ready. Run without --dry-run to execute.")
        return

    # ── Phase 5: Package submission ──
    print(f"\n{'='*60}")
    print("PHASE 5/5: SUBMISSION PACKAGING")
    print("=" * 60)
    from pipeline.package_submission import package_submission, validate_submission_zip
    zip_path = package_submission(
        scenes=scenes,
        variant=args.variant,
        output_name=args.output,
    )
    validate_submission_zip(zip_path)

    print(f"\n{'='*60}")
    print("🏆 PIPELINE COMPLETE!")
    print(f"📦 Submission: {zip_path}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
