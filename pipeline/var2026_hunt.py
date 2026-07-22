"""VAR 2026 Hunt — Competition Hunter Strategy for Digital Twin BTS.

Follows competition_hunter pattern:
  hunt → bootstrap → plan → dispatch → poll → download → submit

Key difference: this is NOT a Kaggle competition, so instead of:
  - resolve_competition(url) → we load local VAR 2026 data
  - build_tabular_baseline_notebook() → we build 3DGS kernel scripts
  - competition_sources=[slug] → we use dataset_sources for scene data

Deep integration with competition_hunter:
  - KaggleExecutionClient.from_env() for auth + kernel operations
  - dispatch_planned_kernels() for push phase
  - Same execution_state.json format for interop
  - Same poll→download→package pipeline pattern
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

# Add Viettel_Race_AI/De_1 for pipeline imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Add project root for competition_hunter imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from competition_hunter.execution.kaggle_client import (
    KaggleExecutionClient,
    KernelStatus,
    inspect_execution_environment,
    kaggle_cli_available,
)
from competition_hunter.execution.kaggle_auth import load_kaggle_auth
from competition_hunter.execution.dispatcher import dispatch_planned_kernels
from competition_hunter.models import now_iso

from pipeline.config import (
    DATA_DIR,
    GS_DIR,
    KAGGLE_USERNAME,
    KAGGLE_DATASET_PREFIX,
    SCENES,
    VARIANTS,
    OUTPUT_DIR,
    KERNEL_TIMEOUT_SECONDS,
)


# ═══════════════════════════════════════════════════════════════
#  Phase 1: BOOTSTRAP — validate + environment check
# ═══════════════════════════════════════════════════════════════

def _json_write(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _json_read(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def bootstrap_var2026(workdir: Path | None = None) -> dict:
    """Initialize VAR 2026 workspace (competition_hunter bootstrap pattern).

    Returns workspace paths dict (like start_hunt returns created).
    """
    wd = (workdir or Path.cwd()).resolve()
    ws = wd / ".var2026"
    ws.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("VAR 2026 HUNT — Bootstrap")
    print("=" * 60)

    # 1. Check Kaggle environment
    env = inspect_execution_environment("")
    print(f"  Kaggle auth: {'[OK]' if env.available else '[FAIL]'} ({env.auth_method})")
    print(f"  Kaggle CLI:  {'[OK]' if env.cli_available else '[FAIL]'}")
    print(f"  Username:    {env.username or 'unknown'}")

    if not env.available:
        print("\n  [FAIL] Kaggle not configured. Set up ~/.kaggle/kaggle.json first.")
        return {"workspace_root": str(ws), "ready": False, "error": "kaggle_auth_missing"}

    # 2. Validate all scenes
    scene_reports = {}
    all_valid = True
    for scene in SCENES:
        sd = DATA_DIR / scene
        checks = {
            "images": (sd / "train" / "images").exists() or (sd / "images").exists(),
            "sparse_cameras": (sd / "train" / "sparse" / "0" / "cameras.bin").exists()
                or (sd / "sparse" / "0" / "cameras.bin").exists(),
            "test_poses": (sd / "test" / "test_poses.csv").exists()
                or (sd / "test_poses.csv").exists(),
        }
        scene_valid = all(checks.values())
        all_valid = all_valid and scene_valid
        scene_reports[scene] = checks
        status = "[OK]" if scene_valid else "[FAIL]"
        missing = [k for k, v in checks.items() if not v]
        detail = f"  (missing: {', '.join(missing)})" if missing else ""
        print(f"  {status} {scene}{detail}")

    if not all_valid:
        print("\n  [FAIL] Some scenes have missing data. Fix before continuing.")
        return {"workspace_root": str(ws), "ready": False, "error": "invalid_scenes"}

    # 3. Detect dataset slugs (already uploaded or need upload)
    dataset_slugs = {}
    for scene in SCENES:
        dataset_slugs[scene] = f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}"

    # 4. Write bootstrap state
    bootstrap_state = {
        "competition": "var2026-digital-twin-bts",
        "started_at": now_iso(),
        "kaggle_user": env.username,
        "scenes": SCENES,
        "scene_reports": scene_reports,
        "dataset_slugs": dataset_slugs,
        "variants": [v.name for v in VARIANTS],
        "num_variants": len(VARIANTS),
        "stages": {
            "bootstrap": "ready",
            "datasets": "pending",
            "kernels": "pending",
            "training": "pending",
            "packaging": "pending",
        },
    }
    _json_write(ws / "bootstrap_state.json", bootstrap_state)

    print(f"\n  Bootstrap complete. Workspace: {ws}")
    print(f"  Scenes:  {len(SCENES)} valid")
    print(f"  Variants: {len(VARIANTS)} per scene ({len(VARIANTS) * len(SCENES)} total runs)")

    return {
        "workspace_root": str(ws),
        "ready": True,
        "scenes": SCENES,
        "dataset_slugs": dataset_slugs,
        "kaggle_user": env.username,
    }


# ═══════════════════════════════════════════════════════════════
#  Phase 2: PLAN — build kernel bundles using KaggleExecutionClient
# ═══════════════════════════════════════════════════════════════

def plan_kernels(
    workspace_root: Path,
    dataset_slugs: dict[str, str] | None = None,
) -> dict:
    """Build kernel bundles for all scenes using KaggleExecutionClient.

    Creates execution_state.json compatible with dispatch_planned_kernels().
    Each scene gets ONE kernel that trains all variants sequentially.

    Args:
        workspace_root: .var2026 workspace path
        dataset_slugs: Optional dict mapping scene_name -> kaggle_dataset_slug.
                       If None, uses default pattern: {user}/var2026-{scene}
    """
    client = KaggleExecutionClient.from_env(enable_gpu=True, enable_internet=True)
    ws = Path(workspace_root)
    bundles_dir = ws / "bundles"
    kernel_src = Path(__file__).resolve().parent / "kernel_3dgs_train.py"
    gs_code = GS_DIR

    # Resolve dataset slugs
    if dataset_slugs is None:
        dataset_slugs = {s: f"{KAGGLE_DATASET_PREFIX}-{s.lower()}" for s in SCENES}

    print("\n" + "=" * 60)
    print("VAR 2026 HUNT — Plan Kernels")
    print("=" * 60)
    print(f"  Dataset mapping:")
    for scene, slug in dataset_slugs.items():
        print(f"    {scene} -> {slug}")

    planned_kernels = []
    for scene in SCENES:
        dataset_slug = dataset_slugs.get(scene, f"{KAGGLE_DATASET_PREFIX}-{scene.lower()}")
        experiment_name = f"var2026-{scene.lower()}"
        bundle_root = bundles_dir / scene

        print(f"\n-- {scene} --")
        print(f"  Dataset:  {dataset_slug}")
        print(f"  Bundle:   {bundle_root}")

        # Use KaggleExecutionClient's native methods for kernel ops
        kernel_slug = client.build_kernel_slug(experiment_name, prefix="var2026")

        # Build bundle directory
        bundle_root.mkdir(parents=True, exist_ok=True)

        # Copy kernel script
        import shutil
        shutil.copy2(str(kernel_src), str(bundle_root / "kernel_3dgs_train.py"))

        # Write kernel-metadata.json using client's own method
        client.write_kernel_bundle(
            bundle_root,
            experiment_name=experiment_name,
            code_file="kernel_3dgs_train.py",
            competition_sources=[],
            dataset_sources=[dataset_slug],
        )

        # Override kernel_type to script (client defaults to notebook)
        meta_path = bundle_root / "kernel-metadata.json"
        metadata = json.loads(meta_path.read_text())
        metadata["kernel_type"] = "script"
        meta_path.write_text(json.dumps(metadata, indent=2))

        # Copy 3DGS code (CUDA extensions needed at build time)
        gs_dst = bundle_root / "gaussian-splatting"
        if not gs_dst.exists():
            _copy_gs_essentials(gs_code, gs_dst)

        # Create kernel plan record for execution_state.json
        kernel_plan = {
            "kernel_slug": kernel_slug,
            "kernel_url": f"https://www.kaggle.com/code/{kernel_slug}",
            "code_file": "kernel_3dgs_train.py",
            "dataset_sources": [dataset_slug],
            "competition_sources": [],
            "enable_gpu": True,
            "enable_internet": True,
            "bundle_root": str(bundle_root),
            "scene": scene,
        }
        planned_kernels.append(kernel_plan)
        print(f"  Kernel:   {kernel_slug}")
        print(f"  Files:    kernel_3dgs_train.py + kernel-metadata.json + gaussian-splatting/")

    # Warn if datasets may not exist on Kaggle
    print(f"\n  [WARN] Ensure these datasets exist on Kaggle before dispatching:")
    for scene in SCENES:
        print(f"         {KAGGLE_DATASET_PREFIX}-{scene.lower()}")
    print(f"  [TIP]  Upload with: python -m pipeline.kaggle_uploader")

    # Write execution_state.json (competition_hunter compatible format)
    execution_state = {
        "available": True,
        "auth_method": "legacy_api_key",
        "auth_source": "~/.kaggle/kaggle.json",
        "username": client.username,
        "cli_available": kaggle_cli_available(),
        "sdk_available": True,
        "competition_access": "n/a",
        "generated_at": now_iso(),
        "planned_kernels": planned_kernels,
        "dispatch_history": [],
    }
    _json_write(ws / "execution_state.json", execution_state)

    # Also write hunt_state.json
    hunt_state = _json_read(ws / "bootstrap_state.json")
    hunt_state["stages"]["kernels"] = "ready"
    hunt_state["num_kernels"] = len(planned_kernels)
    hunt_state["kernel_slugs"] = [k["kernel_slug"] for k in planned_kernels]
    _json_write(ws / "bootstrap_state.json", hunt_state)

    print(f"\n  [OK] {len(planned_kernels)} kernels planned")
    print(f"  Workspace: {ws}")
    return execution_state


def _copy_gs_essentials(src: Path, dst: Path):
    """Copy only essential 3DGS files for kernel build (skip heavy deps)."""
    import shutil
    import fnmatch

    ignore = [
        ".git", "SIBR_viewers", "output",
        "__pycache__", "*.pyc", "assets", "*.ipynb",
    ]

    def _ignore_fn(path, names):
        ignored = set()
        for name in names:
            for pat in ignore:
                if pat.startswith("*"):
                    if fnmatch.fnmatch(name, pat):
                        ignored.add(name)
                elif name == pat:
                    ignored.add(name)
        return ignored

    shutil.copytree(str(src), str(dst), ignore=_ignore_fn, dirs_exist_ok=True)


# ═══════════════════════════════════════════════════════════════
#  Phase 3: DISPATCH — push kernels using competition_hunter dispatcher
# ═══════════════════════════════════════════════════════════════

def dispatch_kernels(workspace_root: Path, dry_run: bool = False) -> dict:
    """Push all planned kernels to Kaggle GPU.

    Uses competition_hunter's dispatch_planned_kernels() for consistency.
    """
    ws = Path(workspace_root)
    print("\n" + "=" * 60)
    print("VAR 2026 HUNT — Dispatch Kernels")
    print("=" * 60)

    if dry_run:
        print("  MODE: DRY RUN (no actual push)")

    result = dispatch_planned_kernels(ws, dry_run=dry_run)

    for r in result.get("results", []):
        status = r["status"]
        icon = "[OK]" if status in ("submitted", "dry_run") else "[FAIL]"
        kernel = r.get("kernel_slug", "?")
        print(f"  {icon} {kernel}: {status}")

    if not dry_run:
        # Update hunt state
        hunt_state = _json_read(ws / "bootstrap_state.json")
        hunt_state["stages"]["training"] = "running"
        hunt_state["dispatch_result"] = {
            "count": result["count"],
            "timestamp": now_iso(),
        }
        _json_write(ws / "bootstrap_state.json", hunt_state)

    return result


# ═══════════════════════════════════════════════════════════════
#  Phase 4: POLL — monitor all kernels
# ═══════════════════════════════════════════════════════════════

def poll_all_kernels(
    workspace_root: Path,
    timeout: int = KERNEL_TIMEOUT_SECONDS,
) -> dict[str, dict]:
    """Poll all kernel statuses until complete/error/timeout.

    Uses KaggleExecutionClient.poll_kernel_status() for consistency
    with competition_hunter.
    """
    client = KaggleExecutionClient.from_env()
    ws = Path(workspace_root)
    exec_state = _json_read(ws / "execution_state.json")
    kernels = exec_state.get("planned_kernels", [])

    print("\n" + "=" * 60)
    print(f"VAR 2026 HUNT — Poll {len(kernels)} Kernels")
    print("=" * 60)

    results = {}
    t0 = time.time()
    active = {k["kernel_slug"]: k for k in kernels}

    while active and (time.time() - t0 < timeout):
        for slug in list(active.keys()):
            status = client.poll_kernel_status(slug)

            if status == KernelStatus.COMPLETE:
                elapsed = time.time() - t0
                print(f"  [OK] {slug}: COMPLETE ({elapsed/60:.0f}m)")
                results[slug] = {"status": "complete", "elapsed": elapsed}
                del active[slug]

            elif status in (KernelStatus.ERROR, KernelStatus.CANCELLED):
                print(f"  [FAIL] {slug}: {status.value}")
                results[slug] = {"status": status.value}
                del active[slug]

            elif status == KernelStatus.RUNNING:
                pass  # Still running, check next loop

            elif status == KernelStatus.QUEUED:
                pass  # Still queued

        if active:
            remaining = len(active)
            elapsed = time.time() - t0
            print(f"  ... {remaining} kernels still running ({elapsed/60:.0f}m elapsed)")
            time.sleep(60)  # Check every minute

    # Handle timeouts
    for slug in active:
        print(f"  [TIME] {slug}: TIMEOUT")
        results[slug] = {"status": "timeout"}

    # Update hunt state
    hunt_state = _json_read(ws / "bootstrap_state.json")
    hunt_state["stages"]["training"] = "complete"
    hunt_state["poll_results"] = results
    _json_write(ws / "bootstrap_state.json", hunt_state)

    print(f"\n  Summary: {sum(1 for r in results.values() if r['status'] == 'complete')}/{len(kernels)} complete")
    return results


# ═══════════════════════════════════════════════════════════════
#  Phase 5: DOWNLOAD — pull kernel outputs
# ═══════════════════════════════════════════════════════════════

def download_all_outputs(workspace_root: Path) -> dict[str, bool]:
    """Download outputs from all completed kernels."""
    import subprocess

    ws = Path(workspace_root)
    exec_state = _json_read(ws / "execution_state.json")
    kernels = exec_state.get("planned_kernels", [])
    hunt_state = _json_read(ws / "bootstrap_state.json")
    poll_results = hunt_state.get("poll_results", {})

    print("\n" + "=" * 60)
    print("VAR 2026 HUNT — Download Outputs")
    print("=" * 60)

    results = {}
    for kernel in kernels:
        slug = kernel["kernel_slug"]
        scene = kernel.get("scene", slug.split("-")[-1])

        if poll_results.get(slug, {}).get("status") != "complete":
            print(f"  [SKIP] {slug}: skipped (not complete)")
            results[slug] = False
            continue

        output_dir = OUTPUT_DIR / "kernel_outputs" / scene
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"  [DL] {slug} -> {output_dir}")
        r = subprocess.run(
            ["kaggle", "kernels", "output", slug, "-p", str(output_dir)],
            capture_output=True, text=True, timeout=300,
        )
        success = r.returncode == 0
        results[slug] = success
        print(f"     {'[OK]' if success else '[FAIL]'}")

    # Update hunt state
    hunt_state["stages"]["packaging"] = "ready" if any(results.values()) else "pending"
    hunt_state["download_results"] = results
    _json_write(ws / "bootstrap_state.json", hunt_state)

    return results


# ═══════════════════════════════════════════════════════════════
#  FULL HUNT — one-call orchestration
# ═══════════════════════════════════════════════════════════════

def start_var2026_hunt(
    workdir: Path | None = None,
    *,
    dry_run: bool = False,
    skip_bootstrap: bool = False,
    skip_dispatch: bool = False,
    skip_poll: bool = False,
    skip_download: bool = False,
) -> dict:
    """Run complete VAR 2026 hunt (competition_hunter strategy).

    Phases:
      1. BOOTSTRAP  — validate data + check Kaggle environment
      2. PLAN       — build kernel bundles + execution_state.json
      3. DISPATCH   — push all kernels to Kaggle GPU
      4. POLL       — monitor until complete/timeout
      5. DOWNLOAD   — pull all outputs locally
      6. PACKAGE    — (use package_submission.py separately)

    Args:
        workdir: working directory (default: current)
        dry_run: build bundles but don't push
        skip_*: skip individual phases
    """
    wd = (workdir or Path.cwd()).resolve()
    ws = wd / ".var2026"
    report: dict[str, Any] = {"started": now_iso(), "phases": {}}

    # Phase 1: Bootstrap
    if not skip_bootstrap:
        boot = bootstrap_var2026(wd)
        report["phases"]["bootstrap"] = boot
        if not boot.get("ready"):
            report["success"] = False
            report["error"] = boot.get("error")
            return report
    else:
        print("[SKIP] Bootstrap")

    # Phase 2: Plan
    plan = plan_kernels(ws)
    report["phases"]["plan"] = {"num_kernels": len(plan.get("planned_kernels", []))}

    # Phase 3: Dispatch
    if not skip_dispatch:
        dispatch = dispatch_kernels(ws, dry_run=dry_run)
        report["phases"]["dispatch"] = {
            "count": dispatch["count"],
            "dry_run": dry_run,
        }
    else:
        print("[SKIP] Dispatch")

    if dry_run:
        report["success"] = True
        report["note"] = "Dry run complete. Run without --dry-run to execute."
        return report

    # Phase 4: Poll
    if not skip_poll:
        poll = poll_all_kernels(ws)
        report["phases"]["poll"] = {
            "total": len(poll),
            "complete": sum(1 for r in poll.values() if r["status"] == "complete"),
            "failed": sum(1 for r in poll.values() if r["status"] not in ("complete",)),
        }
    else:
        print("[SKIP] Poll")

    # Phase 5: Download
    if not skip_download:
        downloads = download_all_outputs(ws)
        report["phases"]["download"] = {
            "total": len(downloads),
            "success": sum(1 for v in downloads.values() if v),
        }
    else:
        print("[SKIP] Download")

    report["success"] = True
    report["workspace"] = str(ws)

    print(f"\n{'='*60}")
    print(">>> VAR 2026 HUNT COMPLETE <<<")
    print(f"   Workspace: {ws}")
    print(f"   Next: python -m pipeline.package_submission")
    print(f"{'='*60}")

    return report


# ═══════════════════════════════════════════════════════════════
#  CLI entry point
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description="VAR 2026 Hunt — competition_hunter strategy for Digital Twin BTS",
    )
    parser.add_argument("--workdir", default=".", help="Working directory")
    parser.add_argument("--bootstrap-only", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-dispatch", action="store_true")
    parser.add_argument("--skip-poll", action="store_true")
    parser.add_argument("--skip-download", action="store_true")
    parser.add_argument(
        "--dataset-slugs", type=str, default=None,
        help="JSON mapping of scene->dataset_slug, e.g. '{\"HCM0421\":\"user/var2026-hcm0421\",...}' "
             "OR comma-separated slugs matching scene order. "
             "Default: {user}/var2026-{scene}"
    )
    args = parser.parse_args()

    wd = Path(args.workdir).resolve()
    ws = wd / ".var2026"

    # Parse dataset slugs if provided
    dataset_slugs = None
    if args.dataset_slugs:
        raw = args.dataset_slugs.strip()
        if raw.startswith("{"):
            dataset_slugs = json.loads(raw)
        else:
            # Comma-separated list, maps to SCENES in order
            slugs = [s.strip() for s in raw.split(",")]
            dataset_slugs = dict(zip(SCENES, slugs))

    if args.bootstrap_only:
        bootstrap_var2026(wd)
    elif args.plan_only:
        if not ws.exists():
            bootstrap_var2026(wd)
        plan_kernels(ws, dataset_slugs=dataset_slugs)
    else:
        start_var2026_hunt(
            workdir=wd,
            dry_run=args.dry_run,
            skip_dispatch=args.skip_dispatch,
            skip_poll=args.skip_poll,
            skip_download=args.skip_download,
        )
