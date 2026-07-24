"""Train a single 3DGS variant locally. Wraps gaussian-splatting/train.py.

Fully leverages ALL baseline features:
  - eval mode, sh_degree, depth scheduling, densification tuning
  - exposure compensation, anti-aliasing, sparse Adam, white/random BG
  - checkpoint resume (big variant from full_60k)
  - fused-ssim (auto-detected), list-based subprocess (no shell injection)
  - --quiet flag for clean output

Usage:
    python train.py --scene bonsai --variant full_60k
    python train.py --scene bonsai --variant fast --iters 7000
    python train.py --scene bonsai --variant big   # resumes from full_60k
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import config as _cfg
from config import GS_DIR, OUTPUT_DIR, VARIANTS, Variant, get_scene_variant, set_data_dir

# Import COLMAP binary reader/writer directly (avoid _3dgs package __init__ which needs CUDA)
import importlib.util
_colmap_loader_spec = importlib.util.spec_from_file_location(
    "colmap_loader", str(ROOT / "_3dgs" / "scene" / "colmap_loader.py"))
_colmap_loader = importlib.util.module_from_spec(_colmap_loader_spec)
_colmap_loader_spec.loader.exec_module(_colmap_loader)
read_extrinsics_binary = _colmap_loader.read_extrinsics_binary
write_extrinsics_binary = _colmap_loader.write_extrinsics_binary


def _filter_colmap_to_existing_images(colmap_dir: Path, img_dir: Path) -> None:
    """Filter COLMAP images.bin to only include images that exist in img_dir.

    Rewrites images.bin in place. Keeps points3D.bin unchanged.
    """
    bin_path = colmap_dir / "images.bin"
    if not bin_path.exists():
        return

    images = read_extrinsics_binary(str(bin_path))
    existing = {f.name for f in img_dir.iterdir() if f.is_file()}
    filtered = {iid: img for iid, img in images.items() if img.name in existing}
    dropped = len(images) - len(filtered)
    if dropped:
        print(f"  [COLMAP] filtering: {len(images)} -> {len(filtered)} ({dropped} images without files)")
        write_extrinsics_binary(filtered, str(bin_path))
    else:
        print(f"  [COLMAP] {len(images)} images, all files present OK")


def prepare_scene(scene: str, work_dir: Path) -> Path:
    """Copy scene data into working directory in COLMAP format 3DGS expects."""
    src = _cfg.DATA_DIR / scene
    dst = work_dir / scene
    dst.mkdir(parents=True, exist_ok=True)

    # Images
    img_src = src / "train" / "images"
    if not img_src.exists():
        img_src = src / "images"
    img_dst = dst / "images"
    if img_src.exists() and not img_dst.exists():
        shutil.copytree(str(img_src), str(img_dst))

    # COLMAP sparse
    sp_src = src / "train" / "sparse"
    if not sp_src.exists():
        sp_src = src / "sparse"
    sp_dst = dst / "sparse"
    if sp_src.exists() and not sp_dst.exists():
        shutil.copytree(str(sp_src), str(sp_dst))

    # Depth maps (pre-generated)
    dep_src = src / "depths"
    dep_dst = dst / "depths"
    if dep_src.exists() and not dep_dst.exists():
        shutil.copytree(str(dep_src), str(dep_dst))

    # depth_params.json
    dp = src / "depth_params.json"
    if dp.exists():
        shutil.copy(str(dp), str(dst / "depth_params.json"))

    # Filter COLMAP data to only reference images that actually exist
    _filter_colmap_to_existing_images(sp_dst / "0", img_dst)

    # test_poses.csv
    for tp in [src / "test" / "test_poses.csv", src / "test_poses.csv"]:
        if tp.exists():
            shutil.copy(str(tp), str(dst / "test_poses.csv"))
            break

    print(f"  [DATA] {scene}: {len(list(img_dst.glob('*')))} images prepared")
    return dst


def train(scene: str, variant: Variant, gs_dir: Path | None = None,
           resume_from: str | None = None,
           resume_path: str | None = None) -> bool:
    """Train one 3DGS variant. Returns True on success.

    Uses list-based subprocess.run for safety (no shell injection).
    Supports checkpoint resume via:
      - variant.start_checkpoint: .pth checkpoint (preserves optimizer state)
      - variant.resume_ply: PLY from another variant (preserves Gaussians only)
      - --resume-from: loads PLY from any variant at runtime
      - --resume-path: custom path to load PLY from (e.g. from other notebook)
    """
    if gs_dir is None:
        gs_dir = GS_DIR
    if not (gs_dir / "train.py").exists():
        print(f"[ERROR] 3DGS not found at {gs_dir}. Set GS_DIR in config.py or use --gs-dir")
        return False

    # ── CUDA memory optimization (T4 friendly) ──
    # PYTORCH_ALLOC_CONF=expandable_segments:True reduces memory fragmentation
    # that causes OOM during densification (split/clone operations).
    # Without this, PyTorch reserves large contiguous blocks that fragment over time.
    if "PYTORCH_ALLOC_CONF" in os.environ:
        os.environ["PYTORCH_ALLOC_CONF"] += ",expandable_segments:True"
    else:
        os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

    work_dir = OUTPUT_DIR / "workspaces"
    scene_dir = prepare_scene(scene, work_dir)

    model_path = OUTPUT_DIR / "models" / scene / variant.name
    depth_dir = scene_dir / "depths"
    depth_arg = str(depth_dir) if depth_dir.exists() and variant.depth else ""

    # ── Checkpoint resume: find base model for resume ──
    base_model: str | None = None
    if variant.start_checkpoint:
        base_candidate = OUTPUT_DIR / "models" / scene / variant.start_checkpoint
        pc_dir = base_candidate / "point_cloud"
        if pc_dir.exists():
            # Find latest iteration
            its = []
            for d in pc_dir.iterdir():
                if d.is_dir() and d.name.startswith("iteration_"):
                    try:
                        its.append(int(d.name.split("_")[1]))
                    except ValueError:
                        pass
            if its:
                load_iter = max(its)
                base_cp = base_candidate / f"chkpnt{load_iter}.pth"
                if base_cp.exists():
                    base_model = str(base_candidate)
                    print(f"  [RESUME.pth] from {base_candidate.name}/chkpnt{load_iter}.pth")
                else:
                    # No .pth but PLY exists → fallback to PLY copy
                    print(f"  [INFO] No .pth at {base_candidate.name}, trying PLY fallback...")
            else:
                print(f"  [WARN] No checkpoint found in {base_candidate}, training from scratch")
        else:
            print(f"  [WARN] Base model {base_candidate} not found, training from scratch")

    # ── Resume from PLY: copy source variant's Gaussians (no optimizer state) ──
    # This allows continuing training from quick variants without restarting from scratch.
    # The optimizer will reinitialize (LR starts from 0), but Gaussian positions/shapes
    # from the previous training are preserved — saving 50-80% training time.
    # Priority: --resume-from CLI override > variant.resume_ply config
    # ── Resolve PLY source: custom path or default model dir ──
    if resume_path and not resume_from:
        # --resume-path without --resume-from: path IS the full variant directory
        ply_source = "custom"
        custom_source = Path(resume_path)
    elif resume_from:
        ply_source = resume_from
    elif variant.resume_ply and not base_model:
        ply_source = variant.resume_ply
    else:
        ply_source = None

    if ply_source:
        # --resume-path overrides default model path (for loading from other notebooks)
        if resume_path and resume_from:
            # --resume-path /path/to/models + --resume-from variant_name
            src_variant = Path(resume_path) / ply_source
        elif resume_path and not resume_from:
            # --resume-path /path/to/models/variant (full path)
            src_variant = custom_source
        else:
            # Default: OUTPUT_DIR/models/scene/variant_name
            src_variant = OUTPUT_DIR / "models" / scene / ply_source
        src_pc_dir = src_variant / "point_cloud"
        if src_pc_dir.exists():
            # Find latest iteration in source model
            its = []
            for d in src_pc_dir.iterdir():
                if d.is_dir() and d.name.startswith("iteration_"):
                    try:
                        its.append(int(d.name.split("_")[1]))
                    except ValueError:
                        pass
            if its:
                load_iter = max(its)
                src_ply = src_pc_dir / f"iteration_{load_iter}" / "point_cloud.ply"
                if src_ply.exists():
                    # Create target model path structure
                    tgt_pc_dir = model_path / "point_cloud" / f"iteration_{load_iter}"
                    tgt_pc_dir.mkdir(parents=True, exist_ok=True)
                    tgt_ply = tgt_pc_dir / "point_cloud.ply"
                    shutil.copy(str(src_ply), str(tgt_ply))
                    # Show useful source info (variant name or actual path)
                    src_label = f"{ply_source}/iter_{load_iter}" if ply_source != "custom" else str(src_variant)
                    print(f"  [RESUME.PLY] {src_label} → {variant.name}")
                    print(f"    Copy {src_ply.name} ({src_ply.stat().st_size / 1024 / 1024:.1f} MB)")
                else:
                    print(f"  [WARN] No PLY at {src_ply}, training from scratch")
            else:
                print(f"  [WARN] No iterations in {src_pc_dir}, training from scratch")
        else:
            print(f"  [WARN] Source model {src_variant} not found, training from scratch")

    # Build command as list (safe, no shell injection)
    cmd = [sys.executable, "train.py"] + variant.args_list(
        str(scene_dir), str(model_path), depth_arg, base_model or ""
    )

    print(f"\n{'='*60}")
    print(f"TRAIN: {scene}/{variant.name} ({variant.iters} iters)")
    print(f"  sh_degree={variant.sh_degree} eval_mode={variant.eval_mode} lambda_dssim={variant.lambda_dssim}")
    print(f"  white_bg={variant.white_bg} depth_l1_weight_init={variant.depth_l1_weight_init}")
    print(f"  densify_until={variant.densify_until_iter} percent_dense={variant.percent_dense}")
    if resume_from:
        print(f"  --resume-from={resume_from}")
    elif variant.start_checkpoint:
        print(f"  resume_from={variant.start_checkpoint}")
    elif variant.resume_ply:
        print(f"  resume_ply={variant.resume_ply}")
    print(f"{'='*60}")

    t0 = time.time()
    r = subprocess.run(cmd, cwd=str(gs_dir), capture_output=True, text=True)
    elapsed = time.time() - t0

    if r.returncode == 0:
        print(f"  [OK] {scene}/{variant.name}: DONE in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  [FAIL] {scene}/{variant.name}: after {elapsed/60:.1f} min")
        for line in (r.stdout + r.stderr).splitlines()[-15:]:
            print(f"    {line}")
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="full_60k")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--check", action="store_true",
                    help="Smoke test: 100 iters, no densification, no eval")
    p.add_argument("--gs-dir", default=None)
    p.add_argument("--data-dir", default=None, help="Override data directory")
    p.add_argument("--resume-from", default=None,
                    help="Load PLY from this variant and continue training (e.g. --resume-from quick_15k)")
    p.add_argument("--resume-path", default=None,
                    help="Custom path to model directory (for loading from other notebooks). "
                         "Combine with --resume-from or use as full path."
                         "E.g. --resume-path TWIN_VAR/output/models/HCM0421 --resume-from fast")
    args = p.parse_args()

    if args.data_dir:
        set_data_dir(args.data_dir)

    vmap = {v.name: v for v in VARIANTS}
    if args.variant not in vmap:
        print(f"Unknown variant '{args.variant}'. Available: {list(vmap)}")
        sys.exit(1)

    variant = vmap[args.variant]
    if args.iters:
        variant.iters = args.iters
        variant.position_lr_max_steps = args.iters  # Sync LR schedule with custom iters

    # Apply per-scene tuning overrides (indoor vs outdoor BTS)
    variant = get_scene_variant(variant, args.scene)

    # Apply smoke-test override LAST so per-scene tuning doesn't undo it
    if args.check:
        variant.iters = 100
        variant.densify_until_iter = 0
        variant.eval_mode = False
        print(f"  [CHECK] Override: {variant.iters} iters, no densify, no eval")

    success = train(args.scene, variant, Path(args.gs_dir) if args.gs_dir else None,
                     resume_from=args.resume_from,
                     resume_path=args.resume_path)
    sys.exit(0 if success else 1)
