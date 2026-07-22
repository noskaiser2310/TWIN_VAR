"""Train a single 3DGS variant locally. Wraps gaussian-splatting/train.py.

Usage:
    python train.py --scene bonsai --variant full_60k
    python train.py --scene bonsai --variant fast --iters 7000
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import DATA_DIR, GS_DIR, OUTPUT_DIR, VARIANTS, Variant


def prepare_scene(scene: str, work_dir: Path) -> Path:
    """Copy scene data into working directory in COLMAP format 3DGS expects."""
    src = DATA_DIR / scene
    dst = work_dir / scene
    dst.mkdir(parents=True, exist_ok=True)

    img_src = src / "train" / "images"
    if not img_src.exists():
        img_src = src / "images"
    img_dst = dst / "images"
    if not img_dst.exists():
        shutil.copytree(str(img_src), str(img_dst))

    sp_src = src / "train" / "sparse"
    if not sp_src.exists():
        sp_src = src / "sparse"
    sp_dst = dst / "sparse"
    if sp_src.exists() and not sp_dst.exists():
        shutil.copytree(str(sp_src), str(sp_dst))

    # depth maps (if pre-generated)
    dep_src = src / "depths"
    dep_dst = dst / "depths"
    if dep_src.exists() and not dep_dst.exists():
        shutil.copytree(str(dep_src), str(dep_dst))

    # depth_params.json
    dp = src / "depth_params.json"
    if dp.exists():
        shutil.copy(str(dp), str(dst / "depth_params.json"))

    # test_poses.csv
    tp = src / "test" / "test_poses.csv"
    if not tp.exists():
        tp = src / "test_poses.csv"
    if tp.exists():
        shutil.copy(str(tp), str(dst / "test_poses.csv"))

    print(f"  [DATA] {scene}: {len(list(img_dst.glob('*')))} images prepared")
    return dst


def train(scene: str, variant: Variant, gs_dir: Path | None = None) -> bool:
    """Train one 3DGS variant. Returns True on success."""
    if gs_dir is None:
        gs_dir = GS_DIR
    if not (gs_dir / "train.py").exists():
        print(f"[ERROR] 3DGS not found at {gs_dir}. Set GS_DIR in config.py or use --gs-dir")
        return False

    work_dir = OUTPUT_DIR / "workspaces"
    scene_dir = prepare_scene(scene, work_dir)

    model_path = OUTPUT_DIR / "models" / scene / variant.name
    depth_dir = scene_dir / "depths"
    depth_arg = str(depth_dir) if depth_dir.exists() and variant.depth else ""

    cmd = (
        f'cd "{gs_dir}" && python train.py '
        f'{variant.args(str(scene_dir), str(model_path), depth_arg)} '
        f'--quiet'
    )

    print(f"\n{'='*60}")
    print(f"TRAIN: {scene}/{variant.name} ({variant.iters} iters)")
    print(f"{'='*60}")

    t0 = time.time()
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    elapsed = time.time() - t0

    if r.returncode == 0:
        print(f"  [OK] {scene}/{variant.name}: DONE in {elapsed/60:.1f} min")
        return True
    else:
        print(f"  [FAIL] {scene}/{variant.name}: after {elapsed/60:.1f} min")
        for line in (r.stdout + r.stderr).splitlines()[-10:]:
            print(f"    {line}")
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--scene", required=True)
    p.add_argument("--variant", default="full_60k")
    p.add_argument("--iters", type=int, default=None)
    p.add_argument("--gs-dir", default=None)
    args = p.parse_args()

    vmap = {v.name: v for v in VARIANTS}
    if args.variant not in vmap:
        print(f"Unknown variant '{args.variant}'. Available: {list(vmap)}")
        sys.exit(1)

    variant = vmap[args.variant]
    if args.iters:
        variant.iters = args.iters

    success = train(args.scene, variant, Path(args.gs_dir) if args.gs_dir else None)
    sys.exit(0 if success else 1)
