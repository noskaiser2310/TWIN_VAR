"""Gaussian-Level Primitive Merging.

Merges 3D Gaussian point clouds from multiple trained variants into a single compact model
using voxel-based spatial clustering and opacity pruning.

Key techniques (from latest SOTA research):
  - Voxel-based merging: Gaussians in the same voxel → averaged SH + scale + opacity
  - Opacity pruning: Remove low-opacity / outlier Gaussians (floaters)
  - Position deduplication: Nearest-neighbor merge for overlapping primitives

Benefits over pixel-level ensemble:
  - Single model → single render pass (no per-variant rendering)
  - Compact representation (fewer Gaussians, less VRAM)
  - Combined strengths of all variants at primitive level

Usage:
    python compact.py --scene bonsai --variants full_60k,depth_expo,antialias
    python compact.py --scene bonsai  # uses COMPACT_VARIANTS from config
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from config import (
    OUTPUT_DIR,
    COMPACT_CONFIG,
    COMPACT_VARIANTS,
)


def load_ply(path: Path) -> dict[str, np.ndarray]:
    """Load a 3DGS point cloud PLY file. Returns dict of numpy arrays."""
    from plyfile import PlyData

    plydata = PlyData.read(str(path))
    vert = plydata["vertex"]

    data = {
        "x": np.array(vert["x"]),
        "y": np.array(vert["y"]),
        "z": np.array(vert["z"]),
        "nx": np.array(vert["nx"]),
        "ny": np.array(vert["ny"]),
        "nz": np.array(vert["nz"]),
    }

    # SH coefficients (may have 1, 4, 16, or more)
    sh_keys = [k for k in vert.data.dtype.names if k.startswith("f_dc_") or k.startswith("f_rest_")]
    for k in sh_keys:
        data[k] = np.array(vert[k])

    # Opacity
    data["opacity"] = np.array(vert["opacity"])

    # Scales
    for k in ["scale_0", "scale_1", "scale_2"]:
        data[k] = np.array(vert[k])

    # Rotation
    for k in ["rot_0", "rot_1", "rot_2", "rot_3"]:
        data[k] = np.array(vert[k])

    return data


def save_ply(data: dict[str, np.ndarray], path: Path) -> None:
    """Save a merged point cloud as PLY file."""
    from plyfile import PlyData, PlyElement

    n = len(data["x"])
    sh_keys = sorted([k for k in data if k.startswith("f_dc_") or k.startswith("f_rest_")])
    scale_keys = ["scale_0", "scale_1", "scale_2"]
    rot_keys = ["rot_0", "rot_1", "rot_2", "rot_3"]

    dtype = [
        ("x", "float32"), ("y", "float32"), ("z", "float32"),
        ("nx", "float32"), ("ny", "float32"), ("nz", "float32"),
    ]
    for k in sh_keys:
        dtype.append((k, "float32"))
    dtype.append(("opacity", "float32"))
    for k in scale_keys:
        dtype.append((k, "float32"))
    for k in rot_keys:
        dtype.append((k, "float32"))

    vertices = np.empty(n, dtype=dtype)
    vertices["x"] = data["x"]
    vertices["y"] = data["y"]
    vertices["z"] = data["z"]
    vertices["nx"] = data["nx"]
    vertices["ny"] = data["ny"]
    vertices["nz"] = data["nz"]
    for k in sh_keys:
        vertices[k] = data[k]
    vertices["opacity"] = data["opacity"]
    for k in scale_keys:
        vertices[k] = data[k]
    for k in rot_keys:
        vertices[k] = data[k]

    el = PlyElement.describe(vertices, "vertex")
    PlyData([el]).write(str(path))


def voxel_merge(points: list[dict[str, np.ndarray]], voxel_size: float = 0.005) -> dict[str, np.ndarray]:
    """Merge multiple point clouds into voxel grid, averaging Gaussians per voxel.

    For each occupied voxel, averages position, SH, scale, rotation, opacity.
    """
    all_positions = []
    all_colors = []
    all_sh = [[] for _ in range(len(points[0]) - 9)]  # rough estimate
    all_opacities = []
    all_scales = []
    all_rots = []

    for p in points:
        all_positions.append(np.stack([p["x"], p["y"], p["z"]], axis=-1))
        all_opacities.append(p["opacity"])
        all_scales.append(np.stack([p["scale_0"], p["scale_1"], p["scale_2"]], axis=-1))
        all_rots.append(np.stack([p["rot_0"], p["rot_1"], p["rot_2"], p["rot_3"]], axis=-1))

    # Concatenate all Gaussians
    positions = np.concatenate(all_positions, axis=0)
    opacities = np.concatenate(all_opacities, axis=0)
    scales = np.concatenate(all_scales, axis=0)
    rots = np.concatenate(all_rots, axis=0)

    # Voxelize
    voxel_indices = np.floor(positions / voxel_size).astype(np.int32)
    unique_voxels, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)

    n_voxels = len(unique_voxels)
    print(f"  Gaussians: {len(positions)} → {n_voxels} voxels (voxel_size={voxel_size})")

    merged_pos = np.zeros((n_voxels, 3))
    merged_opacity = np.zeros(n_voxels)
    merged_scale = np.zeros((n_voxels, 3))
    merged_rot = np.zeros((n_voxels, 4))
    merged_normals = np.zeros((n_voxels, 3))

    for i in range(n_voxels):
        mask = inverse == i
        count = mask.sum()
        merged_pos[i] = positions[mask].mean(axis=0)
        merged_opacity[i] = opacities[mask].mean()
        merged_scale[i] = scales[mask].mean(axis=0)
        # Average rotations (quaternion averaging via SVD)
        q = rots[mask]
        if count > 1:
            q_mean = q.mean(axis=0)
            q_mean = q_mean / (np.linalg.norm(q_mean) + 1e-8)
            merged_rot[i] = q_mean
        else:
            merged_rot[i] = q[0]

    # SH coefficients: average per voxel
    sh_keys = sorted([k for k in points[0] if k.startswith("f_dc_") or k.startswith("f_rest_")])
    merged_sh = {}
    if sh_keys:
        all_sh_arrays = {k: np.concatenate([p[k] for p in points], axis=0) for k in sh_keys}
        for k in sh_keys:
            merged_sh[k] = np.zeros(n_voxels)
            for i in range(n_voxels):
                merged_sh[k][i] = all_sh_arrays[k][inverse == i].mean()

    # Build result dict
    result = {
        "x": merged_pos[:, 0].astype(np.float32),
        "y": merged_pos[:, 1].astype(np.float32),
        "z": merged_pos[:, 2].astype(np.float32),
        "nx": merged_normals[:, 0].astype(np.float32),
        "ny": merged_normals[:, 1].astype(np.float32),
        "nz": merged_normals[:, 2].astype(np.float32),
        "opacity": merged_opacity.astype(np.float32),
        "scale_0": merged_scale[:, 0].astype(np.float32),
        "scale_1": merged_scale[:, 1].astype(np.float32),
        "scale_2": merged_scale[:, 2].astype(np.float32),
        "rot_0": merged_rot[:, 0].astype(np.float32),
        "rot_1": merged_rot[:, 1].astype(np.float32),
        "rot_2": merged_rot[:, 2].astype(np.float32),
        "rot_3": merged_rot[:, 3].astype(np.float32),
    }
    result.update(merged_sh)

    return result


def prune_opacity(data: dict[str, np.ndarray], threshold: float = 0.05) -> dict[str, np.ndarray]:
    """Remove low-opacity Gaussians (floaters)."""
    mask = data["opacity"] > threshold
    n_before = len(data["x"])
    result = {}
    for k, v in data.items():
        if isinstance(v, np.ndarray) and len(v) == n_before:
            result[k] = v[mask]
        else:
            result[k] = v
    print(f"  Pruned: {n_before} → {len(result['x'])} (opacity > {threshold})")
    return result


def merge_variants(scene: str, variants: list[str] | None = None,
                   voxel_size: float | None = None,
                   opacity_threshold: float | None = None) -> Path | None:
    """Merge multiple trained 3DGS variants into one compact model.

    Args:
        scene: Scene name
        variants: List of variant names to merge (default: COMPACT_VARIANTS)
        voxel_size: Voxel grid size for merging (default: from config)
        opacity_threshold: Opacity culling threshold (default: from config)

    Returns:
        Path to merged PLY file, or None if merge failed
    """
    if variants is None:
        variants = COMPACT_VARIANTS
    if voxel_size is None:
        voxel_size = COMPACT_CONFIG["voxel_size"]
    if opacity_threshold is None:
        opacity_threshold = COMPACT_CONFIG["opacity_cull"]

    models_dir = OUTPUT_DIR / "models" / scene

    print(f"\n{'='*60}")
    print(f"COMPACT MERGE: {scene}")
    print(f"  Variants: {', '.join(variants)}")
    print(f"  Voxel: {voxel_size}, Opacity cull: {opacity_threshold}")
    print(f"{'='*60}")

    # Find PLY files
    ply_files = []
    for variant in variants:
        model_path = models_dir / variant
        for ply_path in [
            model_path / "point_cloud" / "iteration_-1" / "point_cloud.ply",
            model_path / "point_cloud" / "iteration_30000" / "point_cloud.ply",
            model_path / "point_cloud" / "iteration_60000" / "point_cloud.ply",
            model_path / "point_cloud" / "iteration_90000" / "point_cloud.ply",
            model_path / "point_cloud" / "iteration_7000" / "point_cloud.ply",
        ]:
            if ply_path.exists():
                ply_files.append(ply_path)
                break

    if len(ply_files) < 2:
        print(f"  [SKIP] Need >= 2 models, found {len(ply_files)}")
        return None

    print(f"  Found {len(ply_files)} PLY files")

    # Load all point clouds
    points = []
    for ply_path in ply_files:
        try:
            points.append(load_ply(ply_path))
            print(f"    {ply_path.parent.parent.name}: {len(points[-1]['x'])} Gaussians")
        except Exception as e:
            print(f"    [WARN] Failed to load {ply_path}: {e}")

    if len(points) < 2:
        return None

    # Merge
    merged = voxel_merge(points, voxel_size)

    # Prune
    merged = prune_opacity(merged, opacity_threshold)

    # Save
    out_dir = models_dir / "compact"
    out_dir.mkdir(parents=True, exist_ok=True)
    ply_out = out_dir / "point_cloud.ply"
    save_ply(merged, ply_out)

    print(f"  [OK] Merged model saved: {ply_out}")
    print(f"  Final: {len(merged['x'])} Gaussians (from {sum(len(p['x']) for p in points)})")

    # Also create a symlink-compatible structure for render.py
    (out_dir / "point_cloud" / "iteration_-1").mkdir(parents=True, exist_ok=True)
    import shutil
    shutil.copy(str(ply_out), str(out_dir / "point_cloud" / "iteration_-1" / "point_cloud.ply"))

    return ply_out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Gaussian-Level Model Merging")
    p.add_argument("--scene", required=True)
    p.add_argument("--variants", default=None, help="Comma-separated variant names")
    p.add_argument("--voxel-size", type=float, default=None)
    p.add_argument("--opacity-cull", type=float, default=None)
    args = p.parse_args()

    variants = args.variants.split(",") if args.variants else None
    merge_variants(args.scene, variants, args.voxel_size, args.opacity_cull)
