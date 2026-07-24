"""
TWIN_VAR — Setup Environment Script
=====================================
All-in-one setup for Kaggle: installs deps, builds CUDA extensions,
patches config, and verifies imports. Run ONCE per notebook session.

Usage:
    python setup_env.py                         # Full setup
    python setup_env.py --skip-apt              # Skip apt installs
    python setup_env.py --skip-pip              # Skip pip installs
    python setup_env.py --skip-cuda-ext         # Skip CUDA extension build
    python setup_env.py --no-patch-sparse-adam  # Don't disable sparse_adam
    python setup_env.py --verify-only           # Only verify imports

Works with Kaggle Internet OFF after first run (cached packages).
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# ── Paths ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent  # TWIN_VAR repo root
SRC = ROOT / "src"
SUBMODULES = SRC / "_3dgs" / "submodules"
CUDA_EXTENSIONS = [
    ("simple-knn", SUBMODULES / "simple-knn"),
    ("fused-ssim", SUBMODULES / "fused-ssim"),
    ("diff-gaussian-rasterization", SUBMODULES / "diff-gaussian-rasterization"),
]


# ── Submodule fallback URLs ──────────────────────────────────
# simple-knn GitLab thường không truy cập được trên Kaggle
SUBMODULE_FALLBACKS = {
    "simple-knn": "https://github.com/camenduru/simple-knn.git",
}


# ── System Dependencies ────────────────────────────────────────
APT_PACKAGES = [
    "colmap",
    "ninja-build",
    "g++-10",
    "gcc-10",
]

PIP_PACKAGES = [
    "plyfile",
    "tqdm",
    "opencv-python",
    "joblib",
    "pillow",
    "scipy",
    "rich",
]

# ── Helpers ─────────────────────────────────────────────────────


def run(cmd: list[str], cwd: Path | None = None, desc: str = "") -> bool:
    """Run a command, print progress, return success."""
    label = f" [{desc}]" if desc else ""
    print(f"  ⏳ RUNNING:{label} {' '.join(str(c) for c in cmd)}")
    try:
        r = subprocess.run(cmd, cwd=cwd, check=False,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True)
        if r.returncode != 0:
            print(f"  ❌ FAILED (exit={r.returncode}):{label}")
            for line in r.stdout.splitlines()[-10:]:
                print(f"     {line}")
            return False
        print(f"  ✅ OK:{label}")
        return True
    except FileNotFoundError as e:
        print(f"  ❌ NOT FOUND:{label} — {e}")
        return False
    except Exception as e:
        print(f"  ❌ ERROR:{label} — {e}")
        return False


# ── Phases ──────────────────────────────────────────────────────


def phase_apt() -> bool:
    """Install system packages via apt-get."""
    print("\n📦 PHASE: System Dependencies (apt-get)")
    ok = run(
        ["apt-get", "update", "-qq"],
        desc="apt-get update",
    )
    if not ok:
        return False
    return run(
        ["apt-get", "install", "-y", "-qq"] + APT_PACKAGES,
        desc=f"apt-get install {' '.join(APT_PACKAGES)}",
    )


def phase_pip() -> bool:
    """Upgrade pip/setuptools/wheel, install Python packages."""
    print("\n🐍 PHASE: Python Dependencies (pip)")

    # Upgrade pip & build tools
    ok = run(
        [sys.executable, "-m", "pip", "install", "--upgrade",
         "pip<27", "setuptools<82", "wheel", "ninja", "-q"],
        desc="upgrade pip/setuptools/wheel/ninja",
    )
    if not ok:
        return False

    # Core packages
    ok = run(
        [sys.executable, "-m", "pip", "install", "-q"] + PIP_PACKAGES,
        desc=f"pip install {' '.join(PIP_PACKAGES)}",
    )
    if not ok:
        return False

    # requirements.txt (optional, may be subset of above)
    req_txt = SRC / "requirements.txt"
    if req_txt.exists():
        ok = run(
            [sys.executable, "-m", "pip", "install", "-r", str(req_txt), "-q"],
            desc="pip install -r src/requirements.txt",
        )
        if not ok:
            return False

    return True


def _set_compiler_env() -> None:
    """Set GCC-10 as default compiler for CUDA extensions."""
    gcc10 = shutil.which("gcc-10")
    gpp10 = shutil.which("g++-10")
    if gcc10 and gpp10:
        os.environ["CC"] = gcc10
        os.environ["CXX"] = gpp10
        os.environ["CUDAHOSTCXX"] = gpp10
        print(f"  ⚙  Compiler: CC={gcc10}, CXX={gpp10}")
    else:
        print("  ⚠  gcc-10/g++-10 not found, using default compiler")


def phase_init_submodules() -> bool:
    """Initialize 3DGS submodules (simple-knn, fused-ssim, rasterizer).

    Handles fallback URLs for repos that fail to clone from original source.
    """
    print("\n📦 PHASE: Initialize Submodules")
    all_ok = True

    # Submodule paths relative to repo root (as registered in .gitmodules)
    # The submodules are at src/_3dgs/submodules/<name>
    _gs_rel = "src/_3dgs/submodules"
    submodule_infos = [
        ("diff-gaussian-rasterization", f"{_gs_rel}/diff-gaussian-rasterization", True),   # recursive (needs third_party/glm)
        ("fused-ssim", f"{_gs_rel}/fused-ssim", False),
        ("simple-knn", f"{_gs_rel}/simple-knn", False),
    ]

    for name, rel_path, recursive in submodule_infos:
        ext_path = SUBMODULES / name
        if ext_path.exists() and (ext_path / "setup.py").exists():
            print(f"  ✅ {name}: already initialized")
            continue

        # Try git submodule update
        print(f"  ⏳ Initializing {name}...")
        cmd = ["git", "submodule", "update", "--init"]
        if recursive:
            cmd.append("--recursive")
        cmd.append(rel_path)
        r = subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)
        if r.returncode == 0 and (ext_path / "setup.py").exists():
            print(f"  ✅ {name}: initialized via git submodule")
            continue

        # Fallback: clone from alternative URL
        fallback_url = SUBMODULE_FALLBACKS.get(name)
        if fallback_url:
            print(f"  ⏳ {name}: git submodule failed, trying fallback {fallback_url}")
            if ext_path.exists():
                shutil.rmtree(ext_path)
            ext_path.parent.mkdir(parents=True, exist_ok=True)
            r = subprocess.run(
                ["git", "clone", "--recursive" if recursive else "", fallback_url, str(ext_path)],
                cwd=ROOT, check=False, capture_output=True, text=True,
            )
            if r.returncode == 0 and (ext_path / "setup.py").exists():
                print(f"  ✅ {name}: initialized via fallback clone")
                continue

        print(f"  ❌ {name}: failed to initialize (no setup.py)")
        all_ok = False

    return all_ok


def phase_cuda_extensions() -> bool:
    """Build 3DGS CUDA extensions (simple-knn, fused-ssim, rasterizer)."""
    print("\n⚙️  PHASE: Build CUDA Extensions")
    _set_compiler_env()

    # Limit parallel compilation jobs to avoid OOM on Kaggle T4
    os.environ["MAX_JOBS"] = "2"

    # Uninstall first to avoid conflicts
    for ext_name, _ in CUDA_EXTENSIONS:
        run(
            [sys.executable, "-m", "pip", "uninstall", "-y",
             ext_name.replace("-", "_")],
            desc=f"uninstall {ext_name}",
        )

    # Build & install each extension (output streams in real-time)
    all_ok = True
    for ext_name, ext_path in CUDA_EXTENSIONS:
        if not ext_path.exists():
            print(f"  ⚠  SKIP {ext_name}: path not found {ext_path}")
            all_ok = False
            continue
        label = f"build & install {ext_name}"
        print(f"  ⏳ BUILDING: {label}...")
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install",
             "--no-build-isolation", str(ext_path)],
            cwd=ROOT, check=False, text=True,
        )
        if r.returncode == 0:
            print(f"  ✅ OK: {label}")
        else:
            print(f"  ❌ FAILED (exit={r.returncode}): {label}")
            all_ok = False

    return all_ok


def phase_patch_config() -> None:
    """Disable sparse_adam for Kaggle compatibility."""
    config_path = SRC / "config.py"
    if not config_path.exists():
        print("  ⚠  config.py not found, skipping patch")
        return

    with open(config_path) as f:
        content = f.read()

    # Count occurrences
    count_before = content.count("sparse_adam=True")
    content = re.sub(r"sparse_adam=True", "sparse_adam=False", content)
    count_after = content.count("sparse_adam=True")

    if count_before > 0:
        with open(config_path, "w") as f:
            f.write(content)
        print(f"  🔧 PATCHED: sparse_adam=True → False ({count_before} occurrences)")
    else:
        print("  ℹ️  Already patched or no sparse_adam=True found")


def phase_verify() -> bool:
    """Verify all imports work correctly."""
    print("\n✅ PHASE: Verify Imports")

    checks = {
        "PyTorch": lambda: __import__("torch"),
        "simple-knn distCUDA2": lambda: __import__("simple_knn")._C.distCUDA2(
            __import__("torch").zeros((1, 3)).cuda()
        ),
        "fused-ssim fused_ssim": lambda: __import__("fused_ssim").fused_ssim,
        "diff-gaussian-rasterization _C": lambda: __import__(
            "diff_gaussian_rasterization"
        )._C,
        "plyfile": lambda: __import__("plyfile"),
        "tqdm": lambda: __import__("tqdm"),
        "cv2": lambda: __import__("cv2"),
        "PIL": lambda: __import__("PIL"),
        "rich": lambda: __import__("rich"),
    }

    all_ok = True
    for label, fn in checks.items():
        try:
            fn()
            print(f"  ✅ {label}")
        except Exception as e:
            print(f"  ❌ {label}: {e}")
            all_ok = False

    if all_ok:
        torch = __import__("torch")
        print(f"\n  🎯 All imports OK | PyTorch {torch.__version__} | CUDA {torch.version.cuda}")
    else:
        print(f"\n  ⚠️  Some imports failed — check errors above")

    return all_ok


# ── Main ────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser(
        description="TWIN_VAR — Setup Environment Script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--skip-apt", action="store_true", help="Skip apt-get installs")
    p.add_argument("--skip-pip", action="store_true", help="Skip pip installs")
    p.add_argument("--skip-cuda-ext", action="store_true", help="Skip CUDA extension build")
    p.add_argument("--no-patch-sparse-adam", action="store_true",
                   help="Don't disable sparse_adam in config")
    p.add_argument("--verify-only", action="store_true", help="Only verify imports, skip all setup")
    args = p.parse_args()

    print("=" * 60)
    print("TWIN_VAR — Setup Environment")
    print("=" * 60)
    print(f"  Repo root: {ROOT}")
    print(f"  Python:    {sys.version.split()[0]}")
    print(f"  CUDA ext:  {SUBMODULES}")

    # ── Pre-check: are we in the right directory? ──
    if not (SRC / "config.py").exists():
        print(f"\n  ❌ ERROR: Not in TWIN_VAR repo root — src/config.py not found at {SRC}")
        print("     Run this script from the TWIN_VAR directory:")
        print("     cd /kaggle/working/TWIN_VAR && python setup_env.py")
        return

    if args.verify_only:
        phase_verify()
        return

    # ── Run phases ──
    if not args.skip_apt:
        phase_apt()

    if not args.skip_pip:
        phase_pip()

    # Initialize submodules BEFORE building CUDA extensions
    # (git clone --recursive không được dùng, submodules cần init riêng)
    phase_init_submodules()

    if not args.skip_cuda_ext:
        phase_cuda_extensions()

    if not args.no_patch_sparse_adam:
        phase_patch_config()

    phase_verify()

    print("\n" + "=" * 60)
    print("🏁 SETUP COMPLETE!")
    print("=" * 60)
    print("  Now you can run:")
    print("    python src/train.py --scene HCM0421 --variant quick_15k --data-dir ...")
    print("    python src/main.py --scenes HCM0421 --data-dir ...")
    print("=" * 60)


if __name__ == "__main__":
    main()
