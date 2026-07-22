"""VAR 2026 — Digital Twin BTS: Auto Pipeline for Novel View Synthesis.

Architecture:
  Phase 1 (Local):   Depth map generation → data validation
  Phase 2 (Local):   Upload scenes as Kaggle private datasets
  Phase 3 (Kaggle):  Multi-variant 3DGS training per scene (parallel kernels)
  Phase 4 (Kaggle):  Render all test poses → download outputs
  Phase 5 (Local):   Best-variant selection → submission packaging

Multi-variant strategy per scene:
  1. 3DGS Baseline       — default 30k iterations
  2. 3DGS + DepthReg     — Depth-Anything v2 depth regularization
  3. 3DGS + Exposure     — per-image exposure compensation
  4. 3DGS + AntiAlias    — EWA filter (Mip-Splatting)
  5. 3DGS Full Combo     — depth + exposure + antialias + sparse_adam
  6. 3DGS Fast           — 7k iterations (backup if OOM)
"""

__version__ = "1.0.0"
