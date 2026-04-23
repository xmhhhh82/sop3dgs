# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **3D semantic point cloud pipeline** for agricultural phenotyping (corn/maize plant analysis). It processes video of corn plants through a 6-step automated pipeline that produces globally-consistent 3D semantic instance segmentation.

The pipeline is designed to run on a **Linux GPU server** with conda environments, not the local Windows machine where this repo is stored. All hardcoded paths reference the server filesystem (`/datashare/`, `/extdatashare/`).

## Pipeline Architecture (pipeline.sh)

The main entry point is `pipeline.sh`, which orchestrates 6 sequential steps across two conda environments:

| Step | Script | Conda Env | Purpose |
|------|--------|-----------|---------|
| 1 | `vggsfm_cli.py` | `vggsfm` | Video frame extraction (ffmpeg) + VGGSfM sparse reconstruction |
| 2 | `reason_cli.py` | `gaussian_splatting` | YOLO instance segmentation (per-frame 2D masks) |
| 3 | `train_0211之前.py` | `gaussian_splatting` | 2D Gaussian Splatting training (based on INRIA's 2D-GS) |
| 4 | `pirender深度图.py` | `gaussian_splatting` | Depth map rendering for all frames |
| 5 | `run_matchingljp0326.py` | `gaussian_splatting` | 3D semantic instance matching across frames |
| 6 | `pirenderljp0326.py` | `gaussian_splatting` | Final rendering with global semantic IDs |

## Running the Pipeline

```bash
# Full pipeline from video
bash pipeline.sh --video /path/to/video.mp4

# Resume from a specific step (skip completed steps)
bash pipeline.sh --scene_dir /path/to/scene \
    --skip_vggsfm --skip_yolo --skip_train --skip_depth \
    --n_keyframes 20

# Manual max_depth override
bash pipeline.sh --video /path/to/video.mp4 --max_depth 80.0
```

Key parameters: `--n_keyframes` (default 15), `--fps` (default 2), `--yolo_conf` (default 0.25), `--iterations` (default 30000).

## Scene Directory Structure

After a full pipeline run, a scene directory looks like:
```
<scene_dir>/
  images/                     # Extracted video frames (from Step 1)
  sparse/0/                   # COLMAP-format sparse reconstruction
  masks_results/
    integer_masks/            # Per-frame instance masks + class_info JSONs
    visualization/            # Optional YOLO visualization
  output_<name>/
    cameras.json              # Camera parameters from 2D-GS
    train/ours_30000/depth/   # Rendered depth maps
  数据驱动匹配/                # 3D matching output
    id_mapping.json           # Global instance ID mapping
    unified_masks/            # Globally-consistent masks
```

## Key Technical Details

### 3D Matching System (Step 5 — most complex)

`run_matchingljp0326.py` contains `OptimizedCenterPoint3DMatcher`, which extends the base `CenterPoint3DMatcher` from an external script. Core innovations:

- **Auto max_depth detection**: Scans depth maps to set P95 depth as upper bound
- **Smart keyframe selection**: Quality scoring (valid depth ratio + instance count + depth stability), then uniform temporal sampling
- **Three-round progressive matching**: R1 strict threshold -> R2 loose (4x) -> R3 nearest-neighbor (12x) -> new global ID assignment
- **Adaptive spatial threshold**: Adjusts based on point cloud bounding box diagonal (3% for corn plants)
- Matching config is tuned for corn plants: `spatial_threshold=0.15m`, `top_k_min_depths=300` (small leaf area)

### External Dependencies (server paths)

- VGGSfM code: `/extdatashare/liuzy0/code/vggsfm` (contains `demo.py`)
- 2D-GS code: `/datashare/dir_liusha/2d-gaussian-splatting_gsid_seg`
- Original matcher: `/datashare/dir_liusha/xibeinonglin/1_15_提取表型/`
- YOLO model: `/extdatashare/liuzy0/ma_dataljp0330/runs/segment/corn-seg-ljp0326/weights/best.pt`

### matching_fix_patch.py

This is a **documentation/patch file**, not directly executed. It contains method-level patches (with instructions) to be manually integrated into the base `CenterPoint3DMatcher` class. The optimized matcher in `run_matchingljp0326.py` already incorporates these fixes.

## Language Note

Code comments, variable names, and output messages are primarily in **Chinese (Simplified)**. Directory names like `数据驱动匹配` (data-driven matching) and `pirender深度图` (depth map) are part of the codebase conventions.