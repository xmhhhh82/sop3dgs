#!/usr/bin/env python3
"""
修改版：支持命令行指定 max_frames，适配 2_1_2024_08_14_01（58帧）
用法:
  单个文件夹:
    python run_matching.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01
  指定帧数:
    python run_matching.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01 --max_frames 58
  多个文件夹:
    python run_matching.py /path/folder1 /path/folder2 --max_frames 58
"""

import sys
# ========== 把原始脚本所在目录加入路径（按实际情况修改） ==========
ORIGINAL_SCRIPT_DIR = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型"
sys.path.insert(0, ORIGINAL_SCRIPT_DIR)

import os
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
import json
from tqdm import tqdm
import re
import random
from scipy.spatial.distance import cdist
import matplotlib
matplotlib.use('Agg')  # 无显示器环境
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation
import argparse


# ==================== 直接从原始脚本导入 CenterPoint3DMatcher ====================
# 如果无法导入，请把原始脚本里的 CenterPoint3DMatcher 类粘贴到这里
try:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "matcher_module",
        os.path.join(ORIGINAL_SCRIPT_DIR,
                     "0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测+双向匹配_0311_不跑全部帧.py")
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    CenterPoint3DMatcher = mod.CenterPoint3DMatcher
    print("✅ 成功从原始脚本导入 CenterPoint3DMatcher")
except Exception as e:
    print(f"❌ 无法导入原始脚本，请检查路径: {e}")
    sys.exit(1)


# ==================== 自动查找路径 ====================

def process_single_folder(folder_path, output_base_dir=None, max_frames=58,
                           first_frame_candidates=None):
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    if first_frame_candidates is None:
        # 默认候选帧：前5帧
        first_frame_candidates = [1, 2, 3, 4, 5]

    print(f"\n{'='*60}")
    print(f"📁 处理文件夹: {folder_name}")
    print(f"   max_frames = {max_frames}")
    print(f"   first_frame_candidates = {first_frame_candidates}")
    print(f"{'='*60}")

    # 1. 查找 depth_dir
    output_dirs = list(folder_path.glob("output_*"))
    depth_dir = None
    camera_json = None

    for output_dir in output_dirs:
        potential_depth = output_dir / "train" / "ours_30000" / "depth"
        if potential_depth.exists():
            depth_dir = str(potential_depth)
            print(f"✅ 找到深度图目录: {depth_dir}")

        potential_camera = output_dir / "cameras.json"
        if potential_camera.exists():
            camera_json = str(potential_camera)
            print(f"✅ 找到相机参数文件: {camera_json}")

    # 备选深度路径
    if not depth_dir:
        for output_dir in output_dirs:
            potential_depth = output_dir / "depth"
            if potential_depth.exists():
                depth_dir = str(potential_depth)
                print(f"✅ 找到深度图目录(备选): {depth_dir}")
                break

    # 备选相机参数
    if not camera_json:
        potential_camera = folder_path / "cameras.json"
        if potential_camera.exists():
            camera_json = str(potential_camera)
            print(f"✅ 找到相机参数文件(根目录): {camera_json}")

    # 2. 查找 mask_dir
    mask_dir = folder_path / "masks_results" / "integer_masks"
    if mask_dir.exists():
        mask_dir = str(mask_dir)
        print(f"✅ 找到掩码目录: {mask_dir}")
    else:
        mask_candidates = list(folder_path.glob("**/integer_masks"))
        if mask_candidates:
            mask_dir = str(mask_candidates[0])
            print(f"✅ 找到掩码目录(备选): {mask_dir}")
        else:
            mask_dir = None

    # 3. 确定输出目录
    if output_base_dir is None:
        output_dir = folder_path / "数据驱动匹配"
    else:
        output_dir = Path(output_base_dir) / folder_name
    output_dir = str(output_dir)
    print(f"📁 输出目录: {output_dir}")

    # 4. 检查必要路径
    missing = []
    if not depth_dir:   missing.append("深度图目录 (output_xxx/train/ours_30000/depth)")
    if not camera_json: missing.append("相机参数文件 (output_xxx/cameras.json)")
    if not mask_dir:    missing.append("掩码目录 (masks_results/integer_masks)")

    if missing:
        print(f"\n❌ 找不到以下路径，请先完成训练和深度图导出:")
        for m in missing:
            print(f"   - {m}")
        return None

    # 5. 构建 config（max_frames 从参数传入）
    config = {
        'depth_dir':    depth_dir,
        'camera_json':  camera_json,
        'mask_dir':     mask_dir,
        'output_dir':   output_dir,

        'target_class':     3,
        'fixed_class_ids':  {0: 220, 1: 221, 2: 222},
        'first_frame_candidates': first_frame_candidates,

        'depth_scale':          1000.0,
        'depth_format':         '16bit',
        'max_depth':            10.0,
        'spatial_threshold':    0.3,
        'motion_threshold':     0.5,
        'motion_weight':        0.4,
        'bidirectional_weight': 0.4,
        'shape_weight':         0.2,
        'coherence_threshold':  0.2,
        'confidence_threshold': 0.5,
        'min_instance_area':    5,

        # ★ 关键修改：使用传入的 max_frames
        'max_frames': max_frames,

        'use_motion_prior':  True,
        'use_top_k_depths':  True,
        'top_k_min_depths':  2000,
        'colormap':          'tab20',

        'pixel_depth_save': {
            'enabled':            False,   # 58帧全保存太慢，可按需改为True
            'frames_to_save':     [3, 4],
            'depth_decimals':     3,
            'min_depth':          0.0,
            'max_depth':          10.0,
            'include_coordinates': True,
            'separate_files':     True,
            'save_visualization': True,
        },

        'id_font_scale':    0.4,
        'depth_font_scale': 0.3,
        'text_color':       (255, 255, 255),
        'unmatched_color':  (128, 128, 128),
        'first_frame':      None,
    }

    return config


def process_folders(folder_paths, output_base_dir=None,
                    max_frames=58, first_frame_candidates=None):
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    print("\n" + "="*80)
    print(f"🚀 开始处理 {len(folder_paths)} 个文件夹  (max_frames={max_frames})")
    print("="*80)

    results = []
    for folder_path in folder_paths:
        try:
            config = process_single_folder(
                folder_path, output_base_dir,
                max_frames=max_frames,
                first_frame_candidates=first_frame_candidates
            )
            if config:
                matcher = CenterPoint3DMatcher(config)
                success = matcher.run_matching()
                if success:
                    matcher.visualize_results()
                results.append({'folder': folder_path,
                                 'success': success,
                                 'output_dir': config['output_dir']})
            else:
                results.append({'folder': folder_path, 'success': False,
                                 'error': '路径不完整，请先完成训练和深度图导出'})
        except Exception as e:
            import traceback
            traceback.print_exc()
            results.append({'folder': folder_path, 'success': False, 'error': str(e)})

    print("\n" + "="*80)
    print("📊 处理结果汇总")
    print("="*80)
    for r in results:
        name = Path(r['folder']).name
        if r['success']:
            print(f"✅ {name}: 成功  →  {r['output_dir']}")
        else:
            print(f"❌ {name}: 失败  —  {r.get('error','')}")
    return results


# ==================== 命令行入口 ====================

def main():
    parser = argparse.ArgumentParser(
        description='数据驱动3D匹配器（支持自定义帧数）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 处理58帧（默认）
  python run_matching.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01

  # 显式指定帧数
  python run_matching.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01 --max_frames 58

  # 多个文件夹
  python run_matching.py /path/folder1 /path/folder2 --max_frames 58
        """
    )
    parser.add_argument('folders', nargs='+',
                        help='要处理的文件夹路径（一个或多个）')
    parser.add_argument('--max_frames', type=int, default=58,
                        help='最大处理帧数（默认58，即处理全部帧）')
    parser.add_argument('--first_frame_candidates', nargs='+', type=int,
                        default=[1, 2, 3, 4, 5],
                        help='候选初始帧列表（默认：1 2 3 4 5）')
    parser.add_argument('--output', '-o',
                        help='输出基础目录（默认在输入文件夹下创建"数据驱动匹配"）')

    args = parser.parse_args()

    results = process_folders(
        args.folders,
        output_base_dir=args.output,
        max_frames=args.max_frames,
        first_frame_candidates=args.first_frame_candidates
    )

    success_count = sum(1 for r in results if r['success'])
    print(f"\n🎉 完成！成功: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
