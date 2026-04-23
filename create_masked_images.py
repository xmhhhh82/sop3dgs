#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
create_masked_images.py — 生成YOLO掩码图像用于2DGS训练 (STEP 3.5)

★ v2 新增: --images_subdir 参数
  支持从 images_enhanced/ 等任意子目录读取源图像，
  默认仍为 images/ 保持向后兼容。

将源图像与 YOLO 分割结果合并，
生成只保留目标实例的 masked_images/ 目录。

2DGS 训练时通过 --images masked_images 参数使用该目录，
相机位姿仍来自 VGGSfM 的 sparse/0/（文件名完全对齐）。

用法（pipeline_enhanced.sh 自动调用）:
  # 标准模式（从 images/ 读取）
  python create_masked_images.py /path/to/scene

  # ★ 增强图像模式（从 images_enhanced/ 读取）
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced --target_classes 3

  # 保留所有类别 + 黑色背景
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced --keep_all_classes --bg_color 0 0 0
"""

import os
import cv2
import json
import argparse
import re
import numpy as np
from pathlib import Path
from tqdm import tqdm


# ══════════════════════════════════════════════════════════════
#  核心处理函数
# ══════════════════════════════════════════════════════════════

def create_masked_images_for_scene(
        scene_dir: str,
        target_classes: list = None,
        keep_all_classes: bool = False,
        bg_color: tuple = (255, 255, 255),
        output_dir_name: str = "masked_images",
        dilate_pixels: int = 3,
        force_redo: bool = False,
        images_subdir: str = "images",          # ★ 新增
) -> bool:
    """
    对单个场景目录生成掩码图像。

    Args:
        scene_dir        : 场景根目录（包含图像子目录和 masks_results/）
        target_classes   : 要保留的 YOLO class_id 列表，None 时保留所有
        keep_all_classes : True 则忽略 target_classes，保留所有检测到的实例
        bg_color         : 背景填充色，BGR 顺序（默认白色）
        output_dir_name  : 输出子目录名（默认 masked_images）
        dilate_pixels    : 掩码膨胀像素数，用于避免边缘硬切割（默认 3）
        force_redo       : True 时即使输出目录已存在也重新生成
        images_subdir    : ★ 源图像子目录名（默认 images，增强模式传 images_enhanced）

    Returns:
        成功返回 True，否则 False
    """
    scene_dir  = Path(scene_dir)
    mask_dir   = scene_dir / "masks_results" / "integer_masks"
    output_dir = scene_dir / output_dir_name

    # ── ★ 智能查找源图像目录 ────────────────────────────────────
    # 优先使用 images_subdir 指定的目录，回退到 images/
    preferred_images_dir = scene_dir / images_subdir
    fallback_images_dir  = scene_dir / "images"

    if preferred_images_dir.exists():
        images_dir = preferred_images_dir
    elif fallback_images_dir.exists():
        images_dir = fallback_images_dir
        if images_subdir != "images":
            print(f"  ⚠️  未找到 {images_subdir}/，回退到 images/")
    else:
        print(f"❌ 找不到源图像目录: {preferred_images_dir} 或 {fallback_images_dir}")
        return False

    # ── 前置检查 ────────────────────────────────────────────────
    if not mask_dir.exists():
        print(f"❌ 找不到掩码目录: {mask_dir}")
        return False

    # 收集源图像
    img_files = sorted(
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg")) +
        list(images_dir.glob("*.png"))
    )
    if not img_files:
        print(f"❌ 源图像目录为空: {images_dir}")
        return False

    # 检查是否已完成
    if output_dir.exists() and not force_redo:
        existing = list(output_dir.glob("*.jpg")) + list(output_dir.glob("*.png"))
        if len(existing) >= len(img_files):
            print(f"  ✅ {output_dir_name}/ 已存在 ({len(existing)} 张)，"
                  f"跳过 (--force_redo 可强制重生成)")
            return True

    output_dir.mkdir(parents=True, exist_ok=True)

    # 打印配置
    print(f"\n🖼️  生成掩码图像 [{scene_dir.name}]")
    print(f"   源图像目录: {images_dir}  ({len(img_files)} 张)")  # ★ 显示实际使用的目录
    print(f"   掩码来源  : {mask_dir}")
    print(f"   输出目录  : {output_dir}")
    if keep_all_classes:
        print(f"   保留类别  : 全部检测到的实例")
    else:
        print(f"   保留类别  : {target_classes}  (其余区域填充背景)")
    print(f"   背景颜色  : BGR{bg_color}")
    print(f"   掩码膨胀  : {dilate_pixels} 像素")

    dilate_kernel = (
        np.ones((dilate_pixels * 2 + 1, dilate_pixels * 2 + 1), np.uint8)
        if dilate_pixels > 0 else None
    )

    success_count = 0
    no_mask_count = 0

    for img_file in tqdm(img_files, desc="合成掩码图像"):
        # 解析帧号
        stem  = img_file.stem
        match = re.search(r'(\d+)', stem)
        if not match:
            print(f"  ⚠️  无法解析帧号: {img_file.name}，跳过")
            continue

        frame_num = int(match.group(1))

        # 读源图像（增强图像或原始图像）
        img = cv2.imread(str(img_file))
        if img is None:
            print(f"  ⚠️  无法读取: {img_file}")
            continue
        h, w = img.shape[:2]

        # ── 生成目标掩码 ────────────────────────────────────────
        mask_file       = mask_dir / f"mask_{frame_num:04d}.png"
        class_info_file = mask_dir / f"class_info_{frame_num:04d}.json"

        if not mask_file.exists():
            # 该帧无 YOLO 检测结果：整帧用背景色填充
            no_mask_count += 1
            masked_img = np.full((h, w, 3), bg_color, dtype=np.uint8)
        else:
            raw_mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if raw_mask is None:
                masked_img = np.full((h, w, 3), bg_color, dtype=np.uint8)
            else:
                if len(raw_mask.shape) == 3:
                    raw_mask = raw_mask[:, :, 0]

                # 与源图尺寸对齐（增强后分辨率可能变化）
                if raw_mask.shape != (h, w):
                    raw_mask = cv2.resize(raw_mask, (w, h),
                                          interpolation=cv2.INTER_NEAREST)

                # 读类别信息
                instance_to_class = {}
                if class_info_file.exists():
                    with open(class_info_file, 'r') as f:
                        cdata = json.load(f)
                    for inst in cdata.get('instances', []):
                        instance_to_class[inst['instance_id']] = inst['class_id']

                # 构建二值目标掩码
                if keep_all_classes or target_classes is None:
                    target_mask = (raw_mask > 0).astype(np.uint8)
                else:
                    target_mask = np.zeros((h, w), dtype=np.uint8)
                    for inst_id in np.unique(raw_mask):
                        if inst_id == 0:
                            continue
                        cls = instance_to_class.get(int(inst_id), -1)
                        if cls in target_classes:
                            target_mask[raw_mask == inst_id] = 1

                # 膨胀（避免边缘硬切割）
                if dilate_kernel is not None and target_mask.any():
                    target_mask = cv2.dilate(target_mask, dilate_kernel, iterations=1)

                # 合成：目标区域用源图，背景区域填充背景色
                masked_img = np.empty_like(img)
                bg = np.array(bg_color, dtype=np.uint8)
                for c in range(3):
                    masked_img[:, :, c] = np.where(
                        target_mask > 0, img[:, :, c], bg[c])

        # 保存（保持原始文件名和格式，保证与 sparse/0/ 中的名字对齐）
        out_path = output_dir / img_file.name
        cv2.imwrite(str(out_path), masked_img)
        success_count += 1

    print(f"\n✅ 完成: {success_count}/{len(img_files)} 张  "
          f"(无掩码帧: {no_mask_count} 张 → 已填充背景色)")
    print(f"   源图像  : {images_dir}")
    print(f"   输出路径: {output_dir}")
    return True


# ══════════════════════════════════════════════════════════════
#  验证工具：检查掩码图像与源图的对齐情况
# ══════════════════════════════════════════════════════════════

def verify_alignment(scene_dir: str,
                     output_dir_name: str = "masked_images",
                     images_subdir: str = "images",
                     check_n: int = 5):
    """简单验证：确认 masked_images/ 中的文件名和数量与源图完全一致。"""
    scene_dir   = Path(scene_dir)

    # ★ 使用实际的源图像目录
    preferred = scene_dir / images_subdir
    images_dir = preferred if preferred.exists() else scene_dir / "images"
    masked_dir  = scene_dir / output_dir_name

    orig_files   = set(f.name for f in images_dir.glob("*.*")
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    masked_files = set(f.name for f in masked_dir.glob("*.*")
                       if f.suffix.lower() in {'.jpg', '.jpeg', '.png'})

    missing   = orig_files - masked_files
    extra     = masked_files - orig_files

    print(f"\n🔍 对齐验证 [{scene_dir.name}]")
    print(f"   源图像目录: {images_dir}")
    print(f"   源图像数: {len(orig_files)}")
    print(f"   掩码图像数: {len(masked_files)}")

    if missing:
        print(f"   ⚠️  缺失文件 ({len(missing)}): "
              f"{sorted(missing)[:5]}{'...' if len(missing)>5 else ''}")
    if extra:
        print(f"   ⚠️  多余文件 ({len(extra)}): "
              f"{sorted(extra)[:5]}{'...' if len(extra)>5 else ''}")
    if not missing and not extra:
        print(f"   ✅ 文件名完全对齐")

    # 尺寸抽查
    sampled = sorted(orig_files)[:check_n]
    mismatch = []
    for fname in sampled:
        orig_img   = cv2.imread(str(images_dir / fname))
        masked_img = cv2.imread(str(masked_dir / fname))
        if orig_img is None or masked_img is None:
            mismatch.append(fname)
            continue
        if orig_img.shape != masked_img.shape:
            mismatch.append(
                f"{fname}(src={orig_img.shape}, masked={masked_img.shape})")

    if mismatch:
        print(f"   ⚠️  尺寸不一致: {mismatch}")
    else:
        print(f"   ✅ 抽查 {len(sampled)} 张，尺寸全部一致")

    return len(missing) == 0 and len(extra) == 0 and len(mismatch) == 0


# ══════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='生成 YOLO 掩码图像用于 2DGS 训练 (Pipeline STEP 3.5) v2',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
类别说明（玉米植株 YOLO 模型默认）:
  class 0 = 茎 (stem)
  class 1 = 根 (root)
  class 2 = 叶鞘 (leaf sheath)
  class 3 = 叶片 (leaf) ← 通常最感兴趣的目标类

示例:
  # ★ 增强图像模式（推荐）：从 images_enhanced/ 读取
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced

  # 只保留叶片（class 3），白色背景
  python create_masked_images.py /path/to/scene

  # 保留茎 + 叶片（class 0 和 3），黑色背景
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced \\
      --target_classes 0 3 --bg_color 0 0 0

  # 保留所有 YOLO 检测到的实例
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced --keep_all_classes

  # 批量处理多个场景
  python create_masked_images.py /path/f1 /path/f2 /path/f3 \\
      --images_subdir images_enhanced

  # 完成后验证文件名对齐
  python create_masked_images.py /path/to/scene \\
      --images_subdir images_enhanced --verify
        """
    )

    parser.add_argument('scene_dirs', nargs='+',
                        help='场景目录（一个或多个）')

    # ★ 新增: 源图像子目录
    parser.add_argument('--images_subdir', type=str, default='images',
                        help='源图像子目录名（默认: images，增强模式传 images_enhanced）')

    parser.add_argument('--target_classes', nargs='+', type=int, default=[3],
                        help='要保留的 YOLO class_id（默认: 3 叶片）')
    parser.add_argument('--keep_all_classes', action='store_true',
                        help='保留所有检测到的类别（忽略 --target_classes）')
    parser.add_argument('--bg_color', nargs=3, type=int,
                        default=[255, 255, 255], metavar=('B', 'G', 'R'),
                        help='背景颜色 BGR（默认: 255 255 255 白色）')
    parser.add_argument('--output_dir_name', default='masked_images',
                        help='输出子目录名（默认: masked_images）')
    parser.add_argument('--dilate_pixels', type=int, default=3,
                        help='掩码膨胀像素（默认: 3，软化边缘）')
    parser.add_argument('--force_redo', action='store_true',
                        help='即使输出目录已存在也重新生成')
    parser.add_argument('--verify', action='store_true',
                        help='完成后验证文件名对齐')

    args = parser.parse_args()
    bg   = tuple(args.bg_color)

    results = []
    for sd in args.scene_dirs:
        print(f"\n{'='*70}")
        print(f"📁  {Path(sd).name}  (源图像: {args.images_subdir}/)")
        print(f"{'='*70}")

        ok = create_masked_images_for_scene(
            scene_dir        = sd,
            target_classes   = args.target_classes,
            keep_all_classes = args.keep_all_classes,
            bg_color         = bg,
            output_dir_name  = args.output_dir_name,
            dilate_pixels    = args.dilate_pixels,
            force_redo       = args.force_redo,
            images_subdir    = args.images_subdir,    # ★ 传入
        )

        if ok and args.verify:
            verify_alignment(sd, args.output_dir_name,
                             images_subdir=args.images_subdir)

        results.append({'scene': sd, 'success': ok})

    print(f"\n{'='*70}")
    n_ok = sum(1 for r in results if r['success'])
    print(f"🎉 完成！成功: {n_ok}/{len(results)}")
    if n_ok < len(results):
        import sys; sys.exit(1)


if __name__ == "__main__":
    main()
