#!/usr/bin/env python3
"""
reason_cli.py — YOLO 实例分割推理（逐帧检测 + 诊断日志）

★ v2 新增: --images_subdir 参数
  支持从任意子目录（如 images_enhanced/）读取图像，
  默认仍为 images/ 保持向后兼容。

用法:
  # 标准模式（从 images/ 读取）
  python reason_cli.py --scene_dir /path/to/scene --model /path/to/best.pt

  # 增强图像模式（从 images_enhanced/ 读取）
  python reason_cli.py --scene_dir /path/to/scene --model /path/to/best.pt \\
      --images_subdir images_enhanced

  # 批量处理
  python reason_cli.py --root_dir /path/to/root --model /path/to/best.pt

输出结构:
  <scene_dir>/
  └── masks_results/
      ├── integer_masks/
      │   ├── mask_0001.png            # 灰度实例掩码（像素值 = 实例 ID）
      │   ├── class_info_0001.json     # {"instances": [...]}
      │   └── ...
      ├── visualization/               # --save_viz 时生成
      │   └── ...
      └── diagnostics/                 # 诊断信息
          ├── detection_log.csv        # 每帧检测汇总
          ├── instance_details.jsonl   # 每个实例的详细信息
          └── summary.json            # 全场景统计
"""

import os
import re
import sys
import glob
import json
import csv
import argparse
import cv2
import numpy as np
import torch
from pathlib import Path
from typing import Optional
from ultralytics import YOLO


# ─────────────────────────── 工具函数 ───────────────────────────

IMG_EXTS = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]


def extract_frame_number(filename: str):
    """从文件名中提取帧号整数，找不到则返回 None。"""
    stem = Path(filename).stem
    match = re.search(r'(\d+)', stem)
    return int(match.group(1)) if match else None


def collect_image_paths(folder: str, images_subdir: str = "images") -> list:
    """
    收集图像路径（排序）。

    优先级:
      1. folder/images_subdir/  （--images_subdir 指定的子目录）
      2. folder/images/          （兜底，向后兼容）
      3. folder/                 （直接在根目录下查找）
    """
    # ★ 优先使用指定的子目录
    preferred = os.path.join(folder, images_subdir)
    if os.path.isdir(preferred):
        search_dir = preferred
    else:
        # 回退到 images/
        fallback = os.path.join(folder, "images")
        if os.path.isdir(fallback):
            search_dir = fallback
            if images_subdir != "images":
                print(f"  ⚠️  未找到 {images_subdir}/，回退到 images/")
        elif os.path.isdir(folder):
            search_dir = folder
        else:
            return []

    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(search_dir, ext)))
    return sorted(paths)


# ─────────────────────────── 核心推理 ───────────────────────────

def process_single_image(model, img_path: str,
                          output_mask_dir: str,
                          output_viz_dir: Optional[str],
                          diag_writer=None,
                          diag_detail_f=None,
                          conf: float = 0.25) -> bool:
    """
    对单张图像执行 YOLO 推理，保存掩码 PNG 和 class_info JSON。
    同时输出诊断信息用于排查点云混杂问题。
    """
    filename  = os.path.basename(img_path)
    frame_num = extract_frame_number(filename)

    if frame_num is None:
        print(f"    ⚠️  无法提取帧号，跳过: {filename}")
        return False

    results = model(img_path, conf=conf, verbose=False)
    result  = results[0]
    orig_h, orig_w = result.orig_shape[:2]

    instance_mask = np.zeros((orig_h, orig_w), dtype=np.uint16)
    class_info    = {"instances": []}

    # ── 诊断数据收集 ──
    n_total_detections = 0
    n_by_class = {}          # class_id -> count
    overlap_pixels = 0       # 被多个实例覆盖的像素数（掩码重叠）
    instance_details = []    # 每个实例的详细信息

    if result.masks is not None and len(result.masks) > 0:
        masks_tensor = result.masks.data
        classes      = result.boxes.cls.cpu().numpy().astype(int)
        confs        = result.boxes.conf.cpu().numpy()
        boxes        = result.boxes.xyxy.cpu().numpy()

        n_total_detections = len(classes)

        # 用于检测掩码重叠
        coverage_count = np.zeros((orig_h, orig_w), dtype=np.int32)

        for inst_idx, (mask_tensor, cls_id) in enumerate(
                zip(masks_tensor, classes), start=1):
            mask_np      = mask_tensor.cpu().numpy()
            mask_resized = cv2.resize(mask_np, (orig_w, orig_h),
                                      interpolation=cv2.INTER_NEAREST)
            binary_mask  = (mask_resized > 0.5)

            # 统计掩码重叠
            coverage_count[binary_mask] += 1

            instance_mask[binary_mask] = inst_idx
            class_info["instances"].append({
                "instance_id": inst_idx,
                "class_id":    int(cls_id)
            })

            # 按类别计数
            n_by_class[int(cls_id)] = n_by_class.get(int(cls_id), 0) + 1

            # ── 诊断：每个实例的详细信息 ──
            inst_area = int(np.sum(binary_mask))
            box = boxes[inst_idx - 1]
            box_w = float(box[2] - box[0])
            box_h = float(box[3] - box[1])
            inst_conf = float(confs[inst_idx - 1])

            # 掩码质量：掩码面积 / bbox 面积
            box_area = box_w * box_h
            mask_fill_ratio = inst_area / box_area if box_area > 0 else 0

            # 掩码紧凑度：4*pi*area / perimeter^2，越接近1越规则
            contours, _ = cv2.findContours(binary_mask.astype(np.uint8),
                                            cv2.RETR_EXTERNAL,
                                            cv2.CHAIN_APPROX_SIMPLE)
            perimeter = sum(cv2.arcLength(c, True) for c in contours)
            compactness = (4 * np.pi * inst_area) / (perimeter**2) if perimeter > 0 else 0

            detail = {
                "frame": frame_num,
                "instance_id": inst_idx,
                "class_id": int(cls_id),
                "confidence": round(inst_conf, 4),
                "area_pixels": inst_area,
                "bbox": [round(float(x), 1) for x in box],
                "bbox_wh": [round(box_w, 1), round(box_h, 1)],
                "mask_fill_ratio": round(mask_fill_ratio, 3),
                "compactness": round(compactness, 3),
                "n_contours": len(contours),
            }
            instance_details.append(detail)

            # 打印目标类（class=3）的详细信息
            if cls_id == 3:
                print(f"      叶#{inst_idx}: conf={inst_conf:.3f} "
                      f"area={inst_area:5d}px "
                      f"bbox=[{box_w:.0f}x{box_h:.0f}] "
                      f"fill={mask_fill_ratio:.2f} "
                      f"compact={compactness:.3f} "
                      f"contours={len(contours)}")

        # 计算重叠像素
        overlap_pixels = int(np.sum(coverage_count > 1))

    else:
        print(f"    ℹ️  帧 {frame_num:04d} 无检测结果（{filename}）")

    # ── 保存掩码 PNG ──
    mask_path = os.path.join(output_mask_dir, f"mask_{frame_num:04d}.png")
    cv2.imwrite(mask_path, instance_mask)

    # ── 保存 class_info JSON ──
    json_path = os.path.join(output_mask_dir, f"class_info_{frame_num:04d}.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(class_info, f, indent=2, ensure_ascii=False)

    # ── 打印帧级汇总 ──
    n_target = n_by_class.get(3, 0)  # class_id=3 是目标类（叶子）
    n_inst = len(class_info["instances"])
    overlap_pct = (overlap_pixels / (orig_h * orig_w) * 100) if (orig_h * orig_w) > 0 else 0

    class_str = " ".join([f"C{k}:{v}" for k, v in sorted(n_by_class.items())])
    overlap_warn = f" ⚠️ 重叠{overlap_pixels}px({overlap_pct:.2f}%)" if overlap_pixels > 100 else ""
    print(f"    ✅ 帧 {frame_num:04d} | 总:{n_inst:3d} 叶:{n_target:3d} | "
          f"{class_str}{overlap_warn}")

    # ── 诊断日志写入 ──
    if diag_writer:
        diag_writer.writerow({
            "frame": frame_num,
            "filename": filename,
            "n_total": n_inst,
            "n_leaf_class3": n_target,
            "n_class0": n_by_class.get(0, 0),
            "n_class1": n_by_class.get(1, 0),
            "n_class2": n_by_class.get(2, 0),
            "overlap_pixels": overlap_pixels,
            "overlap_pct": round(overlap_pct, 3),
        })

    if diag_detail_f and instance_details:
        for d in instance_details:
            diag_detail_f.write(json.dumps(d, ensure_ascii=False) + "\n")

    # ── 可视化 ──
    if output_viz_dir:
        viz_img  = result.plot()
        viz_path = os.path.join(output_viz_dir, f"viz_{frame_num:04d}_{filename}")
        cv2.imwrite(viz_path, viz_img)

    return True


def process_single_folder(model, scene_dir: str,
                           conf: float = 0.25,
                           save_viz: bool = True,
                           images_subdir: str = "images") -> dict:
    """
    处理单个场景文件夹，逐帧独立检测。

    ★ v2: 新增 images_subdir 参数，支持从增强图像目录检测。
    """
    folder_name = os.path.basename(scene_dir)
    # ★ 传入 images_subdir
    img_paths   = collect_image_paths(scene_dir, images_subdir=images_subdir)

    if not img_paths:
        print(f"  ⚠️  [{folder_name}] 未找到图像文件"
              f"（{images_subdir}/ 或 images/ 或根目录），跳过")
        return {"folder": folder_name, "total": 0, "success": 0, "skipped": 0}

    # 输出目录（始终写到 scene_dir/masks_results/，与图像子目录无关）
    output_mask_dir = os.path.join(scene_dir, "masks_results", "integer_masks")
    output_viz_dir  = os.path.join(scene_dir, "masks_results", "visualization") \
                      if save_viz else None
    diag_dir        = os.path.join(scene_dir, "masks_results", "diagnostics")

    # 检查是否已完成
    existing_masks = glob.glob(os.path.join(output_mask_dir, "mask_*.png"))
    if len(existing_masks) >= len(img_paths):
        print(f"  ℹ️  [{folder_name}] 已有 {len(existing_masks)} 个掩码，跳过推理")
        return {"folder": folder_name, "total": len(img_paths),
                "success": len(existing_masks), "skipped": 0, "cached": True}

    os.makedirs(output_mask_dir, exist_ok=True)
    if output_viz_dir:
        os.makedirs(output_viz_dir, exist_ok=True)
    os.makedirs(diag_dir, exist_ok=True)

    # ★ 打印使用的图像目录
    used_dir = os.path.join(scene_dir, images_subdir)
    print(f"\n  📂 [{folder_name}]  图像: {len(img_paths)} 张")
    print(f"     图像来源: {used_dir}")
    print(f"     掩码输出: {output_mask_dir}")
    print(f"     诊断输出: {diag_dir}")

    # ── 初始化诊断日志 ──
    csv_path = os.path.join(diag_dir, "detection_log.csv")
    csv_f = open(csv_path, "w", newline="", encoding="utf-8")
    csv_fields = ["frame", "filename", "n_total", "n_leaf_class3",
                  "n_class0", "n_class1", "n_class2",
                  "overlap_pixels", "overlap_pct"]
    csv_writer = csv.DictWriter(csv_f, fieldnames=csv_fields)
    csv_writer.writeheader()

    detail_path = os.path.join(diag_dir, "instance_details.jsonl")
    detail_f = open(detail_path, "w", encoding="utf-8")

    success, skipped = 0, 0
    all_leaf_counts = []  # 每帧叶子数，用于统计波动

    for img_path in img_paths:
        ok = process_single_image(model, img_path, output_mask_dir,
                                   output_viz_dir,
                                   diag_writer=csv_writer,
                                   diag_detail_f=detail_f,
                                   conf=conf)
        if ok:
            success += 1
            # 统计叶子数
            frame_num = extract_frame_number(os.path.basename(img_path))
            json_path = os.path.join(output_mask_dir,
                                     f"class_info_{frame_num:04d}.json")
            if os.path.exists(json_path):
                with open(json_path, "r") as f:
                    info = json.load(f)
                n_leaf = sum(1 for x in info["instances"] if x["class_id"] == 3)
                all_leaf_counts.append((frame_num, n_leaf))
        else:
            skipped += 1

    csv_f.close()
    detail_f.close()

    # ── 全场景诊断汇总 ──
    if all_leaf_counts:
        leaf_nums = [c for _, c in all_leaf_counts]
        avg_leaf = np.mean(leaf_nums)
        std_leaf = np.std(leaf_nums)
        min_leaf = min(leaf_nums)
        max_leaf = max(leaf_nums)

        # 找出叶子数量异常的帧（偏离均值超过2倍标准差）
        outlier_frames = []
        for fn, n in all_leaf_counts:
            if abs(n - avg_leaf) > 2 * std_leaf and std_leaf > 0:
                outlier_frames.append({"frame": fn, "n_leaf": n,
                                       "deviation": round((n - avg_leaf) / std_leaf, 2)})

        summary = {
            "scene": folder_name,
            "total_frames": len(img_paths),
            "success_frames": success,
            "images_subdir": images_subdir,   # ★ 记录使用的图像子目录
            "leaf_stats": {
                "mean": round(avg_leaf, 2),
                "std": round(std_leaf, 2),
                "min": min_leaf,
                "max": max_leaf,
                "min_frame": all_leaf_counts[leaf_nums.index(min_leaf)][0],
                "max_frame": all_leaf_counts[leaf_nums.index(max_leaf)][0],
            },
            "outlier_frames": outlier_frames,
            "conf_threshold": conf,
        }

        summary_path = os.path.join(diag_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        # 打印诊断汇总
        print(f"\n  {'─'*50}")
        print(f"  📊 诊断汇总 [{folder_name}]")
        print(f"     图像来源: {images_subdir}/")
        print(f"     叶子(C3)数量: 均值={avg_leaf:.1f} 标准差={std_leaf:.1f} "
              f"范围=[{min_leaf}, {max_leaf}]")
        if outlier_frames:
            print(f"     ⚠️  异常帧（叶子数量偏离>2σ）:")
            for o in outlier_frames:
                print(f"        帧{o['frame']:04d}: {o['n_leaf']}片叶子 "
                      f"(偏差={o['deviation']:.1f}σ)")
        else:
            print(f"     ✅ 各帧叶子数量稳定，无异常波动")
        print(f"     诊断文件: {diag_dir}")
        print(f"  {'─'*50}")

    return {
        "folder":   folder_name,
        "total":    len(img_paths),
        "success":  success,
        "skipped":  skipped,
        "mask_dir": output_mask_dir,
    }


# ─────────────────────────── 主函数 ─────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="YOLO 实例分割推理（逐帧检测 + 诊断日志）v2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单场景（默认从 images/ 读取）
  python reason_cli.py \\
      --scene_dir /path/to/scene \\
      --model /path/to/best.pt

  # ★ 使用增强图像（从 images_enhanced/ 读取）
  python reason_cli.py \\
      --scene_dir /path/to/scene \\
      --model /path/to/best.pt \\
      --images_subdir images_enhanced

  # 批量处理
  python reason_cli.py \\
      --root_dir /extdatashare/liuzy0/maljp0326 \\
      --model /path/to/best.pt \\
      --images_subdir images_enhanced

  # 自定义置信度
  python reason_cli.py \\
      --scene_dir /path/to/scene \\
      --model /path/to/best.pt --conf 0.3
        """
    )

    # 模式参数（二选一）
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scene_dir", type=str,
                       help="单场景目录（单场景模式）")
    group.add_argument("--root_dir",  type=str,
                       help="批量模式：扫描该目录下所有包含图像子目录的子文件夹")

    # 必填
    parser.add_argument("--model", type=str, required=True,
                        help="YOLO 模型权重路径（best.pt）")

    # ★ 新增: 图像子目录参数
    parser.add_argument("--images_subdir", type=str, default="images",
                        help="图像子目录名（默认: images，增强模式传 images_enhanced）")

    # 可选
    parser.add_argument("--conf",     type=float, default=0.25,
                        help="置信度阈值（默认 0.25）")
    parser.add_argument("--save_viz", action="store_true", default=True,
                        help="是否保存彩色可视化结果（默认开启）")
    parser.add_argument("--no_viz",   action="store_true",
                        help="关闭可视化保存（--save_viz 的反义）")

    args = parser.parse_args()

    save_viz = args.save_viz and not args.no_viz

    # ── 检查模型文件 ──────────────────────────────────────────
    if not os.path.isfile(args.model):
        print(f"❌ 模型文件不存在: {args.model}")
        sys.exit(1)

    # ★ 打印使用的图像子目录
    print(f"🖼️  图像来源子目录: {args.images_subdir}/")

    # ── 加载模型 ──────────────────────────────────────────────
    print(f"🖥  设备: {'GPU 0' if torch.cuda.is_available() else 'CPU'}")
    print(f"📦 加载模型: {args.model}")
    model = YOLO(args.model)

    # ── 单场景模式 ────────────────────────────────────────────
    if args.scene_dir:
        if not os.path.isdir(args.scene_dir):
            print(f"❌ 场景目录不存在: {args.scene_dir}")
            sys.exit(1)

        stats = process_single_folder(model, args.scene_dir,
                                       conf=args.conf, save_viz=save_viz,
                                       images_subdir=args.images_subdir)
        print(f"\n✅ 完成: {stats['success']}/{stats['total']} 帧")
        sys.exit(0)

    # ── 批量模式 ──────────────────────────────────────────────
    if not os.path.isdir(args.root_dir):
        print(f"❌ 根目录不存在: {args.root_dir}")
        sys.exit(1)

    SKIP_DIRS = {"masks_results", "inference_results", "output", "runs", "video"}
    sub_folders = []
    for entry in sorted(os.scandir(args.root_dir), key=lambda e: e.name):
        if not entry.is_dir() or entry.name in SKIP_DIRS:
            continue
        # ★ 检查 images_subdir 目录或回退到 images/
        target_subdir = os.path.join(entry.path, args.images_subdir)
        fallback_dir  = os.path.join(entry.path, "images")
        has_images = (
            os.path.isdir(target_subdir) and
            any(glob.glob(os.path.join(target_subdir, f"*{ext}"))
                for ext in [".jpg", ".jpeg", ".png", ".bmp"])
        ) or (
            os.path.isdir(fallback_dir) and
            any(glob.glob(os.path.join(fallback_dir, f"*{ext}"))
                for ext in [".jpg", ".jpeg", ".png", ".bmp"])
        )
        if has_images:
            sub_folders.append(entry.path)
        else:
            print(f"  ⚠️  跳过（无图像）: {entry.name}")

    if not sub_folders:
        print(f"❌ 在 {args.root_dir} 下未找到含图像的子文件夹")
        sys.exit(1)

    print(f"\n📋 待处理场景: {len(sub_folders)} 个")

    stats_list = []
    for idx, folder_path in enumerate(sub_folders, 1):
        print(f"\n{'='*60}")
        print(f"[{idx}/{len(sub_folders)}]  {os.path.basename(folder_path)}")
        print(f"{'='*60}")
        stats = process_single_folder(model, folder_path,
                                       conf=args.conf, save_viz=save_viz,
                                       images_subdir=args.images_subdir)
        stats_list.append(stats)

    # 汇总
    print("\n" + "="*60)
    print("📊 处理汇总")
    print("="*60)
    total_imgs    = sum(s["total"]   for s in stats_list)
    total_success = sum(s["success"] for s in stats_list)
    total_skipped = sum(s["skipped"] for s in stats_list)

    for s in stats_list:
        mark = "✅" if s["skipped"] == 0 else "⚠️ "
        cached_tag = " (已缓存)" if s.get("cached") else ""
        print(f"  {mark} {s['folder']:<40} "
              f"成功: {s['success']:4d}/{s['total']:4d}  "
              f"跳过: {s['skipped']:3d}{cached_tag}")

    print("-"*60)
    print(f"  文件夹: {len(stats_list)}  |  图像: {total_imgs}  |  "
          f"成功: {total_success}  |  跳过: {total_skipped}")

    sys.exit(0)


if __name__ == "__main__":
    main()
