#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
realbasicvsr_cli.py — RealBasicVSR 图像增强封装（Pipeline STEP 1.5）v2

★ 修复: 适配 inference_realbasicvsr.py 的实际接口（位置参数）

实际调用格式:
  python inference_realbasicvsr.py \\
      <config> <checkpoint> <input_dir> <output_dir> \\
      [--max_seq_len N] [--is_save_as_png 0/1] [--fps N]

用法:
  # 标准调用（需指定 config 和 checkpoint）
  python realbasicvsr_cli.py \\
      --scene_dir  /path/to/scene \\
      --config     /path/to/options/test/RealBasicVSR/test_RealBasicVSR_x4.yml \\
      --checkpoint /path/to/checkpoints/RealBasicVSR_x4.pth

  # 批量处理
  python realbasicvsr_cli.py /path/scene1 /path/scene2 \\
      --config     /path/to/config.yml \\
      --checkpoint /path/to/model.pth

  # 保存为PNG格式
  python realbasicvsr_cli.py --scene_dir /path/to/scene \\
      --config /path/cfg.yml --checkpoint /path/model.pth \\
      --is_save_as_png 1

目录结构:
  <scene_dir>/
  ├── images/              ← 原始抽帧图像（VGGSfM 使用）
  └── images_enhanced/     ← 增强后图像（本脚本输出，YOLO/3DGS 使用）
"""

import os
import sys
import glob
import argparse
import subprocess
from pathlib import Path


# ══════════════════════════════════════════════════════════════
#  常量 & 默认路径
# ══════════════════════════════════════════════════════════════

DEFAULT_RBVSR_DIR = "/extdatashare/liuzy0/code/RealBasicVSR"
ENHANCED_SUBDIR   = "images_enhanced"
IMG_EXTS          = (".jpg", ".jpeg", ".png", ".bmp")


# ══════════════════════════════════════════════════════════════
#  工具函数
# ══════════════════════════════════════════════════════════════

def find_inference_script(rbvsr_dir: str) -> str:
    """在 rbvsr_dir 内查找 inference_realbasicvsr.py。"""
    candidates = [
        os.path.join(rbvsr_dir, "inference_realbasicvsr.py"),
        os.path.join(rbvsr_dir, "inference.py"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(
        f"❌ 在 {rbvsr_dir} 找不到推理脚本，请用 --inference_script 手动指定"
    )


def find_config_auto(rbvsr_dir: str) -> str:
    """自动查找配置文件，按优先级排列。"""
    patterns = [
        "options/test/RealBasicVSR/*.yml",
        "options/test/RealBasicVSR/*.yaml",
        "options/test/**/*.yml",
        "configs/realbasicvsr*.py",
        "configs/*.py",
        "options/**/*.yml",
    ]
    for pattern in patterns:
        found = glob.glob(os.path.join(rbvsr_dir, pattern), recursive=True)
        if found:
            return found[0]
    return None


def find_checkpoint_auto(rbvsr_dir: str) -> str:
    """自动查找模型权重文件，按优先级排列。"""
    search_dirs = [
        os.path.join(rbvsr_dir, "checkpoints"),
        os.path.join(rbvsr_dir, "weights"),
        os.path.join(rbvsr_dir, "experiments"),
        rbvsr_dir,
    ]
    for d in search_dirs:
        if not os.path.isdir(d):
            continue
        for ext in ["*.pth", "*.pkl", "*.pt"]:
            found = glob.glob(os.path.join(d, ext))
            if found:
                # 优先含 RealBasicVSR 关键字的文件
                for f in found:
                    if "RealBasicVSR" in os.path.basename(f):
                        return f
                return found[0]
    # 递归兜底
    for ext in ["*.pth", "*.pkl"]:
        found = glob.glob(os.path.join(rbvsr_dir, "**", ext), recursive=True)
        if found:
            return found[0]
    return None


def collect_image_paths(directory: str) -> list:
    """收集目录下所有图像路径（排序）。"""
    paths = []
    for ext in IMG_EXTS:
        paths.extend(glob.glob(os.path.join(directory, f"*{ext}")))
        paths.extend(glob.glob(os.path.join(directory, f"*{ext.upper()}")))
    return sorted(set(paths))


def rename_to_match_original(enhanced_dir: str, original_dir: str) -> bool:
    """
    将增强目录中的文件重命名以匹配原始帧文件名（按排序顺序一一对应）。
    兼容扩展名变化（如 jpg→png）。
    """
    orig_files = collect_image_paths(original_dir)
    enh_files  = collect_image_paths(enhanced_dir)

    if not orig_files:
        return True

    if len(orig_files) != len(enh_files):
        print(f"  ⚠️  原始帧数({len(orig_files)}) ≠ 增强帧数({len(enh_files)})，"
              f"跳过重命名，请人工检查")
        return False

    renamed = 0
    for orig_path, enh_path in zip(orig_files, enh_files):
        orig_stem  = Path(orig_path).stem
        enh_ext    = Path(enh_path).suffix          # 保留增强后的扩展名
        target_name = orig_stem + enh_ext
        target_path = os.path.join(enhanced_dir, target_name)

        if os.path.abspath(enh_path) != os.path.abspath(target_path):
            os.rename(enh_path, target_path)
            renamed += 1

    if renamed > 0:
        print(f"  ✅ 重命名 {renamed} 个增强帧以对齐原始文件名")
    return True


# ══════════════════════════════════════════════════════════════
#  核心处理函数
# ══════════════════════════════════════════════════════════════

def enhance_single_scene(scene_dir: str,
                          rbvsr_dir: str = DEFAULT_RBVSR_DIR,
                          inference_script: str = None,
                          config: str = None,
                          checkpoint: str = None,
                          force_redo: bool = False,
                          max_seq_len: int = None,
                          is_save_as_png: int = None,
                          fps_out: int = None) -> bool:
    """
    对单个场景目录执行 RealBasicVSR 增强。

    调用格式:
        python inference_realbasicvsr.py
            config checkpoint input_dir output_dir
            [--max_seq_len N] [--is_save_as_png 0/1] [--fps N]
    """
    scene_dir    = Path(scene_dir)
    images_dir   = scene_dir / "images"
    enhanced_dir = scene_dir / ENHANCED_SUBDIR

    print(f"\n{'='*60}")
    print(f"🔆 RealBasicVSR 增强: {scene_dir.name}")
    print(f"{'='*60}")

    # ── 前置检查 ────────────────────────────────────────────────
    if not images_dir.exists():
        print(f"❌ 找不到原始图像目录: {images_dir}")
        return False

    orig_images = collect_image_paths(str(images_dir))
    if not orig_images:
        print(f"❌ images/ 目录为空: {images_dir}")
        return False

    # ── 检查是否已完成 ──────────────────────────────────────────
    if enhanced_dir.exists() and not force_redo:
        existing = collect_image_paths(str(enhanced_dir))
        if len(existing) >= len(orig_images):
            print(f"  ✅ images_enhanced/ 已存在 ({len(existing)} 张)，跳过")
            return True

    enhanced_dir.mkdir(parents=True, exist_ok=True)

    # ── 查找推理脚本 ────────────────────────────────────────────
    if inference_script is None:
        try:
            inference_script = find_inference_script(rbvsr_dir)
        except FileNotFoundError as e:
            print(e)
            return False

    # ── 查找 config ─────────────────────────────────────────────
    if config is None:
        config = find_config_auto(rbvsr_dir)
        if config is None:
            print(f"❌ 未找到配置文件，请用 --config 手动指定")
            print(f"   查找范围: {rbvsr_dir}/options/test/ 和 configs/")
            return False
        print(f"  🔍 自动找到 config   : {config}")
    else:
        if not os.path.isfile(config):
            print(f"❌ 配置文件不存在: {config}")
            return False

    # ── 查找 checkpoint ─────────────────────────────────────────
    if checkpoint is None:
        checkpoint = find_checkpoint_auto(rbvsr_dir)
        if checkpoint is None:
            print(f"❌ 未找到模型权重，请用 --checkpoint 手动指定")
            print(f"   查找范围: {rbvsr_dir}/checkpoints/ 和 weights/")
            return False
        print(f"  🔍 自动找到 checkpoint: {checkpoint}")
    else:
        if not os.path.isfile(checkpoint):
            print(f"❌ 模型权重不存在: {checkpoint}")
            return False

    print(f"   原始图像  : {images_dir}  ({len(orig_images)} 张)")
    print(f"   输出目录  : {enhanced_dir}")
    print(f"   推理脚本  : {inference_script}")
    print(f"   config    : {config}")
    print(f"   checkpoint: {checkpoint}")

    # ── ★ 构建命令（位置参数格式）──────────────────────────────
    # 正确格式: python inference_realbasicvsr.py
    #               <config> <checkpoint> <input_dir> <output_dir>
    #               [--max_seq_len N] [--is_save_as_png 0/1] [--fps N]
    cmd = [
        sys.executable,
        inference_script,
        config,               # 位置参数 1: config
        checkpoint,           # 位置参数 2: checkpoint
        str(images_dir),      # 位置参数 3: input_dir
        str(enhanced_dir),    # 位置参数 4: output_dir
    ]

    # 追加可选参数
    if max_seq_len is not None:
        cmd += ["--max_seq_len", str(max_seq_len)]
    if is_save_as_png is not None:
        cmd += ["--is_save_as_png", str(is_save_as_png)]
    if fps_out is not None:
        cmd += ["--fps", str(fps_out)]

    print(f"\n   执行命令: {' '.join(cmd)}")
    print(f"   工作目录: {rbvsr_dir}\n")

    try:
        subprocess.run(cmd, cwd=rbvsr_dir, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ RealBasicVSR 推理失败 (returncode={e.returncode})")
        return False

    # ── 验证输出 ────────────────────────────────────────────────
    enh_images = collect_image_paths(str(enhanced_dir))
    if not enh_images:
        print(f"\n❌ 增强后目录为空，请检查推理脚本输出: {enhanced_dir}")
        return False

    print(f"\n✅ 增强完成: {len(enh_images)} 张 → {enhanced_dir}")

    # ── 文件名对齐 ──────────────────────────────────────────────
    rename_to_match_original(str(enhanced_dir), str(images_dir))

    # ── 最终验证 ────────────────────────────────────────────────
    final_images = collect_image_paths(str(enhanced_dir))
    orig_stems   = {Path(p).stem for p in orig_images}
    enh_stems    = {Path(p).stem for p in final_images}
    missing      = orig_stems - enh_stems

    if missing:
        print(f"  ⚠️  {len(missing)} 个原始帧在增强目录中缺失:")
        for m in sorted(missing)[:10]:
            print(f"       {m}")
        return False

    print(f"  ✅ 验证通过：{len(final_images)} 张增强帧与原始帧完全对齐")
    return True


# ══════════════════════════════════════════════════════════════
#  批量处理
# ══════════════════════════════════════════════════════════════

def process_folders(scene_dirs, rbvsr_dir, inference_script,
                    config, checkpoint, force_redo,
                    max_seq_len, is_save_as_png, fps_out):
    results = []
    for sd in scene_dirs:
        ok = enhance_single_scene(
            sd,
            rbvsr_dir=rbvsr_dir,
            inference_script=inference_script,
            config=config,
            checkpoint=checkpoint,
            force_redo=force_redo,
            max_seq_len=max_seq_len,
            is_save_as_png=is_save_as_png,
            fps_out=fps_out,
        )
        results.append({"scene": sd, "success": ok})

    print(f"\n{'='*60}")
    print(f"📊 RealBasicVSR 批量增强汇总")
    print(f"{'='*60}")
    for r in results:
        name = Path(r["scene"]).name
        print(f"  {'✅' if r['success'] else '❌'}  {name}")
    ok_count = sum(1 for r in results if r["success"])
    print(f"\n  成功: {ok_count}/{len(results)}")
    return results


# ══════════════════════════════════════════════════════════════
#  命令行入口
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="RealBasicVSR 图像增强封装 v2（修复位置参数接口）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
inference_realbasicvsr.py 的实际接口（位置参数）:
  python inference_realbasicvsr.py \\
      <config> <checkpoint> <input_dir> <output_dir> \\
      [--max_seq_len N] [--is_save_as_png 0/1] [--fps N]

示例:
  # 指定 config 和 checkpoint（推荐）
  python realbasicvsr_cli.py \\
      --scene_dir  /path/to/scene \\
      --config     /extdatashare/.../options/test/RealBasicVSR/test_RealBasicVSR_x4.yml \\
      --checkpoint /extdatashare/.../checkpoints/RealBasicVSR_x4.pth

  # 自动查找 config/checkpoint（在 rbvsr_dir 下搜索）
  python realbasicvsr_cli.py \\
      --scene_dir /path/to/scene \\
      --rbvsr_dir /extdatashare/liuzy0/code/RealBasicVSR

  # 保存为 PNG
  python realbasicvsr_cli.py \\
      --scene_dir /path/to/scene --config /cfg.yml --checkpoint /model.pth \\
      --is_save_as_png 1

  # 批量处理
  python realbasicvsr_cli.py /path/s1 /path/s2 \\
      --config /cfg.yml --checkpoint /model.pth
        """
    )

    parser.add_argument("scene_dirs", nargs="*",
                        help="场景目录列表（与 --scene_dir 二选一）")
    parser.add_argument("--scene_dir", type=str, default=None)

    # RealBasicVSR 路径
    parser.add_argument("--rbvsr_dir", type=str, default=DEFAULT_RBVSR_DIR)
    parser.add_argument("--inference_script", type=str, default=None,
                        help="推理脚本路径（不填则自动查找）")

    # ★ 修复：config 和 checkpoint 作为独立参数
    parser.add_argument("--config", type=str, default=None,
                        help="配置文件路径（不填则在 rbvsr_dir 中自动查找）")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="模型权重路径（不填则在 rbvsr_dir/checkpoints/ 中自动查找）")

    # inference_realbasicvsr.py 的可选参数
    parser.add_argument("--max_seq_len", type=int, default=None,
                        help="最大序列长度")
    parser.add_argument("--is_save_as_png", type=int, default=None,
                        choices=[0, 1], help="是否保存为PNG: 0/1")
    parser.add_argument("--fps", type=int, default=None,
                        dest="fps_out", help="输出帧率")

    parser.add_argument("--force_redo", action="store_true",
                        help="强制重新生成（忽略已有 images_enhanced/）")

    args = parser.parse_args()

    all_dirs = list(args.scene_dirs)
    if args.scene_dir:
        all_dirs.append(args.scene_dir)
    if not all_dirs:
        parser.error("请提供至少一个场景目录")

    valid_dirs = [d for d in all_dirs if os.path.isdir(d)]
    invalid    = [d for d in all_dirs if not os.path.isdir(d)]
    for d in invalid:
        print(f"⚠️  目录不存在，跳过: {d}")

    if not valid_dirs:
        print("❌ 没有有效的场景目录")
        sys.exit(1)

    results = process_folders(
        valid_dirs,
        rbvsr_dir=args.rbvsr_dir,
        inference_script=args.inference_script,
        config=args.config,
        checkpoint=args.checkpoint,
        force_redo=args.force_redo,
        max_seq_len=args.max_seq_len,
        is_save_as_png=args.is_save_as_png,
        fps_out=args.fps_out,
    )

    sys.exit(1 if any(not r["success"] for r in results) else 0)


if __name__ == "__main__":
    main()
