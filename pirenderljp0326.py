#!/usr/bin/env python3
"""
my_pirender.py —— 替代原版 pirender.py
修复内容：
  1. 查找路径支持 "数据驱动匹配/unified_masks" 和 "数据驱动匹配/id_mapping.json"
  2. render.py 调用时动态注入 --mask_dir 参数（无需修改原始 render.py）
  3. valid_frames 支持命令行传入，不再写死别人的路径

用法（单个文件夹）:
  python my_pirender.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01 \
      --gs_id --skip_train --skip_test --skip_mesh \
      --valid_frames /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01/valid_frames.txt

用法（多个文件夹）:
  python my_pirender.py /path/folder1 /path/folder2 \
      --gs_id --skip_train --skip_test --skip_mesh
"""

import os
import re
import subprocess
import sys
import argparse
from pathlib import Path

# ========== 原始 render.py 路径（不需要修改这个文件） ==========
RENDER_SCRIPT_PATH = "/datashare/dir_liusha/2d-gaussian-splatting_gsid_seg/render.py"


def find_required_paths(folder_path):
    """
    根据输入文件夹自动查找所需路径
    支持多种目录命名方式
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    print(f"\n{'='*60}")
    print(f"📁 自动查找路径: {folder_name}")
    print(f"{'='*60}")

    # ---------- 1. model_path ----------
    output_dirs = list(folder_path.glob("output_*"))
    model_path = None
    if output_dirs:
        model_path = str(output_dirs[0])
        print(f"✅ 找到 model_path: {model_path}")
    else:
        print(f"❌ 未找到 output_* 目录")
        return None

    # ---------- 2. source_path ----------
    source_path = str(folder_path)
    print(f"✅ 设置 source_path: {source_path}")

    # ---------- 3. id_mapping.json ----------
    id_mapping_candidates = [
        folder_path / "数据驱动匹配" / "id_mapping.json",          # ← 新增
        folder_path / "数据驱动匹配_同类匹配" / "id_mapping.json",
        folder_path / "id_mapping.json",
    ]
    id_mapping = None
    for c in id_mapping_candidates:
        if c.exists():
            id_mapping = str(c)
            print(f"✅ 找到 id_mapping: {id_mapping}")
            break
    if not id_mapping:
        print(f"⚠️  未找到 id_mapping.json")

    # ---------- 4. mask_dir (unified_masks) ----------
    mask_dir_candidates = [
        folder_path / "数据驱动匹配" / "unified_masks",            # ← 新增
        folder_path / "数据驱动匹配_同类匹配" / "unified_masks",
        folder_path / "unified_masks",
        folder_path / "masks_results" / "unified_masks",
    ]
    mask_dir = None
    for c in mask_dir_candidates:
        if c.exists() and c.is_dir():
            mask_dir = str(c)
            print(f"✅ 找到 mask_dir: {mask_dir}")
            break
    if not mask_dir:
        print(f"⚠️  未找到 unified_masks 目录")

    return {
        'model_path':  model_path,
        'source_path': source_path,
        'id_mapping':  id_mapping,
        'mask_dir':    mask_dir,
    }


def check_render_supports_mask_dir():
    """
    检查 render.py 是否已支持 --mask_dir 参数
    """
    try:
        with open(RENDER_SCRIPT_PATH, 'r') as f:
            content = f.read()
        return 'mask_dir' in content
    except Exception:
        return False


def process_single_folder(folder_path, base_args):
    """
    处理单个文件夹
    """
    print(f"\n{'='*80}")
    print(f"🚀 开始处理: {folder_path}")
    print(f"{'='*80}")

    paths = find_required_paths(folder_path)
    if not paths or not paths['model_path']:
        print(f"❌ 错误: 找不到 output_* 目录，跳过处理")
        return False

    # ---------- 构建命令行 ----------
    cmd = [sys.executable, RENDER_SCRIPT_PATH]

    # 添加 base_args 中的参数（排除 folders / valid_frames / id_mapping / mask_dir）
    skip_keys = {'folders', 'valid_frames', 'id_mapping', 'mask_dir',
                 'model_path', 'source_path'}
    for key, value in vars(base_args).items():
        if key in skip_keys:
            continue
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{key}")
        elif value is not None:
            cmd.extend([f"--{key}", str(value)])

    # 强制写入路径参数
    cmd.extend(["--model_path",  paths['model_path']])
    cmd.extend(["--source_path", paths['source_path']])

    # valid_frames：优先用命令行传入的，否则自动生成
    if base_args.valid_frames and os.path.exists(base_args.valid_frames):
        cmd.extend(["--valid_frames", base_args.valid_frames])
        print(f"✅ 使用 valid_frames: {base_args.valid_frames}")
    else:
        # 自动生成 valid_frames（包含该文件夹下所有图片帧）
        images_dir = Path(folder_path) / "images"
        if images_dir.exists():
            frames = sorted([
                f.stem for f in images_dir.glob("*.jpg")
            ] + [
                f.stem for f in images_dir.glob("*.png")
            ])
            # 过滤：只保留有 unified_mask 的帧，避免 render.py 因缺失 mask 而崩溃
            if paths['mask_dir']:
                mask_dir_path = Path(paths['mask_dir'])
                original_count = len(frames)
                filtered_frames = []
                for frame_name in frames:
                    m = re.search(r'(\d+)', frame_name)
                    if m:
                        fnum = int(m.group(1))
                        mask_file = mask_dir_path / f"unified_mask_{fnum:04d}.png"
                        if mask_file.exists():
                            filtered_frames.append(frame_name)
                        else:
                            print(f"  ⚠️  跳过帧 {frame_name}（无 unified_mask）")
                if filtered_frames:
                    frames = filtered_frames
                    if len(frames) < original_count:
                        print(f"  ℹ️  过滤后 {len(frames)}/{original_count} 帧有效")
            auto_vf = Path(folder_path) / "valid_frames.txt"
            with open(auto_vf, 'w') as f:
                f.write('\n'.join(frames))
            cmd.extend(["--valid_frames", str(auto_vf)])
            print(f"✅ 自动生成 valid_frames ({len(frames)} 帧): {auto_vf}")
        else:
            print(f"⚠️  未指定 valid_frames 且找不��� images/ 目录")

    # id_mapping
    if paths['id_mapping']:
        cmd.extend(["--id_mapping", paths['id_mapping']])

    # mask_dir：只有 render.py 支持时才传入
    if paths['mask_dir']:
        if check_render_supports_mask_dir():
            cmd.extend(["--mask_dir", paths['mask_dir']])
            print(f"✅ 传入 mask_dir: {paths['mask_dir']}")
        else:
            print(f"⚠️  render.py 不支持 --mask_dir 参数，跳过（不影响主流程）")

    print(f"\n📊 执行命令:")
    print(f"  {' '.join(cmd)}\n")

    try:
        result = subprocess.run(cmd, check=True)
        print(f"\n✅ 成功处理: {folder_path}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n❌ 处理失败: {folder_path}")
        print(f"   错误: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="批量处理多个文件夹（替代原版 pirender.py）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python my_pirender.py /extdatashare/liuzy0/maljp0326/2_1_2024_08_14_01 \\
      --gs_id --skip_train --skip_test --skip_mesh

  python my_pirender.py /path/folder1 /path/folder2 \\
      --gs_id --skip_train --skip_test --skip_mesh \\
      --valid_frames /path/to/valid_frames.txt
        """
    )

    # 文件夹列表
    parser.add_argument("folders", nargs='+',
                        help='要处理的文件夹路径（一个或多个）')

    # render.py 支持的参数
    parser.add_argument("--iteration",        default=-1,    type=int)
    parser.add_argument("--skip_train",       action="store_true")
    parser.add_argument("--skip_test",        action="store_true")
    parser.add_argument("--skip_mesh",        action="store_true")
    parser.add_argument("--quiet",            action="store_true")
    parser.add_argument("--render_path",      action="store_true")
    parser.add_argument("--gs_id",            action="store_true")
    parser.add_argument("--voxel_size",       default=-1.0,  type=float)
    parser.add_argument("--depth_trunc",      default=-1.0,  type=float)
    parser.add_argument("--sdf_trunc",        default=-1.0,  type=float)
    parser.add_argument("--num_cluster",      default=50,    type=int)
    parser.add_argument("--unbounded",        action="store_true")
    parser.add_argument("--mesh_res",         default=1024,  type=int)
    parser.add_argument("--min_valid_frames", default=3,     type=int)

    # 路径参数（由脚本自动查找，也可手动指定）
    parser.add_argument("--valid_frames", type=str, default=None,
                        help='valid_frames.txt 路径（不指定则自动生成）')
    parser.add_argument("--id_mapping",   type=str, default=None,
                        help='id_mapping.json 路径（不指定则自动查找）')
    parser.add_argument("--mask_dir",     type=str, default=None,
                        help='unified_masks 目录路径（不指定则自动查找）')

    args = parser.parse_args()

    print("\n" + "="*80)
    print("🚀 开始处理多个文件夹")
    print("="*80)
    print(f"要处理的文件夹: {args.folders}")

    results = []
    for folder_path in args.folders:
        success = process_single_folder(folder_path, args)
        results.append({'folder': folder_path, 'success': success})

    print("\n" + "="*80)
    print("📊 处理结果汇总")
    print("="*80)
    success_count = 0
    for r in results:
        name = Path(r['folder']).name
        if r['success']:
            print(f"✅ {name}: 成功")
            success_count += 1
        else:
            print(f"❌ {name}: 失败")

    print(f"\n🎉 完成！成功: {success_count}/{len(results)}")


if __name__ == "__main__":
    main()
