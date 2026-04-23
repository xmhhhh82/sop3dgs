#!/usr/bin/env python3
"""
vggsfm_cli.py — 命令行版 VGGSfM 流水线适配脚本
原始 vggsfm.py 中的 main_folder / video_folder 硬编码路径已全部改为命令行参数。

用法:
  # 处理单个视频
  python vggsfm_cli.py --video /path/to/video.mp4 --scene_dir /path/to/scene --fps 2

  # 也可批量处理（兼容原始逻辑：扫描 video/ 子文件夹）
  python vggsfm_cli.py --main_dir /extdatashare/liuzy0/maljp0326 --fps 2

注意:
  本脚本需要在 VGGSfM 代码目录（VGGSFM_DIR）下执行，
  因为 `python demo.py SCENE_DIR=...` 依赖当前工作目录。

  pipeline.sh 会在调用前 cd 到 VGGSFM_DIR，
  或者你可以用 --vggsfm_dir 指定路径。
"""

import os
import subprocess
import sys
import argparse
import glob
from pathlib import Path


def extract_frames(video_path: str, images_dir: str, fps: int = 2) -> bool:
    """用 ffmpeg 从视频中抽帧，保存到 images_dir。"""
    os.makedirs(images_dir, exist_ok=True)

    # 检查是否已有图像
    existing = glob.glob(os.path.join(images_dir, "*.jpg")) + \
               glob.glob(os.path.join(images_dir, "*.png"))
    if existing:
        print(f"  ℹ  images/ 已有 {len(existing)} 张图像，跳过抽帧")
        return True

    cmd = (
        f'ffmpeg -i "{video_path}" '
        f'-vf fps={fps} '
        f'"{images_dir}/%04d.jpg" '
        f'-q:v 2 -loglevel warning'
    )
    print(f"  🎬 抽帧: {cmd}")
    ret = subprocess.run(cmd, shell=True)
    if ret.returncode != 0:
        print(f"  ❌ ffmpeg 失败 (returncode={ret.returncode})")
        return False

    n = len(glob.glob(os.path.join(images_dir, "*.jpg")))
    print(f"  ✅ 抽帧完成，共 {n} 帧 → {images_dir}")
    return True


def run_vggsfm(scene_dir: str, vggsfm_dir: str = None) -> bool:
    """
    执行 VGGSfM 稀疏重建。
    vggsfm_dir: VGGSfM 代码所在目录（需要包含 demo.py）。
                为 None 时使用当前工作目录。
    """
    sparse_path = os.path.join(scene_dir, "sparse")
    if os.path.exists(sparse_path):
        print(f"  ℹ  sparse/ 已存在，跳过 VGGSfM: {sparse_path}")
        return True

    command = f"python demo.py SCENE_DIR={scene_dir}"

    run_kwargs = {}
    if vggsfm_dir:
        run_kwargs["cwd"] = vggsfm_dir

    print(f"  🔭 运行 VGGSfM: {command}")
    if vggsfm_dir:
        print(f"     工作目录: {vggsfm_dir}")

    try:
        subprocess.run(command, shell=True, check=True, **run_kwargs)
        print(f"  ✅ VGGSfM 完成: {scene_dir}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ❌ VGGSfM 失败: {e}")
        return False


def process_single_video(video_path: str, scene_dir: str,
                         fps: int = 2, vggsfm_dir: str = None) -> bool:
    """处理单个视频：抽帧 + 稀疏重建。"""
    video_path = os.path.abspath(video_path)
    scene_name = Path(scene_dir).name

    print(f"\n{'='*60}")
    print(f"📽  视频: {os.path.basename(video_path)}")
    print(f"📁  场景目录: {scene_dir}")
    print(f"{'='*60}")

    if not os.path.isfile(video_path):
        print(f"  ❌ 视频文件不存在: {video_path}")
        return False

    os.makedirs(scene_dir, exist_ok=True)
    images_dir = os.path.join(scene_dir, "images")

    # Step 1: 抽帧
    ok = extract_frames(video_path, images_dir, fps=fps)
    if not ok:
        return False

    # Step 2: VGGSfM 重建
    ok = run_vggsfm(scene_dir, vggsfm_dir=vggsfm_dir)
    return ok


def process_main_dir(main_dir: str, fps: int = 2,
                     vggsfm_dir: str = None) -> list:
    """
    批量模式：扫描 main_dir/video/ 下的所有 .mp4 文件。
    与原始 vggsfm.py 行为一致。
    """
    video_folder = os.path.join(main_dir, "video")
    if not os.path.isdir(video_folder):
        print(f"❌ video/ 目录不存在: {video_folder}")
        sys.exit(1)

    mp4_files = sorted(glob.glob(os.path.join(video_folder, "*.mp4")))
    if not mp4_files:
        print(f"❌ 未找到 .mp4 文件: {video_folder}")
        sys.exit(1)

    print(f"找到 {len(mp4_files)} 个视频:")
    for f in mp4_files:
        print(f"  - {os.path.basename(f)}")

    results = []
    for video_path in mp4_files:
        scene_name = Path(video_path).stem
        # 检查是否已有 sparse 输出
        sparse_path = os.path.join(main_dir, scene_name, "sparse")
        if os.path.exists(sparse_path):
            print(f"\n⏭  跳过 {scene_name}（已有 sparse 输出）")
            results.append({"scene": scene_name, "success": True, "skipped": True})
            continue

        scene_dir = os.path.join(main_dir, scene_name)
        ok = process_single_video(video_path, scene_dir,
                                   fps=fps, vggsfm_dir=vggsfm_dir)
        results.append({"scene": scene_name, "success": ok, "skipped": False})

    return results


def main():
    parser = argparse.ArgumentParser(
        description="VGGSfM 命令行适配脚本：抽帧 + 稀疏重建",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 单个视频（pipeline.sh 调用方式）
  python vggsfm_cli.py --video /data/video/scene.mp4 --scene_dir /data/scene --fps 2

  # 批量模式（扫描 main_dir/video/ 下所有视频，与原始脚本行为相同）
  python vggsfm_cli.py --main_dir /extdatashare/liuzy0/maljp0326 --fps 2

  # 指定 VGGSfM 代码目录（包含 demo.py）
  python vggsfm_cli.py --video /data/video/scene.mp4 \\
      --scene_dir /data/scene \\
      --vggsfm_dir /code/vggsfm
        """
    )

    # 模式一：单视频
    parser.add_argument("--video",      type=str, default=None,
                        help="输入视频文件路径 (.mp4)")
    parser.add_argument("--scene_dir",  type=str, default=None,
                        help="场景输出目录（单视频模式必填）")

    # 模式二：批量
    parser.add_argument("--main_dir",   type=str, default=None,
                        help="批量模式：包含 video/ 子文件夹的根目录")

    # 共用参数
    parser.add_argument("--fps",        type=int, default=2,
                        help="抽帧帧率，默认 2")
    parser.add_argument("--vggsfm_dir", type=str, default=None,
                        help="VGGSfM 代码目录（包含 demo.py），默认使用当前目录")

    args = parser.parse_args()

    # 校验参数
    if args.video is None and args.main_dir is None:
        parser.error("请指定 --video 或 --main_dir 之一")
    if args.video is not None and args.scene_dir is None:
        # 自动推断 scene_dir：与视频同级目录下，以 scene_name 命名
        video_dir = os.path.dirname(os.path.abspath(args.video))
        if os.path.basename(video_dir) == "video":
            main_dir = os.path.dirname(video_dir)
        else:
            main_dir = video_dir
        scene_name = Path(args.video).stem
        args.scene_dir = os.path.join(main_dir, scene_name)
        print(f"  ℹ  自动推断 scene_dir: {args.scene_dir}")

    if args.video:
        # 单视频模式
        ok = process_single_video(
            video_path=args.video,
            scene_dir=args.scene_dir,
            fps=args.fps,
            vggsfm_dir=args.vggsfm_dir,
        )
        sys.exit(0 if ok else 1)
    else:
        # 批量模式
        results = process_main_dir(
            main_dir=args.main_dir,
            fps=args.fps,
            vggsfm_dir=args.vggsfm_dir,
        )
        success = sum(1 for r in results if r["success"])
        print(f"\n🎉 批量完成！成功: {success}/{len(results)}")
        print("\n汇总:")
        for r in results:
            mark = "✅" if r["success"] else "❌"
            tag  = " (已跳过)" if r.get("skipped") else ""
            print(f"  {mark} {r['scene']}{tag}")
        sys.exit(0 if success == len(results) else 1)


if __name__ == "__main__":
    main()
