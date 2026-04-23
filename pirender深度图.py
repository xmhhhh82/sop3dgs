#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量运行render_导出深度图.py的脚本
支持在命令行中直接指定多个文件夹
"""

import os
import sys
import subprocess
import argparse
import glob
import time
from datetime import datetime
from pathlib import Path

def setup_arg_parser():
    """设置命令行参数解析器"""
    parser = argparse.ArgumentParser(description='批量运行渲染脚本')
    
    # 添加位置参数（可以直接指定多个文件夹）
    parser.add_argument('folders', nargs='*', help='要处理的文件夹路径（可以指定多个）')
    
    # 其他选项
    parser.add_argument('--python_script', type=str,
                       default='/datashare/dir_liusha/2d-gaussian-splatting_gsid_seg/render_导出深度图.py',
                       help='Python脚本路径')
    parser.add_argument('--iteration', type=int, default=30000, help='迭代次数')
    parser.add_argument('--resolution', type=int, default=-1, help='分辨率')
    parser.add_argument('--skip_mesh', action='store_true', default=True, help='跳过mesh生成')
    parser.add_argument('--log_dir', type=str, default='render_logs', help='日志保存目录')
    parser.add_argument('--pattern', type=str, help='通配符模式，例如: "/path/to/*_frames/output_*"')
    
    return parser

def get_directories_to_process(args):
    """获取需要处理的目录列表"""
    dirs_to_process = []
    
    # 如果直接在命令行指定了文件夹
    if args.folders:
        for folder in args.folders:
            if os.path.isdir(folder):
                dirs_to_process.append(folder)
            else:
                print(f"警告: 目录不存在，跳过 - {folder}")
    
    # 如果使用了通配符模式
    elif args.pattern:
        matched_dirs = glob.glob(args.pattern)
        for d in matched_dirs:
            if os.path.isdir(d):
                dirs_to_process.append(d)
        if not matched_dirs:
            print(f"警告: 没有匹配到任何目录 - {args.pattern}")
    
    # 如果没有指定任何文件夹，提示用法
    else:
        print("错误: 请指定要处理的文件夹路径")
        print("\n用法示例:")
        print("  # 直接指定多个文件夹")
        print("  python batch_render_depth.py /path/to/folder1 /path/to/folder2 /path/to/folder3")
        print("\n  # 使用通配符")
        print("  python batch_render_depth.py --pattern '/path/to/*_frames/output_*'")
        print("\n  # 指定其他参数")
        print("  python batch_render_depth.py /path/to/folder1 --iteration 30000 --resolution -1")
        sys.exit(1)
    
    return dirs_to_process

def run_render_command(model_path, python_script, iteration, resolution, skip_mesh, log_file):
    """运行单个渲染命令"""
    cmd = [
        'python', python_script,
        '--model_path', model_path,
        '--iteration', str(iteration),
        '--resolution', str(resolution)
    ]
    
    if skip_mesh:
        cmd.append('--skip_mesh')
    
    # 打印命令
    print(f"运行命令: {' '.join(cmd)}")
    
    # 运行命令并记录输出
    start_time = time.time()
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"命令: {' '.join(cmd)}\n")
            f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("-" * 80 + "\n")
            f.flush()
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # 实时输出并写入文件
            for line in process.stdout:
                print(line, end='')
                f.write(line)
                f.flush()
            
            return_code = process.wait()
            
            f.write("-" * 80 + "\n")
            f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"返回代码: {return_code}\n")
        
        elapsed_time = time.time() - start_time
        return return_code == 0, elapsed_time
        
    except Exception as e:
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"错误: {str(e)}\n")
        return False, time.time() - start_time

def main():
    # 解析命令行参数
    parser = setup_arg_parser()
    args = parser.parse_args()
    
    # 获取要处理的目录列表
    dirs_to_process = get_directories_to_process(args)
    
    print(f"\n{'='*60}")
    print(f"批量渲染任务开始 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python脚本: {args.python_script}")
    print(f"迭代次数: {args.iteration}")
    print(f"分辨率: {args.resolution}")
    print(f"跳过mesh: {args.skip_mesh}")
    print(f"待处理目录数: {len(dirs_to_process)}")
    print('='*60)
    
    # 列出所有要处理的目录
    print("\n要处理的目录:")
    for i, d in enumerate(dirs_to_process, 1):
        print(f"  {i}. {d}")
    print()
    
    # 创建日志目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = f"{args.log_dir}_{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建汇总文件
    summary_file = os.path.join(log_dir, "summary.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"批量渲染任务汇总\n")
        f.write(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Python脚本: {args.python_script}\n")
        f.write(f"迭代次数: {args.iteration}\n")
        f.write(f"分辨率: {args.resolution}\n")
        f.write(f"跳过mesh: {args.skip_mesh}\n")
        f.write(f"待处理目录数: {len(dirs_to_process)}\n")
        f.write("-" * 80 + "\n\n")
        for i, d in enumerate(dirs_to_process, 1):
            f.write(f"{i}. {d}\n")
        f.write("\n" + "-" * 80 + "\n\n")
    
    # 处理每个目录
    results = []
    for i, model_path in enumerate(dirs_to_process, 1):
        print(f"\n{'='*60}")
        print(f"[{i}/{len(dirs_to_process)}] 处理目录: {model_path}")
        print('='*60)
        
        # 创建日志文件
        dir_name = os.path.basename(model_path)
        safe_name = "".join(c for c in dir_name if c.isalnum() or c in ('-', '_')).rstrip()
        log_file = os.path.join(log_dir, f"{i:02d}_{safe_name}_render.log")
        
        # 运行渲染命令
        success, elapsed_time = run_render_command(
            model_path=model_path,
            python_script=args.python_script,
            iteration=args.iteration,
            resolution=args.resolution,
            skip_mesh=args.skip_mesh,
            log_file=log_file
        )
        
        # 记录结果
        status = "✅ 成功" if success else "❌ 失败"
        results.append((model_path, success, elapsed_time))
        
        # 写入汇总
        with open(summary_file, 'a', encoding='utf-8') as f:
            f.write(f"[{i}/{len(dirs_to_process)}] {status} - {model_path}\n")
            f.write(f"    耗时: {elapsed_time:.2f}秒\n")
            f.write(f"    日志: {log_file}\n\n")
        
        # 检查输出目录
        train_dir = os.path.join(model_path, 'train')
        if success and os.path.exists(train_dir):
            depth_dir = os.path.join(train_dir, 'depth')
            if os.path.exists(depth_dir):
                depth_files = glob.glob(os.path.join(depth_dir, "*.png"))
                print(f"    深度图数量: {len(depth_files)}")
    
    # 输出最终汇总
    success_count = sum(1 for _, s, _ in results if s)
    failed_count = len(results) - success_count
    total_time = sum(t for _, _, t in results)
    
    print(f"\n{'='*60}")
    print(f"批量渲染任务完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"总计处理: {len(results)} 个目录")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"日志目录: {log_dir}")
    print(f"汇总文件: {summary_file}")
    print('='*60)
    
    # 更新汇总文件
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总计处理: {len(results)} 个目录\n")
        f.write(f"成功: {success_count}\n")
        f.write(f"失败: {failed_count}\n")
        f.write(f"总耗时: {total_time:.2f}秒\n")
    
    # 如果有失败的，返回非零退出码
    if failed_count > 0:
        sys.exit(1)

if __name__ == "__main__":
    main()


# 单个文件：python /datashare/dir_liusha/2d-gaussian-splatting_gsid_seg/pirender深度图.py /datashare/dir_liusha/xibeinonglin/样本数据/chang7_2-3_frames/output_chang7_2-3 --iteration 30000     --skip_mesh     --resolution 1
# 多个文件：python /datashare/dir_liusha/2d-gaussian-splatting_gsid_seg/pirender深度图.py /datashare/dir_liusha/xibeinonglin/样本数据/chang7_2-3_frames/output_chang7_2-3 /datashare/dir_liusha/xibeinonglin/样本数据/DK517M-2_frames/output_DK517M-2  --iteration 30000     --skip_mesh     --resolution 1