#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量提取参照物点云指标脚本
使用双重RANSAC矩形拟合方法提取所有参照物指标
"""

import os
import sys
import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
import argparse
import glob
from datetime import datetime
import traceback


# ============================
# 双重RANSAC矩形拟合函数（从原代码复制）
# ============================

def fit_rectangle_ransac(points_3d, max_iterations=500, distance_threshold=0.01, verbose=False):
    """
    使用双重RANSAC算法拟合长方形（对离群点极其鲁棒）
    输入：
        points_3d - 参照物点云，n×3的数组
        max_iterations - RANSAC最大迭代次数（2D矩形拟合）
        distance_threshold - 点到矩形的距离阈值（用于判断内点）
        verbose - 是否打印详细信息
    输出：
        包含拟合结果的字典
    """
    if len(points_3d) < 10:
        if verbose:
            print("⚠️ 参照物点数太少，无法拟合")
        return None
    
    try:
        # 转换为open3d点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # ========== 第1步：使用RANSAC拟合主平面 ==========
        # RANSAC平面拟合
        plane_model, plane_inliers = pcd.segment_plane(distance_threshold=distance_threshold * 0.5,
                                                        ransac_n=3,
                                                        num_iterations=200)
        
        # 提取平面上的点
        plane_points = np.asarray(pcd.select_by_index(plane_inliers).points)
        
        if len(plane_points) < 10:
            if verbose:
                print("  错误：平面上点数太少")
            return None
        
        # ========== 第2步：将3D点投影到平面（转换为2D问题） ==========
        # 提取平面参数: ax + by + cz + d = 0
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # 构建平面上的局部坐标系
        if abs(normal[2]) < 0.9:
            if c != 0:
                u = np.array([1, 0, -a/c])
            else:
                u = np.array([1, 0, 0])
            u = u / np.linalg.norm(u)
        else:
            u = np.array([0, 1, 0])
        
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        u = np.cross(v, normal)
        u = u / np.linalg.norm(u)
        
        # 计算平面中心
        plane_center = np.mean(plane_points, axis=0)
        
        # 将所有点投影到2D平面
        points_2d = np.zeros((len(plane_points), 2))
        for i, p in enumerate(plane_points):
            points_2d[i, 0] = np.dot(p - plane_center, u)
            points_2d[i, 1] = np.dot(p - plane_center, v)
        
        # ========== 第3步：RANSAC拟合矩形（在2D空间中） ==========
        best_inliers = None
        best_rectangle = None
        best_score = 0
        best_corners_2d = None
        
        for iteration in range(max_iterations):
            # 随机采样4个点
            sample_indices = np.random.choice(len(points_2d), 4, replace=False)
            sample = points_2d[sample_indices]
            
            try:
                x_min = np.min(sample[:, 0])
                x_max = np.max(sample[:, 0])
                y_min = np.min(sample[:, 1])
                y_max = np.max(sample[:, 1])
                
                rect_width = x_max - x_min
                rect_height = y_max - y_min
                aspect_ratio = max(rect_width, rect_height) / (min(rect_width, rect_height) + 1e-6)
                
                if aspect_ratio > 5:
                    continue
                
                # 计算内点
                dx = np.maximum(x_min - points_2d[:, 0], 0) + np.maximum(points_2d[:, 0] - x_max, 0)
                dy = np.maximum(y_min - points_2d[:, 1], 0) + np.maximum(points_2d[:, 1] - y_max, 0)
                distances = np.sqrt(dx**2 + dy**2)
                
                inlier_mask = distances <= distance_threshold
                inlier_count = np.sum(inlier_mask)
                
                if inlier_count > best_score:
                    best_score = inlier_count
                    best_inliers = inlier_mask
                    best_rectangle = (x_min, x_max, y_min, y_max)
                    best_corners_2d = np.array([
                        [x_min, y_min],
                        [x_max, y_min],
                        [x_max, y_max],
                        [x_min, y_max]
                    ])
                        
            except Exception:
                continue
        
        if best_rectangle is None:
            if verbose:
                print("  错误：未找到有效的矩形")
            return None
        
        # ========== 第4步：使用内点重新精炼矩形 ==========
        inlier_points = points_2d[best_inliers]
        inlier_count = len(inlier_points)
        
        # 使用内点重新计算精确的矩形边界
        x_min = np.min(inlier_points[:, 0])
        x_max = np.max(inlier_points[:, 0])
        y_min = np.min(inlier_points[:, 1])
        y_max = np.max(inlier_points[:, 1])
        
        # 计算长度和宽度
        length = max(x_max - x_min, y_max - y_min)
        width = min(x_max - x_min, y_max - y_min)
        
        # 确保length >= width，并调整角点顺序
        if (x_max - x_min) < (y_max - y_min):
            corners_2d = np.array([
                [x_min, y_min],
                [x_min, y_max],
                [x_max, y_max],
                [x_max, y_min]
            ])
        else:
            corners_2d = np.array([
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max]
            ])
        
        # ========== 第5步：将2D角点转换回3D ==========
        corners_3d = []
        for corner_2d in corners_2d:
            point_3d = plane_center + corner_2d[0] * u + corner_2d[1] * v
            corners_3d.append(point_3d)
        
        corners_3d = np.array(corners_3d)
        
        # 计算矩形的几何特性
        edge1 = corners_3d[1] - corners_3d[0]
        edge2 = corners_3d[3] - corners_3d[0]
        rectangle_area = np.linalg.norm(edge1) * np.linalg.norm(edge2)
        
        # 计算对角线长度
        diagonal = np.linalg.norm(corners_3d[2] - corners_3d[0])
        
        # 计算周长
        perimeter = 2 * (length + width)
        
        return {
            'length': length,  # 矩形长度（较长边）
            'width': width,    # 矩形宽度（较短边）
            'aspect_ratio': length / width if width > 0 else 0,  # 长宽比
            'rectangle_area': rectangle_area,  # 矩形面积
            'diagonal_length': diagonal,  # 对角线长度
            'perimeter': perimeter,  # 周长
            'plane_normal': normal,  # 平面法向量
            'centroid': plane_center,  # 平面中心点
            'corners_3d': corners_3d,  # 3D角点
            'point_count': len(points_3d),  # 原始点数
            'plane_inlier_count': len(plane_points),  # 平面内点数
            'rectangle_inlier_count': inlier_count,  # 矩形内点数
            'plane_inlier_ratio': len(plane_points) / len(points_3d),  # 平面内点比例
            'rectangle_inlier_ratio': inlier_count / len(plane_points) if len(plane_points) > 0 else 0,  # 矩形内点比例
            'total_inlier_ratio': inlier_count / len(points_3d)  # 总内点比例
        }
        
    except Exception as e:
        if verbose:
            print(f"双重RANSAC矩形拟合过程中出错: {e}")
        return None


def extract_reference_metrics(ply_file_path, ransac_iterations=500, ransac_threshold=0.01, verbose=True):
    """
    从单个PLY文件中提取参照物指标
    
    参数:
        ply_file_path: PLY文件路径
        ransac_iterations: RANSAC迭代次数
        ransac_threshold: RANSAC距离阈值
        verbose: 是否打印详细信息
    
    返回:
        包含所有指标的字典
    """
    if verbose:
        print(f"\n处理: {os.path.basename(ply_file_path)}")
    
    # 读取点云
    try:
        pcd = o3d.io.read_point_cloud(ply_file_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            print(f"  ❌ 点云为空")
            return None
        
        if verbose:
            print(f"  原始点数: {len(points)}")
        
        # 可选：下采样以减少计算量（如果点太多）
        if len(points) > 10000:
            pcd_down = pcd.voxel_down_sample(voxel_size=0.002)
            points = np.asarray(pcd_down.points)
            if verbose:
                print(f"  下采样后点数: {len(points)}")
        
        # 执行RANSAC矩形拟合
        result = fit_rectangle_ransac(points, 
                                      max_iterations=ransac_iterations,
                                      distance_threshold=ransac_threshold,
                                      verbose=verbose)
        
        if result is None:
            print(f"  ❌ 矩形拟合失败")
            return None
        
        # 提取文件信息
        file_name = os.path.basename(ply_file_path)
        folder_name = os.path.basename(os.path.dirname(ply_file_path))
        
        # 构建结果字典
        metrics = {
            '文件名': file_name,
            '文件夹名': folder_name,
            '完整路径': ply_file_path,
            '原始点数': result['point_count'],
            '平面内点数': result['plane_inlier_count'],
            '矩形内点数': result['rectangle_inlier_count'],
            '平面内点比例(%)': result['plane_inlier_ratio'] * 100,
            '矩形内点比例(%)': result['rectangle_inlier_ratio'] * 100,
            '总内点比例(%)': result['total_inlier_ratio'] * 100,
            '长度': result['length'],
            '宽度': result['width'],
            '长宽比': result['aspect_ratio'],
            '面积': result['rectangle_area'],
            '对角线长度': result['diagonal_length'],
            '周长': result['perimeter'],
            '中心点_X': result['centroid'][0],
            '中心点_Y': result['centroid'][1],
            '中心点_Z': result['centroid'][2],
            '法向量_X': result['plane_normal'][0],
            '法向量_Y': result['plane_normal'][1],
            '法向量_Z': result['plane_normal'][2],
            '角点1_X': result['corners_3d'][0][0],
            '角点1_Y': result['corners_3d'][0][1],
            '角点1_Z': result['corners_3d'][0][2],
            '角点2_X': result['corners_3d'][1][0],
            '角点2_Y': result['corners_3d'][1][1],
            '角点2_Z': result['corners_3d'][1][2],
            '角点3_X': result['corners_3d'][2][0],
            '角点3_Y': result['corners_3d'][2][1],
            '角点3_Z': result['corners_3d'][2][2],
            '角点4_X': result['corners_3d'][3][0],
            '角点4_Y': result['corners_3d'][3][1],
            '角点4_Z': result['corners_3d'][3][2],
        }
        
        if verbose:
            print(f"  ✅ 拟合成功: 长度={result['length']:.4f}, 宽度={result['width']:.4f}, 面积={result['rectangle_area']:.4f}")
            print(f"     内点比例: {result['total_inlier_ratio']*100:.1f}%")
        
        return metrics
        
    except Exception as e:
        print(f"  ❌ 处理出错: {e}")
        traceback.print_exc()
        return None


def batch_extract_references(input_folder, output_excel=None, 
                             ransac_iterations=500, ransac_threshold=0.01,
                             recursive=True, verbose=True):
    """
    批量提取文件夹中所有参照物点云的指标
    
    参数:
        input_folder: 输入文件夹路径（包含参照物PLY文件）
        output_excel: 输出Excel文件路径（默认自动生成）
        ransac_iterations: RANSAC迭代次数
        ransac_threshold: RANSAC距离阈值
        recursive: 是否递归搜索子文件夹
        verbose: 是否打印详细信息
    """
    print("=" * 80)
    print("批量提取参照物点云指标")
    print("=" * 80)
    print(f"输入文件夹: {input_folder}")
    print(f"递归搜索: {'是' if recursive else '否'}")
    print(f"RANSAC迭代次数: {ransac_iterations}")
    print(f"RANSAC距离阈值: {ransac_threshold}")
    print("=" * 80)
    
    # 查找所有PLY文件
    if recursive:
        ply_files = glob.glob(os.path.join(input_folder, "**", "*.ply"), recursive=True)
    else:
        ply_files = glob.glob(os.path.join(input_folder, "*.ply"))
    
    if not ply_files:
        print(f"错误: 在 {input_folder} 中未找到任何PLY文件")
        return None
    
    print(f"\n找到 {len(ply_files)} 个PLY文件")
    
    # 处理每个文件
    all_results = []
    success_count = 0
    fail_count = 0
    
    for i, ply_file in enumerate(ply_files, 1):
        print(f"\n[{i}/{len(ply_files)}] ", end="")
        
        result = extract_reference_metrics(ply_file, 
                                           ransac_iterations=ransac_iterations,
                                           ransac_threshold=ransac_threshold,
                                           verbose=verbose)
        
        if result:
            all_results.append(result)
            success_count += 1
        else:
            fail_count += 1
    
    # 保存结果
    if all_results:
        # 创建DataFrame
        df = pd.DataFrame(all_results)
        
        # 排序（按文件夹名和文件名）
        df = df.sort_values(['文件夹名', '文件名'])
        
        # 生成输出文件名
        if output_excel is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_excel = f"reference_metrics_{timestamp}.xlsx"
        
        # 保存到Excel
        with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:
            # 主要结果表
            df.to_excel(writer, sheet_name='参照物指标汇总', index=False)
            
            # 统计信息表
            stats = {
                '统计项': ['总文件数', '成功提取数', '失败数', '成功率(%)'],
                '数值': [len(ply_files), success_count, fail_count, 
                        100 * success_count / len(ply_files) if len(ply_files) > 0 else 0]
            }
            df_stats = pd.DataFrame(stats)
            df_stats.to_excel(writer, sheet_name='统计信息', index=False)
            
            # 长度统计
            if '长度' in df.columns:
                length_stats = {
                    '统计项': ['平均长度', '标准差', '最小值', '最大值', '中位数'],
                    '数值': [
                        df['长度'].mean(), df['长度'].std(), 
                        df['长度'].min(), df['长度'].max(), df['长度'].median()
                    ]
                }
                df_length = pd.DataFrame(length_stats)
                df_length.to_excel(writer, sheet_name='长度统计', index=False)
            
            # 宽度统计
            if '宽度' in df.columns:
                width_stats = {
                    '统计项': ['平均宽度', '标准差', '最小值', '最大值', '中位数'],
                    '数值': [
                        df['宽度'].mean(), df['宽度'].std(),
                        df['宽度'].min(), df['宽度'].max(), df['宽度'].median()
                    ]
                }
                df_width = pd.DataFrame(width_stats)
                df_width.to_excel(writer, sheet_name='宽度统计', index=False)
            
            # 面积统计
            if '面积' in df.columns:
                area_stats = {
                    '统计项': ['平均面积', '标准差', '最小值', '最大值', '中位数'],
                    '数值': [
                        df['面积'].mean(), df['面积'].std(),
                        df['面积'].min(), df['面积'].max(), df['面积'].median()
                    ]
                }
                df_area = pd.DataFrame(area_stats)
                df_area.to_excel(writer, sheet_name='面积统计', index=False)
            
            # 内点比例统计
            if '总内点比例(%)' in df.columns:
                inlier_stats = {
                    '统计项': ['平均内点比例(%)', '标准差', '最小值(%)', '最大值(%)', '中位数(%)'],
                    '数值': [
                        df['总内点比例(%)'].mean(), df['总内点比例(%)'].std(),
                        df['总内点比例(%)'].min(), df['总内点比例(%)'].max(), 
                        df['总内点比例(%)'].median()
                    ]
                }
                df_inlier = pd.DataFrame(inlier_stats)
                df_inlier.to_excel(writer, sheet_name='内点比例统计', index=False)
        
        print("\n" + "=" * 80)
        print("处理完成！")
        print("=" * 80)
        print(f"成功: {success_count} 个文件")
        print(f"失败: {fail_count} 个文件")
        print(f"成功率: {100 * success_count / len(ply_files):.1f}%")
        print(f"结果已保存到: {output_excel}")
        
        return output_excel
    else:
        print("\n没有成功提取任何参照物指标")
        return None


def main():
    parser = argparse.ArgumentParser(description='批量提取参照物点云指标（使用双重RANSAC矩形拟合）')
    
    parser.add_argument('input_folder', type=str,
                       help='包含参照物PLY文件的文件夹路径')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='输出Excel文件路径（默认自动生成）')
    parser.add_argument('--iterations', '-i', type=int, default=500,
                       help='RANSAC迭代次数（默认500）')
    parser.add_argument('--threshold', '-t', type=float, default=0.01,
                       help='RANSAC距离阈值（默认0.01）')
    parser.add_argument('--no-recursive', action='store_true',
                       help='不递归搜索子文件夹（默认递归搜索）')
    parser.add_argument('--quiet', '-q', action='store_true',
                       help='安静模式（减少输出信息）')
    
    args = parser.parse_args()
    
    # 检查输入文件夹是否存在
    if not os.path.exists(args.input_folder):
        print(f"错误: 文件夹不存在: {args.input_folder}")
        sys.exit(1)
    
    # 执行批量提取
    batch_extract_references(
        input_folder=args.input_folder,
        output_excel=args.output,
        ransac_iterations=args.iterations,
        ransac_threshold=args.threshold,
        recursive=not args.no_recursive,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()