# """
# 完整的叶片表型参数提取系统（命令行版）- 改进版
# 功能：
#   1. 自动运行SLBC骨架提取（输入：数据文件夹/output_*/hybrid_filtered_final.ply，输出：数据文件夹/contracted_points.ply）
#   2. 自动运行RANSAC主干提取（输入：数据文件夹/contracted_points.ply，输出：数据文件夹/ransac_stem_points.ply）
#   3. 提取叶片表型、株高和穗位高（输入：数据文件夹下的两个PLY文件）
#   4. 计算每个叶片与主干的夹角
#   5. 改进的叶长计算（中轴线法）和叶宽计算（分段宽度法）
#   6. 分析object值≥300的点云与穗部的关系
# 输入：一个或多个数据文件夹路径（如 /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames）
# 输出：在当前目录生成包含所有结果的Excel文件
# """

# import os
# import sys
# import numpy as np
# from pathlib import Path
# import pandas as pd
# import open3d as o3d
# import traceback
# from datetime import datetime
# import argparse
# import sys
# import glob
# import struct
# import subprocess
# from sklearn.decomposition import PCA
# from scipy.spatial import KDTree

# # ============================
# # 新增：双重RANSAC矩形拟合函数（基于独立脚本的方法）
# # ============================

# def fit_rectangle_ransac(points_3d, max_iterations=500, distance_threshold=0.01, verbose=True):
#     """
#     使用双重RANSAC算法拟合长方形（对离群点极其鲁棒）
#     输入：
#         points_3d - 参照物点云，n×3的数组
#         max_iterations - RANSAC最大迭代次数（2D矩形拟合）
#         distance_threshold - 点到矩形的距离阈值（用于判断内点）
#         verbose - 是否打印详细信息
#     输出：
#         包含拟合结果的字典，包括拟合后的点云
#     """
#     if len(points_3d) < 10:
#         print("⚠️ 参照物点数太少，无法拟合")
#         return None
    
#     if verbose:
#         print(f"\n{'='*50}")
#         print("参照物双重RANSAC矩形拟合")
#         print(f"{'='*50}")
#         print(f"参照物点数: {len(points_3d)}")
#         print(f"最大迭代次数: {max_iterations}")
#         print(f"距离阈值: {distance_threshold}")
    
#     try:
#         # 转换为open3d点云
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points_3d)
        
#         # ========== 第1步：使用RANSAC拟合主平面 ==========
#         if verbose:
#             print("\n第1步：拟合主平面（去除不在平面上的离群点）")
        
#         # RANSAC平面拟合
#         plane_model, plane_inliers = pcd.segment_plane(distance_threshold=distance_threshold * 0.5,
#                                                         ransac_n=3,
#                                                         num_iterations=200)
        
#         # 提取平面上的点
#         plane_points = np.asarray(pcd.select_by_index(plane_inliers).points)
        
#         if verbose:
#             print(f"  平面模型: [{plane_model[0]:.4f}, {plane_model[1]:.4f}, {plane_model[2]:.4f}, {plane_model[3]:.4f}]")
#             print(f"  平面内点: {len(plane_points)}/{len(points_3d)} 个 ({100*len(plane_points)/len(points_3d):.1f}%)")
        
#         if len(plane_points) < 10:
#             print("  错误：平面上点数太少")
#             return None
        
#         # ========== 第2步：将3D点投影到平面（转换为2D问题） ==========
#         if verbose:
#             print("\n第2步：将点投影到平面（3D → 2D）")
        
#         # 提取平面参数: ax + by + cz + d = 0
#         [a, b, c, d] = plane_model
#         normal = np.array([a, b, c])
#         normal = normal / np.linalg.norm(normal)
        
#         # 构建平面上的局部坐标系
#         # 选择一个与法向量不平行的向量作为参考
#         if abs(normal[2]) < 0.9:
#             # 使用(1,0,-a/c)作为u轴
#             if c != 0:
#                 u = np.array([1, 0, -a/c])
#             else:
#                 u = np.array([1, 0, 0])
#             u = u / np.linalg.norm(u)
#         else:
#             # 如果法向量接近垂直，使用(0,1,0)作为u轴
#             u = np.array([0, 1, 0])
        
#         # v轴垂直于u和法向量
#         v = np.cross(normal, u)
#         v = v / np.linalg.norm(v)
        
#         # 重新调整u轴使其垂直于v（确保正交）
#         u = np.cross(v, normal)
#         u = u / np.linalg.norm(u)
        
#         # 计算平面中心
#         plane_center = np.mean(plane_points, axis=0)
        
#         # 将所有点投影到2D平面
#         points_2d = np.zeros((len(plane_points), 2))
#         for i, p in enumerate(plane_points):
#             points_2d[i, 0] = np.dot(p - plane_center, u)
#             points_2d[i, 1] = np.dot(p - plane_center, v)
        
#         if verbose:
#             print(f"  2D点云范围: X=[{np.min(points_2d[:,0]):.4f}, {np.max(points_2d[:,0]):.4f}]")
#             print(f"              Y=[{np.min(points_2d[:,1]):.4f}, {np.max(points_2d[:,1]):.4f}]")
        
#         # ========== 第3步：RANSAC拟合矩形（在2D空间中） ==========
#         if verbose:
#             print("\n第3步：RANSAC拟合矩形（2D空间）")
        
#         best_inliers = None
#         best_rectangle = None
#         best_score = 0
#         best_corners_2d = None
        
#         for iteration in range(max_iterations):
#             # 随机采样4个点（矩形需要4个点来定义边界）
#             sample_indices = np.random.choice(len(points_2d), 4, replace=False)
#             sample = points_2d[sample_indices]
            
#             try:
#                 # 计算这4个点的边界矩形
#                 x_min = np.min(sample[:, 0])
#                 x_max = np.max(sample[:, 0])
#                 y_min = np.min(sample[:, 1])
#                 y_max = np.max(sample[:, 1])
                
#                 # 检查矩形是否合理（长宽比不能太极端）
#                 rect_width = x_max - x_min
#                 rect_height = y_max - y_min
#                 aspect_ratio = max(rect_width, rect_height) / (min(rect_width, rect_height) + 1e-6)
                
#                 # 参照物的长宽比应该在1:1到3:1之间（长方形）
#                 if aspect_ratio > 5:
#                     continue
                
#                 # 计算内点（点到矩形的距离小于阈值）
#                 # 点到矩形边界的最小距离
#                 dx = np.maximum(x_min - points_2d[:, 0], 0) + np.maximum(points_2d[:, 0] - x_max, 0)
#                 dy = np.maximum(y_min - points_2d[:, 1], 0) + np.maximum(points_2d[:, 1] - y_max, 0)
#                 distances = np.sqrt(dx**2 + dy**2)
                
#                 inlier_mask = distances <= distance_threshold
#                 inlier_count = np.sum(inlier_mask)
                
#                 # 更新最佳结果
#                 if inlier_count > best_score:
#                     best_score = inlier_count
#                     best_inliers = inlier_mask
#                     best_rectangle = (x_min, x_max, y_min, y_max)
#                     best_corners_2d = np.array([
#                         [x_min, y_min],
#                         [x_max, y_min],
#                         [x_max, y_max],
#                         [x_min, y_max]
#                     ])
                    
#                     if verbose and iteration % 100 == 0:
#                         print(f"  迭代 {iteration}: 找到 {inlier_count} 个内点 ({100*inlier_count/len(points_2d):.1f}%)")
                        
#             except Exception as e:
#                 continue
        
#         if best_rectangle is None:
#             print("  错误：未找到有效的矩形")
#             return None
        
#         # ========== 第4步：使用内点重新精炼矩形 ==========
#         if verbose:
#             print("\n第4步：使用内点精炼矩形")
        
#         # 提取内点
#         inlier_points = points_2d[best_inliers]
#         inlier_count = len(inlier_points)
        
#         if verbose:
#             print(f"  内点数量: {inlier_count}/{len(points_2d)} ({100*inlier_count/len(points_2d):.1f}%)")
        
#         # 使用内点重新计算精确的矩形边界
#         x_min = np.min(inlier_points[:, 0])
#         x_max = np.max(inlier_points[:, 0])
#         y_min = np.min(inlier_points[:, 1])
#         y_max = np.max(inlier_points[:, 1])
        
#         # 计算长度和宽度
#         length = max(x_max - x_min, y_max - y_min)
#         width = min(x_max - x_min, y_max - y_min)
        
#         # 确保length >= width，并调整角点顺序
#         if (x_max - x_min) < (y_max - y_min):
#             # 交换方向
#             corners_2d = np.array([
#                 [x_min, y_min],
#                 [x_min, y_max],
#                 [x_max, y_max],
#                 [x_max, y_min]
#             ])
#         else:
#             corners_2d = np.array([
#                 [x_min, y_min],
#                 [x_max, y_min],
#                 [x_max, y_max],
#                 [x_min, y_max]
#             ])
        
#         if verbose:
#             print(f"\n{'='*50}")
#             print("矩形拟合结果")
#             print(f"{'='*50}")
#             print(f"长度: {length:.6f}")
#             print(f"宽度: {width:.6f}")
#             print(f"长宽比: {length/width:.2f}")
#             print(f"内点比例: {100*inlier_count/len(points_2d):.1f}%")
#             print(f"矩形面积: {length * width:.6f}")
        
#         # ========== 第5步：将2D角点转换回3D ==========
#         corners_3d = []
#         for corner_2d in corners_2d:
#             point_3d = plane_center + corner_2d[0] * u + corner_2d[1] * v
#             corners_3d.append(point_3d)
        
#         corners_3d = np.array(corners_3d)
        
#         # ========== 第6步：生成拟合后的矩形点云 ==========
#         fitted_pcd = o3d.geometry.PointCloud()
        
#         # 生成矩形点云（边界和内部）
#         rectangle_points = []
        
#         # 定义矩形的四条边
#         edges = [
#             (corners_3d[0], corners_3d[1]),  # 底边
#             (corners_3d[1], corners_3d[2]),  # 右边
#             (corners_3d[2], corners_3d[3]),  # 顶边
#             (corners_3d[3], corners_3d[0])   # 左边
#         ]
        
#         # 每条边上生成50个点
#         num_points_per_edge = 50
#         for edge in edges:
#             start, end = edge
#             for i in range(num_points_per_edge):
#                 t = i / (num_points_per_edge - 1)
#                 point = start + t * (end - start)
#                 rectangle_points.append(point)
        
#         # 生成矩形内部的网格点
#         num_grid_x = 20
#         num_grid_y = 20
        
#         # 获取矩形的两条边向量
#         edge1 = corners_3d[1] - corners_3d[0]
#         edge2 = corners_3d[3] - corners_3d[0]
        
#         for i in range(num_grid_x + 1):
#             for j in range(num_grid_y + 1):
#                 t1 = i / num_grid_x
#                 t2 = j / num_grid_y
#                 point = corners_3d[0] + t1 * edge1 + t2 * edge2
#                 rectangle_points.append(point)
        
#         rectangle_points = np.array(rectangle_points)
#         fitted_pcd.points = o3d.utility.Vector3dVector(rectangle_points)
        
#         return {
#             'length_2d': length,
#             'width_2d': width,
#             'rectangle_area_2d': length * width,
#             'centroid': plane_center,
#             'plane_normal': normal,
#             'u_axis': u,
#             'v_axis': v,
#             'corners_3d': corners_3d,
#             'corners_2d': corners_2d,
#             'fitted_point_cloud': fitted_pcd,
#             'fitted_points': rectangle_points,
#             'inlier_points': points_2d[best_inliers] if best_inliers is not None else points_2d,
#             'projected_points': points_2d,
#             'point_count': len(points_3d),
#             'inlier_count': inlier_count,
#             'inlier_ratio': inlier_count / len(points_2d)
#         }
        
#     except Exception as e:
#         print(f"双重RANSAC矩形拟合过程中出错: {e}")
#         traceback.print_exc()
#         return None


# def save_fitted_reference_cloud(reference_result, output_dir, original_filename):
#     """
#     保存拟合后的参照物点云
#     """
#     if reference_result is None or reference_result.get('fitted_point_cloud') is None:
#         return None
    
#     base_name = os.path.splitext(os.path.basename(original_filename))[0]
#     reference_dir = os.path.join(output_dir, f"{base_name}_reference")
#     os.makedirs(reference_dir, exist_ok=True)
    
#     # 保存拟合的矩形点云
#     fitted_path = os.path.join(reference_dir, "fitted_reference_rectangle.ply")
#     o3d.io.write_point_cloud(fitted_path, reference_result['fitted_point_cloud'])
#     print(f"拟合后的参照物点云已保存: {fitted_path}")
    
#     # 保存矩形角点（作为点云）
#     corners_pcd = o3d.geometry.PointCloud()
#     corners_pcd.points = o3d.utility.Vector3dVector(reference_result['corners_3d'])
#     corners_path = os.path.join(reference_dir, "rectangle_corners.ply")
#     o3d.io.write_point_cloud(corners_path, corners_pcd)
#     print(f"矩形角点已保存: {corners_path}")
    
#     # 保存内点云（用于可视化）
#     if reference_result.get('inlier_points') is not None:
#         # 将2D内点转换回3D并保存
#         inlier_points_3d = []
#         plane_center = reference_result['centroid']
#         u = reference_result['u_axis']
#         v = reference_result['v_axis']
#         for point_2d in reference_result['inlier_points']:
#             point_3d = plane_center + point_2d[0] * u + point_2d[1] * v
#             inlier_points_3d.append(point_3d)
        
#         inlier_pcd = o3d.geometry.PointCloud()
#         inlier_pcd.points = o3d.utility.Vector3dVector(np.array(inlier_points_3d))
#         inlier_path = os.path.join(reference_dir, "rectangle_inliers.ply")
#         o3d.io.write_point_cloud(inlier_path, inlier_pcd)
#         print(f"矩形内点云已保存: {inlier_path}")
    
#     return reference_dir


# # ============================
# # 修改：去除聚类的文件解析函数
# # ============================

# def parse_ply_file_detailed(file_path, enable_clustering=False, eps=0.02, min_points=10):
#     """
#     详细解析PLY文件，返回所有属性和按object_value值分类的点云
#     修改：enable_clustering默认为False，不进行聚类
#     """
#     points_dict = {}
#     all_points = []
    
#     print(f"\n正在解析PLY文件: {file_path}")
#     if enable_clustering:
#         print(f"启用聚类滤除: eps={eps}, min_points={min_points}")
#     else:
#         print("不启用聚类，直接使用原始点云")
    
#     try:
#         with open(file_path, 'r') as f:
#             # 读取头部
#             header_lines = []
#             while True:
#                 line = f.readline().strip()
#                 header_lines.append(line)
#                 if line == 'end_header':
#                     break
            
#             # 获取顶点数量和属性
#             vertex_count = 0
#             properties = []
            
#             for line in header_lines:
#                 if line.startswith('element vertex'):
#                     vertex_count = int(line.split()[2])
#                 elif line.startswith('property'):
#                     prop_parts = line.split()
#                     if len(prop_parts) >= 3:
#                         prop_type = prop_parts[1]
#                         prop_name = prop_parts[2]
#                         properties.append((prop_name, prop_type))
            
#             # 找到x, y, z和object_value的索引
#             x_idx = y_idx = z_idx = obj_idx = -1
#             for i, (prop_name, _) in enumerate(properties):
#                 if prop_name == 'x':
#                     x_idx = i
#                 elif prop_name == 'y':
#                     y_idx = i
#                 elif prop_name == 'z':
#                     z_idx = i
#                 elif prop_name == 'object_value':
#                     obj_idx = i
            
#             if -1 in [x_idx, y_idx, z_idx, obj_idx]:
#                 print("错误: 找不到必要的属性")
#                 return {}, np.array([])
            
#             # 按object_value收集所有点
#             raw_points_dict = {}
#             for i in range(vertex_count):
#                 line = f.readline().strip()
#                 if not line:
#                     continue
                
#                 values = line.split()
#                 try:
#                     x = float(values[x_idx])
#                     y = float(values[y_idx])
#                     z = float(values[z_idx])
#                     obj_val = int(values[obj_idx])
                    
#                     if obj_val not in raw_points_dict:
#                         raw_points_dict[obj_val] = []
#                     raw_points_dict[obj_val].append([x, y, z])
#                 except:
#                     continue
            
#             # 如果启用聚类，则进行聚类滤除；否则直接使用原始点
#             if enable_clustering:
#                 for obj_val, points_list in raw_points_dict.items():
#                     if len(points_list) < min_points:
#                         points_dict[obj_val] = np.array(points_list)
#                         continue
                    
#                     points = np.array(points_list)
#                     pcd = o3d.geometry.PointCloud()
#                     pcd.points = o3d.utility.Vector3dVector(points)
                    
#                     labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
                    
#                     if len(labels) == 0 or np.max(labels) < 0:
#                         points_dict[obj_val] = points
#                         continue
                    
#                     unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
#                     max_cluster_idx = unique_labels[np.argmax(counts)]
#                     mask = labels == max_cluster_idx
#                     filtered_points = points[mask]
                    
#                     points_dict[obj_val] = filtered_points
                    
#                     for point in filtered_points:
#                         all_points.append([point[0], point[1], point[2], obj_val])
#             else:
#                 # 不使用聚类，直接使用原始点云
#                 for obj_val, points_list in raw_points_dict.items():
#                     points_dict[obj_val] = np.array(points_list)
#                     for point in points_list:
#                         all_points.append([point[0], point[1], point[2], obj_val])
            
#             all_points = np.array(all_points) if all_points else np.array([])
            
#             return points_dict, all_points
            
#     except Exception as e:
#         print(f"解析PLY文件时出错: {e}")
#         return {}, np.array([])


# def parse_binary_ply_for_y(file_path):
#     """
#     解析binary格式的PLY文件，只获取y轴坐标信息（用于株高计算）
#     """
#     print(f"\n正在解析辅助PLY文件: {file_path}")
    
#     try:
#         with open(file_path, 'rb') as f:
#             # 读取头部
#             header_lines = []
#             while True:
#                 line = f.readline().decode('utf-8', errors='ignore').strip()
#                 header_lines.append(line)
#                 if line == 'end_header':
#                     break
            
#             # 获取顶点数量和属性
#             vertex_count = 0
#             properties = []
            
#             for line in header_lines:
#                 if line.startswith('element vertex'):
#                     vertex_count = int(line.split()[2])
#                 elif line.startswith('property'):
#                     prop_parts = line.split()
#                     if len(prop_parts) >= 3:
#                         prop_type = prop_parts[1]
#                         prop_name = prop_parts[2]
#                         properties.append((prop_name, prop_type))
            
#             # 找到y的索引
#             y_idx = -1
#             for i, (prop_name, _) in enumerate(properties):
#                 if prop_name == 'y':
#                     y_idx = i
#                     break
            
#             if y_idx == -1:
#                 print("错误: 找不到y属性")
#                 return None, None, 0
            
#             # 确定每个属性的大小
#             type_sizes = {'double': 8, 'float': 4, 'int': 4, 'uint': 4,
#                          'uchar': 1, 'char': 1, 'short': 2, 'ushort': 2}
            
#             point_size = 0
#             for _, prop_type in properties:
#                 base_type = prop_type.split('(')[0] if '(' in prop_type else prop_type
#                 point_size += type_sizes.get(base_type, 4)
            
#             # 读取所有点的y坐标
#             y_values = []
#             for i in range(vertex_count):
#                 data = f.read(point_size)
#                 if len(data) < point_size:
#                     break
                
#                 offset = 0
#                 for j, (_, prop_type) in enumerate(properties):
#                     base_type = prop_type.split('(')[0] if '(' in prop_type else prop_type
#                     size = type_sizes.get(base_type, 4)
                    
#                     if j == y_idx:
#                         if base_type in ['double', 'float64']:
#                             fmt = '<d'
#                         elif base_type in ['float', 'float32']:
#                             fmt = '<f'
#                         else:
#                             fmt = '<f'
                        
#                         try:
#                             y_val = struct.unpack(fmt, data[offset:offset+size])[0]
#                             if not (np.isnan(y_val) or np.isinf(y_val)):
#                                 y_values.append(y_val)
#                         except:
#                             pass
#                         break
                    
#                     offset += size
            
#             if not y_values:
#                 return None, None, 0
            
#             return np.min(y_values), np.max(y_values), len(y_values)
            
#     except Exception as e:
#         print(f"解析binary PLY文件时出错: {e}")
#         return None, None, 0


# def save_all_object_pointclouds(points_dict, output_dir, original_filename):
#     """
#     保存所有object的点云为PLY文件
#     """
#     print("\n" + "=" * 60)
#     print("保存所有object的点云文件")
#     print("=" * 60)
    
#     base_name = os.path.splitext(os.path.basename(original_filename))[0]
#     objects_dir = os.path.join(output_dir, f"{base_name}_objects")
#     os.makedirs(objects_dir, exist_ok=True)
    
#     saved_count = 0
#     object_info = []
    
#     for obj_val, points in points_dict.items():
#         if len(points) == 0:
#             continue
        
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
        
#         filename = os.path.join(objects_dir, f"object_{obj_val:03d}.ply")
#         o3d.io.write_point_cloud(filename, pcd)
        
#         centroid = np.mean(points, axis=0)
#         y_min = np.min(points[:, 1])
#         y_max = np.max(points[:, 1])
        
#         object_info.append({
#             'object_value': obj_val,
#             'point_count': len(points),
#             'y_min': y_min,
#             'y_max': y_max,
#             'centroid_x': centroid[0],
#             'centroid_y': centroid[1],
#             'centroid_z': centroid[2],
#             'filename': filename
#         })
        
#         saved_count += 1
    
#     if object_info:
#         df = pd.DataFrame(object_info)
#         csv_path = os.path.join(objects_dir, "objects_summary.csv")
#         df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
#         print(f"\n已保存 {saved_count} 个object的点云文件到: {objects_dir}")
    
#     return objects_dir, object_info


# # ============================
# # 改进的叶长计算函数（中轴线法）
# # ============================
# def calculate_leaf_length(points, n_sections=20):
#     """
#     计算叶片长度 - 使用中轴线法（更准确，适用于弯曲叶片）
#     输入：叶片点云，分段数
#     输出：叶片长度（点云空间）
#     """
#     if len(points) < 30:
#         return 0.0
    
#     try:
#         # 计算主轴方向
#         centroid = np.mean(points, axis=0)
#         points_centered = points - centroid
        
#         # PCA获取主轴
#         U, S, Vt = np.linalg.svd(points_centered, full_matrices=False)
#         length_axis = Vt[0]  # 第一主成分（长度方向）
        
#         # 投影到长度轴
#         projections = np.dot(points_centered, length_axis)
        
#         # 分段数调整（确保不要太少）
#         n_sections = min(n_sections, len(points) // 10)
#         if n_sections < 2:
#             # 点太少，使用简单投影法
#             return np.max(projections) - np.min(projections)
        
#         # 创建分段
#         min_proj = np.min(projections)
#         max_proj = np.max(projections)
#         bins = np.linspace(min_proj, max_proj, n_sections + 1)
        
#         # 收集中轴点
#         axis_points = []
        
#         for i in range(n_sections):
#             # 获取当前段的点
#             if i == n_sections - 1:
#                 mask = (projections >= bins[i]) & (projections <= bins[i+1])
#             else:
#                 mask = (projections >= bins[i]) & (projections < bins[i+1])
            
#             section_points = points[mask]
            
#             if len(section_points) >= 3:
#                 # 计算截面中心（中轴点）
#                 section_center = np.mean(section_points, axis=0)
#                 axis_points.append(section_center)
        
#         if len(axis_points) < 2:
#             # 分段失败，使用简单投影法
#             return np.max(projections) - np.min(projections)
        
#         # 计算中轴线总长度
#         axis_points = np.array(axis_points)
#         axis_length = 0.0
#         for i in range(len(axis_points) - 1):
#             axis_length += np.linalg.norm(axis_points[i+1] - axis_points[i])
        
#         return axis_length
        
#     except Exception as e:
#         print(f"  长度计算错误: {e}")
#         return 0.0


# # ============================
# # 改进的叶宽计算函数（分段宽度法）
# # ============================
# def calculate_leaf_width(points, n_sections=15):
#     """
#     计算叶片宽度 - 分段宽度法（更准确，输出平均宽度和最大宽度）
#     输入：叶片点云，分段数
#     输出：平均宽度, 最大宽度
#     """
#     if len(points) < 30:
#         return 0.0, 0.0
    
#     try:
#         # 计算主轴方向
#         centroid = np.mean(points, axis=0)
#         points_centered = points - centroid
        
#         # PCA获取主轴
#         U, S, Vt = np.linalg.svd(points_centered, full_matrices=False)
#         length_axis = Vt[0]  # 第一主成分（长度方向）
#         width_axis = Vt[1]   # 第二主成分（宽度方向）
        
#         # 投影到长度轴
#         length_projections = np.dot(points_centered, length_axis)
        
#         # 分段数调整
#         n_sections = min(n_sections, len(points) // 10)
#         if n_sections < 2:
#             # 点太少，使用整体宽度
#             width_projections = np.dot(points_centered, width_axis)
#             overall_width = np.max(width_projections) - np.min(width_projections)
#             return overall_width, overall_width
        
#         # 创建分段
#         min_proj = np.min(length_projections)
#         max_proj = np.max(length_projections)
#         bins = np.linspace(min_proj, max_proj, n_sections + 1)
        
#         # 存储每段的宽度
#         section_widths = []
        
#         for i in range(n_sections):
#             # 获取当前段的点
#             if i == n_sections - 1:
#                 mask = (length_projections >= bins[i]) & (length_projections <= bins[i+1])
#             else:
#                 mask = (length_projections >= bins[i]) & (length_projections < bins[i+1])
            
#             section_points = points[mask]
            
#             if len(section_points) >= 3:
#                 # 计算该段的宽度
#                 section_centered = section_points - centroid
#                 width_projections = np.dot(section_centered, width_axis)
#                 section_width = np.max(width_projections) - np.min(width_projections)
#                 section_widths.append(section_width)
        
#         if not section_widths:
#             # 没有有效分段，使用整体宽度
#             width_projections = np.dot(points_centered, width_axis)
#             overall_width = np.max(width_projections) - np.min(width_projections)
#             return overall_width, overall_width
        
#         # 计算宽度统计
#         avg_width = np.mean(section_widths)
#         max_width = np.max(section_widths)
        
#         return avg_width, max_width
        
#     except Exception as e:
#         print(f"  宽度计算错误: {e}")
#         return 0.0, 0.0


# def calculate_leaf_area_alpha_shape(points, alpha=0.015):
#     """
#     计算叶片面积
#     """
#     if len(points) < 30:
#         return 0.0
    
#     try:
#         pcd = o3d.geometry.PointCloud()
#         pcd.points = o3d.utility.Vector3dVector(points)
        
#         if len(points) > 2000:
#             pcd = pcd.voxel_down_sample(voxel_size=0.003)
        
#         mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
#             pcd, alpha=alpha)
        
#         if len(mesh.triangles) > 0:
#             return mesh.get_surface_area()
#         else:
#             return 0.0
            
#     except Exception as e:
#         return 0.0


# def calibrate_leaf_measurements(reference_result, leaf_points, leaf_id, file_name):
#     """
#     使用参照物标定计算叶片真实尺寸
#     修改：使用双重RANSAC拟合结果中的长度和面积进行标定
#     """
#     # 如果没有参照物结果，只返回3D测量值
#     if reference_result is None:
#         leaf_length_3d = calculate_leaf_length(leaf_points)
#         leaf_avg_width_3d, leaf_max_width_3d = calculate_leaf_width(leaf_points)
#         leaf_area_3d = calculate_leaf_area_alpha_shape(leaf_points)
        
#         results = {
#             '文件名': os.path.basename(file_name),
#             '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             '叶片ID': leaf_id,
#             '叶片点数': len(leaf_points),
#             '参照物3D长度': None,
#             '参照物拟合面积': None,
#             '参照物真实长度': 9.5,
#             '参照物真实面积': 47.5,
#             '长度标定系数': None,
#             '面积标定系数': None,
#             '3D长度(中轴线法)': leaf_length_3d,
#             '3D平均宽度(分段法)': leaf_avg_width_3d,
#             '3D最大宽度(分段法)': leaf_max_width_3d,
#             '3D面积': leaf_area_3d,
#             '真实长度': None,
#             '真实平均宽度': None,
#             '真实最大宽度': None,
#             '真实面积': None,
#             '处理状态': '缺失参照物(220)'
#         }
#         return results
    
#     REFERENCE_REAL_LENGTH = 9.5
#     REFERENCE_REAL_AREA = 47.5
    
#     # 使用双重RANSAC拟合得到的长度和面积
#     REFERENCE_3D_LENGTH = reference_result['length_2d']
#     REFERENCE_AREA = reference_result['rectangle_area_2d']
    
#     length_scale = REFERENCE_REAL_LENGTH / REFERENCE_3D_LENGTH
#     area_scale = REFERENCE_REAL_AREA / REFERENCE_AREA
    
#     leaf_length_3d = calculate_leaf_length(leaf_points)
#     leaf_length_real = leaf_length_3d * length_scale
    
#     leaf_avg_width_3d, leaf_max_width_3d = calculate_leaf_width(leaf_points)
#     leaf_avg_width_real = leaf_avg_width_3d * length_scale
#     leaf_max_width_real = leaf_max_width_3d * length_scale
    
#     leaf_area_3d = calculate_leaf_area_alpha_shape(leaf_points)
#     leaf_area_real = leaf_area_3d * area_scale
    
#     results = {
#         '文件名': os.path.basename(file_name),
#         '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#         '叶片ID': leaf_id,
#         '叶片点数': len(leaf_points),
#         '参照物3D长度': REFERENCE_3D_LENGTH,
#         '参照物拟合面积': REFERENCE_AREA,
#         '参照物真实长度': REFERENCE_REAL_LENGTH,
#         '参照物真实面积': REFERENCE_REAL_AREA,
#         '长度标定系数': length_scale,
#         '面积标定系数': area_scale,
#         '3D长度(中轴线法)': leaf_length_3d,
#         '3D平均宽度(分段法)': leaf_avg_width_3d,
#         '3D最大宽度(分段法)': leaf_max_width_3d,
#         '3D面积': leaf_area_3d,
#         '真实长度': leaf_length_real,
#         '真实平均宽度': leaf_avg_width_real,
#         '真实最大宽度': leaf_max_width_real,
#         '真实面积': leaf_area_real,
#         '处理状态': '成功'
#     }
    
#     return results


# # ============================
# # 分析object值≥300的点云与穗部的关系（修改版）
# # ============================

# def analyze_high_object_values(points_dict, stem_points, selected_ear_object, ear_points=None):
#     """
#     分析object值≥300的点云与穗部的关系
    
#     逻辑：
#     1. 找到大于j的所有i中最小的那个i → 得到 object A
#     2. 找到所有小于 i_A 的i中最大的那个i → 得到 object C
    
#     参数:
#         points_dict: 所有object的点云字典
#         stem_points: 主干点云（用于计算距离）
#         selected_ear_object: 用于求穗位高的穗的object值（222-226中的一个）
#         ear_points: 该穗的点云（可选，如果不提供则从points_dict中获取）
    
#     返回:
#         analysis_results: 包含分析结果的字典
#         high_obj_info: 包含每个≥300的object的信息列表
#     """
#     print("\n" + "=" * 60)
#     print("分析object值≥300的点云")
#     print("=" * 60)
    
#     # 获取穗部点云
#     if ear_points is None and selected_ear_object is not None:
#         ear_points = points_dict.get(selected_ear_object, None)
    
#     if ear_points is None or len(ear_points) == 0:
#         print("⚠️ 无法获取穗部点云，跳过分析")
#         return None, None
    
#     # 计算穗部点云中y轴最小的10%的点的y轴中位数j
#     y_values_ear = ear_points[:, 1]
#     n_ear = len(ear_points)
#     k_ear = max(1, int(n_ear * 0.1))
    
#     # 按y值排序，取最小的10%
#     sorted_indices_ear = np.argsort(y_values_ear)
#     bottom_10_indices = sorted_indices_ear[:k_ear]
#     bottom_10_y = y_values_ear[bottom_10_indices]
#     j = np.median(bottom_10_y)
    
#     print(f"\n穗部object值: {selected_ear_object}")
#     print(f"穗部点云总数: {len(ear_points)}")
#     print(f"y轴最小的10%点数: {k_ear}")
#     print(f"这些点的y轴中位数 j: {j:.6f}")
    
#     # 查找object值≥300的点云
#     high_object_values = [obj_val for obj_val in points_dict.keys() if obj_val >= 300]
    
#     if not high_object_values:
#         print("\n未找到object值≥300的点云")
#         return None, None
    
#     print(f"\n找到 {len(high_object_values)} 个object值≥300的点云: {sorted(high_object_values)}")
    
#     # 对每个object值≥300的点云，计算y轴坐标最大的10%的点的y轴中位数i
#     high_obj_info = []
    
#     for obj_val in sorted(high_object_values):
#         points = points_dict[obj_val]
#         if len(points) == 0:
#             continue
        
#         print(f"\n处理 object={obj_val}, 点数={len(points)}")
        
#         # 取y轴坐标最大的10%的点
#         y_values_all = points[:, 1]  # 获取所有点的y坐标
#         k = max(1, int(len(points) * 0.1))  # 10%的点数
        
#         # 按y值从大到小排序，取最大的10%
#         sorted_indices = np.argsort(y_values_all)[::-1]  # 降序排列
#         top_10_indices = sorted_indices[:k]  # 取最大的10%
#         top_10_points = points[top_10_indices]  # 获取这些点的坐标
        
#         # 计算这些点的y轴中位数i
#         y_values_top = top_10_points[:, 1]
#         i = np.median(y_values_top)
        
#         print(f"  y轴最大的10%点数: {len(top_10_points)}")
#         print(f"  这些点的y轴中位数 i: {i:.6f}")
        
#         high_obj_info.append({
#             'object_value': obj_val,
#             'point_count': len(points),
#             'top_points_count': len(top_10_points),
#             'y_median_i': i,
#             'y_min': np.min(points[:, 1]),
#             'y_max': np.max(points[:, 1])
#         })
    
#     if not high_obj_info:
#         print("\n未找到有效的object值≥300的点云信息")
#         return None, None
    
#     # 第一步：找到大于j的所有i中最小的那个i
#     greater_than_j = [info for info in high_obj_info if info['y_median_i'] > j]
#     if greater_than_j:
#         min_greater = min(greater_than_j, key=lambda x: x['y_median_i'])
#         min_greater_obj = min_greater['object_value']
#         min_greater_i = min_greater['y_median_i']
#         print(f"\n步骤1: 大于j({j:.6f})的所有i中最小的i: {min_greater_i:.6f} (object={min_greater_obj})")
#     else:
#         min_greater_obj = None
#         min_greater_i = None
#         print(f"\n步骤1: 没有找到大于j({j:.6f})的i值")
    
#     # 第二步：找到所有小于 min_greater_i 的i中最大的那个i
#     max_less_obj = None
#     max_less_i = None
    
#     if min_greater_i is not None:
#         less_than_min_greater = [info for info in high_obj_info 
#                                   if info['y_median_i'] < min_greater_i]
        
#         if less_than_min_greater:
#             max_less = max(less_than_min_greater, key=lambda x: x['y_median_i'])
#             max_less_obj = max_less['object_value']
#             max_less_i = max_less['y_median_i']
#             print(f"\n步骤2: 所有小于 {min_greater_i:.6f} 的i中最大的i: {max_less_i:.6f} (object={max_less_obj})")
#         else:
#             print(f"\n步骤2: 没有找到小于 {min_greater_i:.6f} 的i值")
#     else:
#         print(f"\n步骤2: 无法执行（步骤1未找到有效值）")
    
#     analysis_results = {
#         'ear_object_value': selected_ear_object,
#         'ear_point_count': len(ear_points),
#         'ear_bottom_10_percent_median_y': j,
#         'greater_than_j_min_object': min_greater_obj,
#         'greater_than_j_min_median_y': min_greater_i,
#         'less_than_min_greater_max_object': max_less_obj,
#         'less_than_min_greater_max_median_y': max_less_i,
#         'high_object_info': high_obj_info
#     }
    
#     return analysis_results, high_obj_info


# # ============================
# # 叶片夹角计算函数
# # ============================

# def compute_pca_direction(points):
#     """
#     计算点云的PCA主方向
#     """
#     if len(points) < 10:
#         return None
    
#     centroid = np.mean(points, axis=0)
#     centered = points - centroid
    
#     pca = PCA(n_components=3)
#     pca.fit(centered)
    
#     return pca.components_[0]

# def compute_angle_between_vectors(v1, v2):
#     """
#     计算两个向量之间的夹角（度）- 取锐角
#     """
#     v1_norm = v1 / np.linalg.norm(v1)
#     v2_norm = v2 / np.linalg.norm(v2)
    
#     cos_angle = np.abs(np.dot(v1_norm, v2_norm))
#     cos_angle = np.clip(cos_angle, -1, 1)
#     angle = np.arccos(cos_angle) * 180 / np.pi
    
#     return angle

# def find_nearest_leaf_points(leaf_points, stem_points, top_percent=30):
#     """
#     找到叶片点云中离主干最近的top_percent%的点
#     """
#     if len(leaf_points) == 0 or len(stem_points) == 0:
#         return leaf_points
    
#     stem_tree = KDTree(stem_points)
#     distances, _ = stem_tree.query(leaf_points)
    
#     k = max(1, int(len(leaf_points) * top_percent / 100))
#     indices = np.argsort(distances)[:k]
    
#     return leaf_points[indices]

# def calculate_leaf_angle(leaf_points, stem_direction, stem_points=None, use_nearest_percent=30):
#     """
#     计算单个叶片与主干的夹角
#     """
#     if len(leaf_points) < 10:
#         return None, None
    
#     # 如果提供了主干点云，只使用离主干最近的点
#     if stem_points is not None and use_nearest_percent > 0:
#         leaf_points_used = find_nearest_leaf_points(leaf_points, stem_points, use_nearest_percent)
#         if len(leaf_points_used) < 10:
#             leaf_points_used = leaf_points
#     else:
#         leaf_points_used = leaf_points
    
#     # 计算叶片方向
#     leaf_direction = compute_pca_direction(leaf_points_used)
    
#     if leaf_direction is None:
#         return None, None
    
#     # 计算夹角
#     angle = compute_angle_between_vectors(stem_direction, leaf_direction)
    
#     return angle, leaf_direction

# def get_stem_direction_from_ransac(ransac_ply_path):
#     """
#     从RANSAC生成的主干点云计算主干方向
#     """
#     print(f"\n📐 读取RANSAC主干点云: {ransac_ply_path}")
    
#     # 读取主干点云
#     stem_pcd = o3d.io.read_point_cloud(ransac_ply_path)
#     stem_points = np.asarray(stem_pcd.points)
    
#     if len(stem_points) < 10:
#         print("❌ 主干点数太少")
#         return None, None
    
#     # PCA计算主干方向
#     stem_direction = compute_pca_direction(stem_points)
    
#     if stem_direction is None:
#         return None, None
    
#     # 修改：确保方向向上（Y轴正方向，因为垂直方向是Y轴）
#     if stem_direction[1] < 0:  # 改为检查Y轴
#         stem_direction = -stem_direction
    
#     # 修改：计算与垂直方向（Y轴）的夹角
#     vertical_angle = np.arccos(np.abs(stem_direction[1])) * 180 / np.pi  # 改为使用Y轴
    
#     print(f"主干方向向量: [{stem_direction[0]:.4f}, {stem_direction[1]:.4f}, {stem_direction[2]:.4f}]")
#     print(f"主干与垂直方向（Y轴）夹角: {vertical_angle:.2f}°")
    
#     return stem_direction, stem_points


# # ============================
# # 原有的文件查找函数
# # ============================

# def find_main_ply_files(folder_path):
#     """
#     在文件夹中自动查找所需的两个PLY文件
#     - point_cloud_main_clusters_merged.ply
#     - hybrid_filtered_final.ply（用于株高计算）
#     """
#     folder = Path(folder_path)
    
#     print(f"\n正在搜索PLY文件: {folder_path}")
    
#     # 查找 point_cloud_main_clusters_merged.ply
#     main_ply_files = list(folder.rglob("point_cloud_main_clusters_merged.ply"))
    
#     # 查找 hybrid_filtered_final.ply
#     hybrid_ply_files = list(folder.rglob("hybrid_filtered_final.ply"))
    
#     if not main_ply_files:
#         print(f"❌ 错误: 未找到 point_cloud_main_clusters_merged.ply")
#         return None, None, False
    
#     if not hybrid_ply_files:
#         print(f"❌ 错误: 未找到 hybrid_filtered_final.ply")
#         return None, None, False
    
#     main_ply = str(main_ply_files[0])
#     hybrid_ply = str(hybrid_ply_files[0])
    
#     print(f"找到主文件: {main_ply}")
#     print(f"找到辅助文件: {hybrid_ply}")
    
#     return main_ply, hybrid_ply, True


# # ============================
# # 运行外部命令的函数
# # ============================

# def get_python_executable():
#     """
#     获取当前运行的Python解释器路径
#     """
#     return sys.executable

# def find_hybrid_ply(folder_path):
#     """
#     在数据文件夹下查找 hybrid_filtered_final.ply
#     位置：数据文件夹/output_*/hybrid_filtered_final.ply
#     """
#     folder = Path(folder_path)
    
#     # 查找所有 output_* 子目录下的 hybrid_filtered_final.ply
#     hybrid_files = list(folder.glob("output_*/hybrid_filtered_final.ply"))
    
#     if not hybrid_files:
#         print(f"❌ 在 {folder_path} 中未找到 output_*/hybrid_filtered_final.ply")
#         return None
    
#     return str(hybrid_files[0])


# def run_slbc_extraction(folder_path):
#     """
#     运行SLBC骨架提取脚本
#     输入：数据文件夹/output_*/hybrid_filtered_final.ply
#     输出：数据文件夹/contracted_points.ply
#     """
#     print("\n" + "=" * 60)
#     print("第一步：运行SLBC骨架提取")
#     print("=" * 60)
    
#     # 查找hybrid_filtered_final.ply
#     hybrid_ply = find_hybrid_ply(folder_path)
#     if not hybrid_ply:
#         print(f"❌ 错误: 在文件夹 {folder_path} 中未找到 hybrid_filtered_final.ply")
#         return None
    
#     print(f"找到输入文件: {hybrid_ply}")
    
#     # 输出文件路径（直接在数据文件夹根目录）
#     contracted_path = os.path.join(folder_path, "contracted_points.ply")
    
#     # SLBC脚本路径
#     script_path = "/datashare/dir_liusha/pc-skeletor/pc-skeletor-main/example_tree.py"
    
#     if not os.path.exists(script_path):
#         print(f"❌ 错误: 找不到SLBC脚本: {script_path}")
#         return None
    
#     # 获取当前Python解释器路径
#     python_exe = get_python_executable()
#     print(f"使用Python解释器: {python_exe}")
    
#     # 创建临时修改版的脚本（只修改输入输出路径）
#     temp_script = os.path.join(folder_path, "temp_slbc_script.py")
    
#     try:
#         with open(script_path, 'r') as f:
#             script_content = f.read()
        
#         # 修改输入路径为找到的hybrid_ply
#         modified_content = script_content.replace(
#             'pcd_youzi0 = o3d.io.read_point_cloud("/datashare/dir_liusha/xibeinonglin/样本数据/91227-2_frames/output_91227-2/hybrid_filtered_final.ply")',
#             f'pcd_youzi0 = o3d.io.read_point_cloud("{hybrid_ply}")'
#         )
        
#         # 修改输出目录为数据文件夹根目录
#         modified_content = modified_content.replace(
#             "output_dir = './output_slbc'",
#             f"output_dir = '{folder_path}'"
#         )
        
#         # 在脚本开头添加路径，确保能找到pc_skeletor模块
#         pc_skeletor_path = "/datashare/dir_liusha/pc-skeletor/pc-skeletor-main"
#         path_addition = f"""
# import sys
# sys.path.insert(0, '{pc_skeletor_path}')
# print(f"添加路径: {{sys.path[0]}}")
# """
        
#         # 在文件开头插入路径
#         modified_content = path_addition + modified_content
        
#         with open(temp_script, 'w') as f:
#             f.write(modified_content)
        
#         print(f"运行SLBC脚本...")
#         print(f"输入: {hybrid_ply}")
#         print(f"输出: {contracted_path}")
        
#         # 使用正确的Python解释器运行
#         result = subprocess.run([python_exe, temp_script], 
#                                capture_output=True, text=True, check=True)
#         print(result.stdout)
#         if result.stderr:
#             print("警告输出:", result.stderr)
        
#     except subprocess.CalledProcessError as e:
#         print(f"❌ SLBC脚本运行失败: {e}")
#         print("标准输出:", e.stdout)
#         print("错误输出:", e.stderr)
#         return None
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")
#         traceback.print_exc()
#         return None
#     finally:
#         # 清理临时文件
#         if os.path.exists(temp_script):
#             os.remove(temp_script)
#             print(f"已清理临时脚本: {temp_script}")
    
#     # 检查输出文件
#     if os.path.exists(contracted_path):
#         print(f"✅ SLBC提取成功: {contracted_path}")
#         return contracted_path
#     else:
#         print(f"❌ SLBC提取失败: 未找到输出文件 {contracted_path}")
#         return None


# def run_ransac_extraction(folder_path, contracted_path):
#     """
#     运行RANSAC主干提取脚本
#     输入：数据文件夹/contracted_points.ply
#     输出：数据文件夹/ransac_stem_points.ply
#     """
#     print("\n" + "=" * 60)
#     print("第二步：运行RANSAC主干提取")
#     print("=" * 60)
    
#     # RANSAC脚本路径
#     ransac_script = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型/只保留主干点云.py"
    
#     if not os.path.exists(ransac_script):
#         print(f"❌ 错误: 找不到RANSAC脚本: {ransac_script}")
#         return None
    
#     # 输出文件路径
#     ransac_path = os.path.join(folder_path, "ransac_stem_points.ply")
    
#     # 获取当前Python解释器路径
#     python_exe = get_python_executable()
    
#     print(f"运行RANSAC脚本...")
#     print(f"输入文件: {contracted_path}")
#     print(f"输出文件: {ransac_path}")
    
#     try:
#         # 直接调用RANSAC脚本，传入输入文件路径作为参数
#         result = subprocess.run([python_exe, ransac_script, contracted_path], 
#                                capture_output=True, text=True, check=True)
#         print(result.stdout)
#         if result.stderr:
#             print("警告输出:", result.stderr)
        
#         # RANSAC脚本可能直接在输入文件所在目录输出结果
#         # 我们需要将结果移动到正确的位置
#         possible_output = os.path.join(os.path.dirname(contracted_path), "ransac_stem_points.ply")
#         if os.path.exists(possible_output) and possible_output != ransac_path:
#             import shutil
#             shutil.move(possible_output, ransac_path)
#             print(f"已移动结果文件到: {ransac_path}")
        
#     except subprocess.CalledProcessError as e:
#         print(f"❌ RANSAC脚本运行失败: {e}")
#         print("标准输出:", e.stdout)
#         print("错误输出:", e.stderr)
#         return None
    
#     # 检查输出文件
#     if os.path.exists(ransac_path):
#         print(f"✅ RANSAC提取成功: {ransac_path}")
#         return ransac_path
#     else:
#         print(f"❌ RANSAC提取失败: 未找到输出文件 {ransac_path}")
#         return None


# # ============================
# # 修改后的主处理函数
# # ============================

# def process_single_data_folder(data_folder_path, eps=0.02, min_points=10, 
#                                use_nearest_percent=30, skip_skeleton=False,
#                                ransac_max_iterations=500, ransac_distance_threshold=0.01):
#     """
#     处理单个数据文件夹
#     参数:
#         data_folder_path: 数据文件夹路径
#         ransac_max_iterations: RANSAC矩形拟合最大迭代次数
#         ransac_distance_threshold: RANSAC矩形拟合距离阈值
#     """
#     print("\n" + "=" * 80)
#     print(f"处理数据文件夹: {data_folder_path}")
#     print("=" * 80)
    
#     folder_name = os.path.basename(os.path.normpath(data_folder_path))
    
#     # ========== 运行骨架提取 ==========
#     if not skip_skeleton:
#         print("\n开始骨架提取流程...")
#         contracted_path = run_slbc_extraction(data_folder_path)
#         if contracted_path is None:
#             print("❌ SLBC骨架提取失败，无法继续")
#             return None
        
#         ransac_path = run_ransac_extraction(data_folder_path, contracted_path)
#         if ransac_path is None:
#             print("⚠️ RANSAC主干提取失败，将继续但无法计算叶片夹角")
#             ransac_path = None
#     else:
#         # 如果跳过，查找已有的文件
#         contracted_path = os.path.join(data_folder_path, "contracted_points.ply")
#         ransac_path = os.path.join(data_folder_path, "ransac_stem_points.ply")
        
#         if not os.path.exists(contracted_path):
#             print(f"❌ 找不到 contracted_points.ply: {contracted_path}")
#             return None
        
#         print(f"跳过骨架提取，使用已有文件:")
#         print(f"  contracted: {contracted_path}")
#         print(f"  ransac: {ransac_path if os.path.exists(ransac_path) else '不存在'}")
    
#     # ========== 计算主干方向 ==========
#     stem_direction = None
#     stem_points = None

#     if ransac_path and os.path.exists(ransac_path):
#         stem_direction, stem_points = get_stem_direction_from_ransac(ransac_path)
#     elif contracted_path and os.path.exists(contracted_path):
#         print("使用收缩点云计算主干方向")
#         contracted_pcd = o3d.io.read_point_cloud(contracted_path)
#         contracted_points = np.asarray(contracted_pcd.points)
#         stem_direction = compute_pca_direction(contracted_points)
#         # 修改：确保方向向上（Y轴正方向）
#         if stem_direction is not None and stem_direction[1] < 0:  # 改为检查Y轴
#             stem_direction = -stem_direction
#         stem_points = contracted_points

#     if stem_direction is None:
#         print("⚠️ 警告: 无法计算主干方向，叶片夹角将无法计算")
    
#     # ========== 查找表型分析所需的文件 ==========
#     main_ply, hybrid_ply, success = find_main_ply_files(data_folder_path)
    
#     if not success:
#         print(f"❌ 错误: 在文件夹 {data_folder_path} 中未找到所需的表型分析文件")
#         return None
    
#     # 创建输出目录（在当前目录下，以文件夹名命名）
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     output_dir = f"results_{folder_name}_{timestamp}"
#     os.makedirs(output_dir, exist_ok=True)
#     print(f"\n表型分析结果将保存到: {output_dir}")
    
#     # 解析主点云文件（修改：不启用聚类）
#     points_dict, all_points = parse_ply_file_detailed(
#         main_ply, enable_clustering=False, eps=eps, min_points=min_points
#     )
    
#     if not points_dict:
#         print("错误: 无法解析点云文件")
#         return None
    
#     # 保存所有object的点云
#     objects_dir, object_info = save_all_object_pointclouds(points_dict, output_dir, main_ply)
    
#     # 解析辅助文件获取y轴信息（用于株高和穗位高）
#     binary_y_min, binary_y_max, binary_point_count = parse_binary_ply_for_y(hybrid_ply)
    
#     if binary_y_min is None or binary_y_max is None:
#         print("错误: 无法解析辅助文件")
#         return None
    
#     # ========== 提取参照物（使用双重RANSAC矩形拟合） ==========
#     reference_result = None
#     missing_reference = False
    
#     if 220 not in points_dict:
#         print("⚠️ 警告: 点云中不存在object_value=220的参照物，叶片尺寸将无法标定")
#         missing_reference = True
#     else:
#         reference_points = points_dict[220]
#         # 使用双重RANSAC矩形拟合
#         reference_result = fit_rectangle_ransac(
#             reference_points,
#             max_iterations=ransac_max_iterations,
#             distance_threshold=ransac_distance_threshold,
#             verbose=True
#         )
#         if reference_result is None:
#             print("⚠️ 警告: 参照物双重RANSAC矩形拟合失败，叶片尺寸将无法标定")
#             missing_reference = True
#         else:
#             # 保存拟合后的参照物点云
#             save_fitted_reference_cloud(reference_result, output_dir, main_ply)
    
#     # 计算标定系数（仅在参照物存在时计算）
#     length_scale = None
#     if reference_result is not None:
#         REFERENCE_REAL_LENGTH = 9.5
#         length_scale = REFERENCE_REAL_LENGTH / reference_result['length_2d']
#     else:
#         length_scale = None
    
#     # ========== 计算株高 ==========
#     plant_base_y = None
#     has_base_221 = 221 in points_dict
    
#     if has_base_221:
#         plant_base_y = np.min(points_dict[221][:, 1])
#         print(f"使用object=221作为基部: y_min={plant_base_y:.6f}")
#     else:
#         plant_base_y = binary_y_min
#         print(f"⚠️ 未找到object=221，使用辅助文件y_min作为基部: {plant_base_y:.6f}")
    
#     plant_height_measured = binary_y_max - plant_base_y
#     plant_height_real = plant_height_measured * length_scale if length_scale is not None else None
    
#     # ========== 计算穗位高 ==========
#     print("\n" + "=" * 60)
#     print("计算穗位高（多object值分析）")
#     print("=" * 60)
    
#     ear_object_values = [222, 223, 224, 225, 226]
#     ear_stats = []
#     missing_ear = True
    
#     for obj_val in ear_object_values:
#         if obj_val in points_dict:
#             points = points_dict[obj_val]
#             if len(points) > 0:
#                 y_median = np.median(points[:, 1])
#                 y_max = np.max(points[:, 1])
#                 ear_stats.append({
#                     'object_value': obj_val,
#                     'point_count': len(points),
#                     'y_median': y_median,
#                     'y_max': y_max
#                 })
#                 print(f"  object {obj_val}: 点数={len(points):6d}, y中位数={y_median:.6f}, y最大值={y_max:.6f}")
#                 missing_ear = False
#             else:
#                 print(f"  object {obj_val}: 存在但点数为0")
#         else:
#             print(f"  object {obj_val}: 不存在")
    
#     selected_ear_object = None
#     ear_position_y = None
#     ear_height_measured = None
#     ear_height_real = None
#     ear_points = None
    
#     if not missing_ear and ear_stats:
#         min_median_obj = min(ear_stats, key=lambda x: x['y_median'])
#         selected_ear_object = min_median_obj['object_value']
#         ear_position_y = min_median_obj['y_max']
#         ear_points = points_dict.get(selected_ear_object, None)
        
#         print(f"\n✅ 选中的object值: {selected_ear_object}")
#         print(f"  该object y中位数: {min_median_obj['y_median']:.6f} (最小)")
#         print(f"  该object y最大值: {ear_position_y:.6f}")
        
#         ear_height_measured = binary_y_max - ear_position_y
#         ear_height_real = ear_height_measured * length_scale if length_scale is not None else None
        
#         print(f"\n穗位高计算结果:")
#         print(f"  辅助文件y最大值: {binary_y_max:.6f}")
#         print(f"  选中的object y最大值: {ear_position_y:.6f}")
#         print(f"  穗位高测量值: {ear_height_measured:.6f}")
#         if ear_height_real is not None:
#             print(f"  真实穗位高: {ear_height_real:.6f} cm")
#         else:
#             print(f"  真实穗位高: 无法计算（缺少参照物）")
#     else:
#         print("\n⚠️ 警告: 未找到任何穗部object值(222-226)的点云")
    
#     # ========== 分析object值≥300的点云 ==========
#     high_object_analysis, high_obj_info = analyze_high_object_values(
#         points_dict, stem_points, selected_ear_object, ear_points
#     )
    
#     # ========== 处理所有叶片并计算夹角 ==========
#     print("\n" + "=" * 60)
#     print("开始处理所有叶片点云并计算夹角")
#     print("=" * 60)
    
#     NON_LEAF_OBJECTS = {220, 221, 222, 223, 224, 225, 226}
    
#     all_leaf_results = []
#     leaf_count = 0
    
#     for obj_val, points in points_dict.items():
#         if obj_val in NON_LEAF_OBJECTS:
#             continue
        
#         if len(points) < 30:
#             print(f"\n跳过 object_value={obj_val}: 点数太少 ({len(points)})")
#             continue
        
#         print(f"\n处理叶片 {leaf_count + 1}: object_value={obj_val}, 点数={len(points)}")
        
#         leaf_result = calibrate_leaf_measurements(reference_result, points, obj_val, main_ply)
        
#         if leaf_result:
#             if stem_direction is not None:
#                 angle, leaf_direction = calculate_leaf_angle(
#                     points, stem_direction, stem_points, use_nearest_percent
#                 )
#                 if angle is not None:
#                     leaf_result['叶片与主干夹角(°)'] = angle
#                     leaf_result['叶片方向_x'] = leaf_direction[0] if leaf_direction is not None else None
#                     leaf_result['叶片方向_y'] = leaf_direction[1] if leaf_direction is not None else None
#                     leaf_result['叶片方向_z'] = leaf_direction[2] if leaf_direction is not None else None
#                     print(f"  叶片夹角: {angle:.2f}°")
#                 else:
#                     leaf_result['叶片与主干夹角(°)'] = None
#                     leaf_result['叶片方向_x'] = None
#                     leaf_result['叶片方向_y'] = None
#                     leaf_result['叶片方向_z'] = None
#             else:
#                 leaf_result['叶片与主干夹角(°)'] = None
#                 leaf_result['叶片方向_x'] = None
#                 leaf_result['叶片方向_y'] = None
#                 leaf_result['叶片方向_z'] = None
            
#             leaf_result['叶片序号'] = leaf_count + 1
#             leaf_result['object值'] = obj_val
            
#             all_leaf_results.append(leaf_result)
#             leaf_count += 1
            
#             if leaf_result['处理状态'] == '成功':
#                 print(f"  长度: {leaf_result['真实长度']:.2f} cm")
#                 print(f"  平均宽度: {leaf_result['真实平均宽度']:.2f} cm")
#                 print(f"  最大宽度: {leaf_result['真实最大宽度']:.2f} cm")
#                 print(f"  面积: {leaf_result['真实面积']:.2f} cm²")
#             else:
#                 print(f"  3D长度: {leaf_result['3D长度(中轴线法)']:.2f}")
#                 print(f"  3D平均宽度: {leaf_result['3D平均宽度(分段法)']:.2f}")
#                 print(f"  3D面积: {leaf_result['3D面积']:.2f}")
#                 print(f"  状态: {leaf_result['处理状态']}")
    
#     print(f"\n叶片处理完成，共找到 {leaf_count} 个叶片")
    
#     if high_object_analysis:
#         print("\n" + "=" * 60)
#         print("高object值(≥300)分析结果摘要")
#         print("=" * 60)
#         print(f"穗部object值: {high_object_analysis['ear_object_value']}")
#         print(f"穗部y轴最小10%点的中位数j: {high_object_analysis['ear_bottom_10_percent_median_y']:.6f}")
        
#         min_greater_obj = high_object_analysis['greater_than_j_min_object']
#         max_less_obj = high_object_analysis['less_than_min_greater_max_object']
        
#         if min_greater_obj:
#             print(f"\n步骤1 - 大于j的最小i对应的object值: {min_greater_obj}")
#         if max_less_obj:
#             print(f"步骤2 - 小于该i的最大i对应的object值: {max_less_obj}")
    
#     all_results = {
#         'folder_name': folder_name,
#         'folder_path': data_folder_path,
#         'main_ply': main_ply,
#         'hybrid_ply': hybrid_ply,
#         'contracted_ply': contracted_path,
#         'ransac_ply': ransac_path if ransac_path and os.path.exists(ransac_path) else None,
#         'missing_flags': {
#             'missing_reference_220': missing_reference,
#             'missing_base_221': not has_base_221,
#             'missing_ear_all': missing_ear
#         },
#         'plant_height': {
#             '测量值': plant_height_measured,
#             '真实值': plant_height_real,
#             '基部y': plant_base_y,
#             '顶部y': binary_y_max,
#             '基部来源': 'object_221' if has_base_221 else 'hybrid_file'
#         } if plant_base_y is not None else None,
#         'ear_height': {
#             '测量值': ear_height_measured,
#             '真实值': ear_height_real,
#             '穗位y': ear_position_y,
#             '顶部y': binary_y_max,
#             '选中object值': selected_ear_object
#         } if ear_position_y is not None else None,
#         'ear_analysis': ear_stats,
#         'high_object_analysis': high_object_analysis,
#         'high_obj_info': high_obj_info,
#         'leaves': all_leaf_results,
#         'length_scale': length_scale,
#         'reference_result': reference_result,
#         'binary_file_info': {
#             'y_min': binary_y_min,
#             'y_max': binary_y_max,
#             'point_count': binary_point_count
#         },
#         'stem_direction': stem_direction.tolist() if stem_direction is not None else None,
#         'output_dir': output_dir
#     }
    
#     return all_results


# def save_results_to_excel(all_results_list, output_path=None):
#     """
#     将所有处理结果保存到一个Excel文件中
#     """
#     if not all_results_list:
#         print("没有结果可保存")
#         return None
    
#     if output_path is None:
#         timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#         output_path = f"all_plant_results_{timestamp}.xlsx"
    
#     with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
#         # 所有叶片结果汇总
#         all_leaves = []
#         for folder_result in all_results_list:
#             if folder_result and 'leaves' in folder_result:
#                 for leaf in folder_result['leaves']:
#                     leaf['文件夹'] = folder_result['folder_name']
#                     leaf['文件夹路径'] = folder_result['folder_path']
#                     missing = folder_result.get('missing_flags', {})
#                     leaf['参照物状态'] = '缺失' if missing.get('missing_reference_220', False) else '正常'
#                     leaf['基部状态'] = '缺失' if missing.get('missing_base_221', False) else '正常'
#                     leaf['穗部状态'] = '缺失' if missing.get('missing_ear_all', False) else '正常'
#                     if folder_result['stem_direction']:
#                         leaf['主干方向_x'] = folder_result['stem_direction'][0]
#                         leaf['主干方向_y'] = folder_result['stem_direction'][1]
#                         leaf['主干方向_z'] = folder_result['stem_direction'][2]
#                     all_leaves.append(leaf)
        
#         if all_leaves:
#             df_leaves = pd.DataFrame(all_leaves)
#             cols = df_leaves.columns.tolist()
#             angle_cols = ['叶片与主干夹角(°)', '叶片方向_x', '叶片方向_y', '叶片方向_z', 
#                          '主干方向_x', '主干方向_y', '主干方向_z']
#             other_cols = [c for c in cols if c not in angle_cols]
#             new_cols = other_cols[:5] + angle_cols + other_cols[5:]
#             new_cols = [c for c in new_cols if c in df_leaves.columns]
#             df_leaves = df_leaves[new_cols]
#             df_leaves.to_excel(writer, sheet_name='所有叶片汇总', index=False)
#             print(f"叶片汇总: {len(all_leaves)} 条记录")
        
#         # 高object值分析结果表
#         high_obj_data = []
#         for folder_result in all_results_list:
#             if folder_result and folder_result.get('high_obj_info'):
#                 high_analysis = folder_result.get('high_object_analysis')
#                 folder_name = folder_result['folder_name']
                
#                 if high_analysis is not None:
#                     greater_obj = high_analysis.get('greater_than_j_min_object')
#                     max_less_obj = high_analysis.get('less_than_min_greater_max_object')
#                 else:
#                     greater_obj = None
#                     max_less_obj = None
                
#                 for info in folder_result['high_obj_info']:
#                     obj_val = info['object_value']
#                     mark_type = ''
#                     if obj_val == greater_obj:
#                         mark_type = '大于j的最小i对应的object'
#                     elif obj_val == max_less_obj:
#                         mark_type = '小于该i的最大i对应的object'
                    
#                     high_obj_data.append({
#                         '文件夹': folder_name,
#                         '文件夹路径': folder_result['folder_path'],
#                         'object值': obj_val,
#                         '点数': info['point_count'],
#                         'y轴最大10%点数': info['top_points_count'],
#                         'y轴最大10%点y轴中位数(i)': info['y_median_i'],
#                         'y_min': info['y_min'],
#                         'y_max': info['y_max'],
#                         '穗部object值': high_analysis.get('ear_object_value') if high_analysis else None,
#                         '穗部y轴最小10%点中位数(j)': high_analysis.get('ear_bottom_10_percent_median_y') if high_analysis else None,
#                         '标记说明': mark_type
#                     })
        
#         if high_obj_data:
#             df_high_obj = pd.DataFrame(high_obj_data)
#             df_high_obj.to_excel(writer, sheet_name='高object值分析(≥300)', index=False)
#             print(f"高object值分析: {len(high_obj_data)} 条记录")
        
#         # 株高结果表
#         plant_height_data = []
#         for folder_result in all_results_list:
#             if folder_result:
#                 missing = folder_result.get('missing_flags', {})
#                 if folder_result.get('plant_height') is not None:
#                     ph = folder_result['plant_height']
#                     plant_height_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
#                         '基部状态': '缺失' if missing.get('missing_base_221', False) else '正常',
#                         '基部来源': ph.get('基部来源', 'unknown'),
#                         '基部y': ph['基部y'],
#                         '顶部y': ph['顶部y'],
#                         '株高测量值': ph['测量值'],
#                         '真实株高': ph['真实值'] if ph.get('真实值') is not None else 'N/A'
#                     })
#                 else:
#                     plant_height_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
#                         '基部状态': '缺失' if missing.get('missing_base_221', False) else '正常',
#                         '基部来源': 'N/A',
#                         '基部y': 'N/A',
#                         '顶部y': 'N/A',
#                         '株高测量值': 'N/A',
#                         '真实株高': 'N/A'
#                     })
        
#         if plant_height_data:
#             pd.DataFrame(plant_height_data).to_excel(writer, sheet_name='株高结果', index=False)
#             print(f"株高结果: {len(plant_height_data)} 条记录")
        
#         # 穗位高结果表
#         ear_height_data = []
#         for folder_result in all_results_list:
#             if folder_result:
#                 missing = folder_result.get('missing_flags', {})
#                 if folder_result.get('ear_height') is not None:
#                     eh = folder_result['ear_height']
#                     ear_height_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
#                         '穗部状态': '缺失' if missing.get('missing_ear_all', False) else '正常',
#                         '选中object值': eh.get('选中object值', 'N/A'),
#                         '穗位y': eh['穗位y'] if eh.get('穗位y') is not None else 'N/A',
#                         '顶部y': eh['顶部y'],
#                         '穗位高测量值': eh['测量值'] if eh.get('测量值') is not None else 'N/A',
#                         '真实穗位高': eh['真实值'] if eh.get('真实值') is not None else 'N/A'
#                     })
#                 else:
#                     ear_height_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
#                         '穗部状态': '缺失' if missing.get('missing_ear_all', False) else '正常',
#                         '选中object值': 'N/A',
#                         '穗位y': 'N/A',
#                         '顶部y': 'N/A',
#                         '穗位高测量值': 'N/A',
#                         '真实穗位高': 'N/A'
#                     })
        
#         if ear_height_data:
#             pd.DataFrame(ear_height_data).to_excel(writer, sheet_name='穗位高结果', index=False)
#             print(f"穗位高结果: {len(ear_height_data)} 条记录")
        
#         # 穗部object详细分析表
#         ear_analysis_data = []
#         for folder_result in all_results_list:
#             if folder_result and folder_result.get('ear_analysis'):
#                 for ear_stat in folder_result['ear_analysis']:
#                     ear_analysis_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         'object值': ear_stat['object_value'],
#                         '点数': ear_stat['point_count'],
#                         'y中位数': ear_stat['y_median'],
#                         'y最大值': ear_stat['y_max']
#                     })
        
#         if ear_analysis_data:
#             pd.DataFrame(ear_analysis_data).to_excel(writer, sheet_name='穗部object分析', index=False)
#             print(f"穗部object分析: {len(ear_analysis_data)} 条记录")
        
#         # 文件夹统计
#         folder_stats = []
#         for folder_result in all_results_list:
#             if folder_result:
#                 missing = folder_result.get('missing_flags', {})
#                 high_analysis = folder_result.get('high_object_analysis')
#                 folder_stats.append({
#                     '文件夹': folder_result['folder_name'],
#                     '文件夹路径': folder_result['folder_path'],
#                     '参照物(220)': '缺失' if missing.get('missing_reference_220', False) else '存在',
#                     '基部(221)': '缺失' if missing.get('missing_base_221', False) else '存在',
#                     '穗部(222-226)': '缺失' if missing.get('missing_ear_all', False) else '存在',
#                     '叶片数量': len(folder_result.get('leaves', [])),
#                     '长度标定系数': folder_result['length_scale'] if folder_result.get('length_scale') is not None else 'N/A',
#                     '参照物3D长度': folder_result['reference_result']['length_2d'] if folder_result.get('reference_result') else 'N/A',
#                     '参照物拟合面积': folder_result['reference_result']['rectangle_area_2d'] if folder_result.get('reference_result') else 'N/A',
#                     '参照物内点比例': f"{folder_result['reference_result']['inlier_ratio']*100:.1f}%" if folder_result.get('reference_result') else 'N/A',
#                     '辅助文件y_min': folder_result['binary_file_info']['y_min'],
#                     '辅助文件y_max': folder_result['binary_file_info']['y_max'],
#                     'contracted_points存在': os.path.exists(folder_result['contracted_ply']) if folder_result['contracted_ply'] else False,
#                     'ransac_points存在': os.path.exists(folder_result['ransac_ply']) if folder_result['ransac_ply'] else False,
#                     '大于j的最小i对应的object值': high_analysis.get('greater_than_j_min_object') if high_analysis else None,
#                     '小于该i的最大i对应的object值': high_analysis.get('less_than_min_greater_max_object') if high_analysis else None
#                 })
        
#         if folder_stats:
#             df_stats = pd.DataFrame(folder_stats)
#             df_stats.to_excel(writer, sheet_name='文件夹统计', index=False)
#             print(f"文件夹统计: {len(folder_stats)} 个")
        
#         # 关键object值摘要表
#         summary_data = []
#         for folder_result in all_results_list:
#             if folder_result:
#                 high_analysis = folder_result.get('high_object_analysis')
#                 if high_analysis:
#                     summary_data.append({
#                         '文件夹': folder_result['folder_name'],
#                         '文件夹路径': folder_result['folder_path'],
#                         '穗部object值': high_analysis.get('ear_object_value'),
#                         '穗部y轴最小10%点中位数(j)': high_analysis.get('ear_bottom_10_percent_median_y'),
#                         '大于j的最小i对应的object值': high_analysis.get('greater_than_j_min_object'),
#                         '该object的i值': high_analysis.get('greater_than_j_min_median_y'),
#                         '小于该i的最大i对应的object值': high_analysis.get('less_than_min_greater_max_object'),
#                         '该object的i值': high_analysis.get('less_than_min_greater_max_median_y')
#                     })
        
#         if summary_data:
#             pd.DataFrame(summary_data).to_excel(writer, sheet_name='关键object值摘要', index=False)
#             print(f"关键object值摘要: {len(summary_data)} 条记录")
    
#     print(f"\n✅ 所有结果已保存到: {output_path}")
#     return output_path


# def main():
#     parser = argparse.ArgumentParser(description='完整的叶片表型参数提取系统 - 改进版（使用双重RANSAC矩形拟合）')
    
#     parser.add_argument('folders', nargs='+', 
#                        help='要处理的数据文件夹路径')
#     parser.add_argument('--eps', type=float, default=0.02, help='DBSCAN聚类半径（已禁用，保留参数兼容性）')
#     parser.add_argument('--min_points', type=int, default=10, help='DBSCAN最小簇点数（已禁用，保留参数兼容性）')
#     parser.add_argument('--excel', '-e', type=str, default=None, help='Excel输出路径')
#     parser.add_argument('--nearest_percent', type=int, default=30, 
#                        help='使用离主干最近的百分比点计算叶片方向')
#     parser.add_argument('--skip_skeleton', action='store_true',
#                        help='跳过骨架提取，使用已有的文件')
#     parser.add_argument('--ransac_iter', type=int, default=500,
#                        help='RANSAC矩形拟合最大迭代次数（默认500）')
#     parser.add_argument('--ransac_thresh', type=float, default=0.01,
#                        help='RANSAC矩形拟合距离阈值（默认0.01）')
    
#     args = parser.parse_args()
    
#     # 展开文件夹列表
#     expanded_folders = []
#     for pattern in args.folders:
#         expanded_folders.extend(glob.glob(pattern))
    
#     expanded_folders = [f for f in expanded_folders if os.path.isdir(f)]
    
#     if not expanded_folders:
#         print("错误: 没有找到匹配的文件夹")
#         sys.exit(1)
    
#     print("=" * 80)
#     print("完整的叶片表型参数提取系统 - 改进版（使用双重RANSAC矩形拟合）")
#     print("=" * 80)
#     print(f"当前Python解释器: {get_python_executable()}")
#     print(f"注意: 聚类功能已禁用，直接在原始点云上处理")
#     print(f"叶片夹角: 使用离主干最近的 {args.nearest_percent}% 点")
#     print(f"骨架提取: {'跳过' if args.skip_skeleton else '自动运行'}")
#     print(f"参照物拟合: 双重RANSAC矩形拟合方法")
#     print(f"  RANSAC迭代次数: {args.ransac_iter}")
#     print(f"  RANSAC距离阈值: {args.ransac_thresh}")
#     print(f"高object值分析: object值≥300的点云，计算与穗部的关系")
#     print(f"  步骤1: 找到大于j的最小i对应的object")
#     print(f"  步骤2: 找到小于该i的最大i对应的object")
#     print(f"待处理数据文件夹: {len(expanded_folders)}")
#     print("=" * 80)
    
#     print("\n找到的数据文件夹:")
#     for folder in expanded_folders:
#         print(f"  - {folder}")
    
#     all_results_list = []
    
#     for i, folder_path in enumerate(expanded_folders, 1):
#         print(f"\n[{i}/{len(expanded_folders)}] 处理: {folder_path}")
        
#         try:
#             results = process_single_data_folder(
#                 folder_path,
#                 eps=args.eps,
#                 min_points=args.min_points,
#                 use_nearest_percent=args.nearest_percent,
#                 skip_skeleton=args.skip_skeleton,
#                 ransac_max_iterations=args.ransac_iter,
#                 ransac_distance_threshold=args.ransac_thresh
#             )
            
#             if results:
#                 all_results_list.append(results)
#                 print(f"\n✅ {folder_path} 处理成功")
#             else:
#                 print(f"\n❌ {folder_path} 处理失败")
#         except Exception as e:
#             print(f"\n❌ {folder_path} 处理失败: {e}")
#             traceback.print_exc()
    
#     if all_results_list:
#         save_results_to_excel(all_results_list, args.excel)
#         print(f"\n处理完成! 成功处理 {len(all_results_list)}/{len(expanded_folders)} 个文件夹")
#     else:
#         print("\n没有成功处理任何文件夹")


# if __name__ == "__main__":
#     main()


# # 运行示例：
# # python 真实表型提取代码_0327_改进.py /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames  --ransac_iter 500 --ransac_thresh 0.01









"""
完整的叶片表型参数提取系统（命令行版）- 改进版
修改：使用外部命令行工具进行主干提取
"""

import os
import sys
import numpy as np
from pathlib import Path
import pandas as pd
import open3d as o3d
import traceback
from datetime import datetime
import argparse
import glob
import struct
import subprocess
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

# ============================
# 新增：双重RANSAC矩形拟合函数（基于独立脚本的方法）
# ============================

def fit_rectangle_ransac(points_3d, max_iterations=500, distance_threshold=0.01, verbose=True):
    """
    使用双重RANSAC算法拟合长方形（对离群点极其鲁棒）
    输入：
        points_3d - 参照物点云，n×3的数组
        max_iterations - RANSAC最大迭代次数（2D矩形拟合）
        distance_threshold - 点到矩形的距离阈值（用于判断内点）
        verbose - 是否打印详细信息
    输出：
        包含拟合结果的字典，包括拟合后的点云
    """
    if len(points_3d) < 10:
        print("⚠️ 参照物点数太少，无法拟合")
        return None
    
    if verbose:
        print(f"\n{'='*50}")
        print("参照物双重RANSAC矩形拟合")
        print(f"{'='*50}")
        print(f"参照物点数: {len(points_3d)}")
        print(f"最大迭代次数: {max_iterations}")
        print(f"距离阈值: {distance_threshold}")
    
    try:
        # 转换为open3d点云
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points_3d)
        
        # ========== 第1步：使用RANSAC拟合主平面 ==========
        if verbose:
            print("\n第1步：拟合主平面（去除不在平面上的离群点）")
        
        # RANSAC平面拟合
        plane_model, plane_inliers = pcd.segment_plane(distance_threshold=distance_threshold * 0.5,
                                                        ransac_n=3,
                                                        num_iterations=200)
        
        # 提取平面上的点
        plane_points = np.asarray(pcd.select_by_index(plane_inliers).points)
        
        if verbose:
            print(f"  平面模型: [{plane_model[0]:.4f}, {plane_model[1]:.4f}, {plane_model[2]:.4f}, {plane_model[3]:.4f}]")
            print(f"  平面内点: {len(plane_points)}/{len(points_3d)} 个 ({100*len(plane_points)/len(points_3d):.1f}%)")
        
        if len(plane_points) < 10:
            print("  错误：平面上点数太少")
            return None
        
        # ========== 第2步：将3D点投影到平面（转换为2D问题） ==========
        if verbose:
            print("\n第2步：将点投影到平面（3D → 2D）")
        
        # 提取平面参数: ax + by + cz + d = 0
        [a, b, c, d] = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)
        
        # 构建平面上的局部坐标系
        # 选择一个与法向量不平行的向量作为参考
        if abs(normal[2]) < 0.9:
            # 使用(1,0,-a/c)作为u轴
            if c != 0:
                u = np.array([1, 0, -a/c])
            else:
                u = np.array([1, 0, 0])
            u = u / np.linalg.norm(u)
        else:
            # 如果法向量接近垂直，使用(0,1,0)作为u轴
            u = np.array([0, 1, 0])
        
        # v轴垂直于u和法向量
        v = np.cross(normal, u)
        v = v / np.linalg.norm(v)
        
        # 重新调整u轴使其垂直于v（确保正交）
        u = np.cross(v, normal)
        u = u / np.linalg.norm(u)
        
        # 计算平面中心
        plane_center = np.mean(plane_points, axis=0)
        
        # 将所有点投影到2D平面
        points_2d = np.zeros((len(plane_points), 2))
        for i, p in enumerate(plane_points):
            points_2d[i, 0] = np.dot(p - plane_center, u)
            points_2d[i, 1] = np.dot(p - plane_center, v)
        
        if verbose:
            print(f"  2D点云范围: X=[{np.min(points_2d[:,0]):.4f}, {np.max(points_2d[:,0]):.4f}]")
            print(f"              Y=[{np.min(points_2d[:,1]):.4f}, {np.max(points_2d[:,1]):.4f}]")
        
        # ========== 第3步：RANSAC拟合矩形（在2D空间中） ==========
        if verbose:
            print("\n第3步：RANSAC拟合矩形（2D空间）")
        
        best_inliers = None
        best_rectangle = None
        best_score = 0
        best_corners_2d = None
        
        for iteration in range(max_iterations):
            # 随机采样4个点（矩形需要4个点来定义边界）
            sample_indices = np.random.choice(len(points_2d), 4, replace=False)
            sample = points_2d[sample_indices]
            
            try:
                # 计算这4个点的边界矩形
                x_min = np.min(sample[:, 0])
                x_max = np.max(sample[:, 0])
                y_min = np.min(sample[:, 1])
                y_max = np.max(sample[:, 1])
                
                # 检查矩形是否合理（长宽比不能太极端）
                rect_width = x_max - x_min
                rect_height = y_max - y_min
                aspect_ratio = max(rect_width, rect_height) / (min(rect_width, rect_height) + 1e-6)
                
                # 参照物的长宽比应该在1:1到3:1之间（长方形）
                if aspect_ratio > 5:
                    continue
                
                # 计算内点（点到矩形的距离小于阈值）
                # 点到矩形边界的最小距离
                dx = np.maximum(x_min - points_2d[:, 0], 0) + np.maximum(points_2d[:, 0] - x_max, 0)
                dy = np.maximum(y_min - points_2d[:, 1], 0) + np.maximum(points_2d[:, 1] - y_max, 0)
                distances = np.sqrt(dx**2 + dy**2)
                
                inlier_mask = distances <= distance_threshold
                inlier_count = np.sum(inlier_mask)
                
                # 更新最佳结果
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
                    
                    if verbose and iteration % 100 == 0:
                        print(f"  迭代 {iteration}: 找到 {inlier_count} 个内点 ({100*inlier_count/len(points_2d):.1f}%)")
                        
            except Exception as e:
                continue
        
        if best_rectangle is None:
            print("  错误：未找到有效的矩形")
            return None
        
        # ========== 第4步：使用内点重新精炼矩形 ==========
        if verbose:
            print("\n第4步：使用内点精炼矩形")
        
        # 提取内点
        inlier_points = points_2d[best_inliers]
        inlier_count = len(inlier_points)
        
        if verbose:
            print(f"  内点数量: {inlier_count}/{len(points_2d)} ({100*inlier_count/len(points_2d):.1f}%)")
        
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
            # 交换方向
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
        
        if verbose:
            print(f"\n{'='*50}")
            print("矩形拟合结果")
            print(f"{'='*50}")
            print(f"长度: {length:.6f}")
            print(f"宽度: {width:.6f}")
            print(f"长宽比: {length/width:.2f}")
            print(f"内点比例: {100*inlier_count/len(points_2d):.1f}%")
            print(f"矩形面积: {length * width:.6f}")
        
        # ========== 第5步：将2D角点转换回3D ==========
        corners_3d = []
        for corner_2d in corners_2d:
            point_3d = plane_center + corner_2d[0] * u + corner_2d[1] * v
            corners_3d.append(point_3d)
        
        corners_3d = np.array(corners_3d)
        
        # ========== 第6步：生成拟合后的矩形点云 ==========
        fitted_pcd = o3d.geometry.PointCloud()
        
        # 生成矩形点云（边界和内部）
        rectangle_points = []
        
        # 定义矩形的四条边
        edges = [
            (corners_3d[0], corners_3d[1]),  # 底边
            (corners_3d[1], corners_3d[2]),  # 右边
            (corners_3d[2], corners_3d[3]),  # 顶边
            (corners_3d[3], corners_3d[0])   # 左边
        ]
        
        # 每条边上生成50个点
        num_points_per_edge = 50
        for edge in edges:
            start, end = edge
            for i in range(num_points_per_edge):
                t = i / (num_points_per_edge - 1)
                point = start + t * (end - start)
                rectangle_points.append(point)
        
        # 生成矩形内部的网格点
        num_grid_x = 20
        num_grid_y = 20
        
        # 获取矩形的两条边向量
        edge1 = corners_3d[1] - corners_3d[0]
        edge2 = corners_3d[3] - corners_3d[0]
        
        for i in range(num_grid_x + 1):
            for j in range(num_grid_y + 1):
                t1 = i / num_grid_x
                t2 = j / num_grid_y
                point = corners_3d[0] + t1 * edge1 + t2 * edge2
                rectangle_points.append(point)
        
        rectangle_points = np.array(rectangle_points)
        fitted_pcd.points = o3d.utility.Vector3dVector(rectangle_points)
        
        return {
            'length_2d': length,
            'width_2d': width,
            'rectangle_area_2d': length * width,
            'centroid': plane_center,
            'plane_normal': normal,
            'u_axis': u,
            'v_axis': v,
            'corners_3d': corners_3d,
            'corners_2d': corners_2d,
            'fitted_point_cloud': fitted_pcd,
            'fitted_points': rectangle_points,
            'inlier_points': points_2d[best_inliers] if best_inliers is not None else points_2d,
            'projected_points': points_2d,
            'point_count': len(points_3d),
            'inlier_count': inlier_count,
            'inlier_ratio': inlier_count / len(points_2d)
        }
        
    except Exception as e:
        print(f"双重RANSAC矩形拟合过程中出错: {e}")
        traceback.print_exc()
        return None


def save_fitted_reference_cloud(reference_result, output_dir, original_filename):
    """
    保存拟合后的参照物点云
    """
    if reference_result is None or reference_result.get('fitted_point_cloud') is None:
        return None
    
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    reference_dir = os.path.join(output_dir, f"{base_name}_reference")
    os.makedirs(reference_dir, exist_ok=True)
    
    # 保存拟合的矩形点云
    fitted_path = os.path.join(reference_dir, "fitted_reference_rectangle.ply")
    o3d.io.write_point_cloud(fitted_path, reference_result['fitted_point_cloud'])
    print(f"拟合后的参照物点云已保存: {fitted_path}")
    
    # 保存矩形角点（作为点云）
    corners_pcd = o3d.geometry.PointCloud()
    corners_pcd.points = o3d.utility.Vector3dVector(reference_result['corners_3d'])
    corners_path = os.path.join(reference_dir, "rectangle_corners.ply")
    o3d.io.write_point_cloud(corners_path, corners_pcd)
    print(f"矩形角点已保存: {corners_path}")
    
    # 保存内点云（用于可视化）
    if reference_result.get('inlier_points') is not None:
        # 将2D内点转换回3D并保存
        inlier_points_3d = []
        plane_center = reference_result['centroid']
        u = reference_result['u_axis']
        v = reference_result['v_axis']
        for point_2d in reference_result['inlier_points']:
            point_3d = plane_center + point_2d[0] * u + point_2d[1] * v
            inlier_points_3d.append(point_3d)
        
        inlier_pcd = o3d.geometry.PointCloud()
        inlier_pcd.points = o3d.utility.Vector3dVector(np.array(inlier_points_3d))
        inlier_path = os.path.join(reference_dir, "rectangle_inliers.ply")
        o3d.io.write_point_cloud(inlier_path, inlier_pcd)
        print(f"矩形内点云已保存: {inlier_path}")
    
    return reference_dir


# ============================
# 修改：去除聚类的文件解析函数
# ============================

def parse_ply_file_detailed(file_path, enable_clustering=False, eps=0.02, min_points=10):
    """
    详细解析PLY文件，返回所有属性和按object_value值分类的点云
    修改：enable_clustering默认为False，不进行聚类
    """
    points_dict = {}
    all_points = []
    
    print(f"\n正在解析PLY文件: {file_path}")
    if enable_clustering:
        print(f"启用聚类滤除: eps={eps}, min_points={min_points}")
    else:
        print("不启用聚类，直接使用原始点云")
    
    try:
        with open(file_path, 'r') as f:
            # 读取头部
            header_lines = []
            while True:
                line = f.readline().strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # 获取顶点数量和属性
            vertex_count = 0
            properties = []
            
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                elif line.startswith('property'):
                    prop_parts = line.split()
                    if len(prop_parts) >= 3:
                        prop_type = prop_parts[1]
                        prop_name = prop_parts[2]
                        properties.append((prop_name, prop_type))
            
            # 找到x, y, z和object_value的索引
            x_idx = y_idx = z_idx = obj_idx = -1
            for i, (prop_name, _) in enumerate(properties):
                if prop_name == 'x':
                    x_idx = i
                elif prop_name == 'y':
                    y_idx = i
                elif prop_name == 'z':
                    z_idx = i
                elif prop_name == 'object_value':
                    obj_idx = i
            
            if -1 in [x_idx, y_idx, z_idx, obj_idx]:
                print("错误: 找不到必要的属性")
                return {}, np.array([])
            
            # 按object_value收集所有点
            raw_points_dict = {}
            for i in range(vertex_count):
                line = f.readline().strip()
                if not line:
                    continue
                
                values = line.split()
                try:
                    x = float(values[x_idx])
                    y = float(values[y_idx])
                    z = float(values[z_idx])
                    obj_val = int(values[obj_idx])
                    
                    if obj_val not in raw_points_dict:
                        raw_points_dict[obj_val] = []
                    raw_points_dict[obj_val].append([x, y, z])
                except:
                    continue
            
            # 如果启用聚类，则进行聚类滤除；否则直接使用原始点
            if enable_clustering:
                for obj_val, points_list in raw_points_dict.items():
                    if len(points_list) < min_points:
                        points_dict[obj_val] = np.array(points_list)
                        continue
                    
                    points = np.array(points_list)
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points)
                    
                    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=False))
                    
                    if len(labels) == 0 or np.max(labels) < 0:
                        points_dict[obj_val] = points
                        continue
                    
                    unique_labels, counts = np.unique(labels[labels >= 0], return_counts=True)
                    max_cluster_idx = unique_labels[np.argmax(counts)]
                    mask = labels == max_cluster_idx
                    filtered_points = points[mask]
                    
                    points_dict[obj_val] = filtered_points
                    
                    for point in filtered_points:
                        all_points.append([point[0], point[1], point[2], obj_val])
            else:
                # 不使用聚类，直接使用原始点云
                for obj_val, points_list in raw_points_dict.items():
                    points_dict[obj_val] = np.array(points_list)
                    for point in points_list:
                        all_points.append([point[0], point[1], point[2], obj_val])
            
            all_points = np.array(all_points) if all_points else np.array([])
            
            return points_dict, all_points
            
    except Exception as e:
        print(f"解析PLY文件时出错: {e}")
        return {}, np.array([])


def parse_binary_ply_for_y(file_path):
    """
    解析binary格式的PLY文件，只获取y轴坐标信息（用于株高计算）
    """
    print(f"\n正在解析辅助PLY文件: {file_path}")
    
    try:
        with open(file_path, 'rb') as f:
            # 读取头部
            header_lines = []
            while True:
                line = f.readline().decode('utf-8', errors='ignore').strip()
                header_lines.append(line)
                if line == 'end_header':
                    break
            
            # 获取顶点数量和属性
            vertex_count = 0
            properties = []
            
            for line in header_lines:
                if line.startswith('element vertex'):
                    vertex_count = int(line.split()[2])
                elif line.startswith('property'):
                    prop_parts = line.split()
                    if len(prop_parts) >= 3:
                        prop_type = prop_parts[1]
                        prop_name = prop_parts[2]
                        properties.append((prop_name, prop_type))
            
            # 找到y的索引
            y_idx = -1
            for i, (prop_name, _) in enumerate(properties):
                if prop_name == 'y':
                    y_idx = i
                    break
            
            if y_idx == -1:
                print("错误: 找不到y属性")
                return None, None, 0
            
            # 确定每个属性的大小
            type_sizes = {'double': 8, 'float': 4, 'int': 4, 'uint': 4,
                         'uchar': 1, 'char': 1, 'short': 2, 'ushort': 2}
            
            point_size = 0
            for _, prop_type in properties:
                base_type = prop_type.split('(')[0] if '(' in prop_type else prop_type
                point_size += type_sizes.get(base_type, 4)
            
            # 读取所有点的y坐标
            y_values = []
            for i in range(vertex_count):
                data = f.read(point_size)
                if len(data) < point_size:
                    break
                
                offset = 0
                for j, (_, prop_type) in enumerate(properties):
                    base_type = prop_type.split('(')[0] if '(' in prop_type else prop_type
                    size = type_sizes.get(base_type, 4)
                    
                    if j == y_idx:
                        if base_type in ['double', 'float64']:
                            fmt = '<d'
                        elif base_type in ['float', 'float32']:
                            fmt = '<f'
                        else:
                            fmt = '<f'
                        
                        try:
                            y_val = struct.unpack(fmt, data[offset:offset+size])[0]
                            if not (np.isnan(y_val) or np.isinf(y_val)):
                                y_values.append(y_val)
                        except:
                            pass
                        break
                    
                    offset += size
            
            if not y_values:
                return None, None, 0
            
            return np.min(y_values), np.max(y_values), len(y_values)
            
    except Exception as e:
        print(f"解析binary PLY文件时出错: {e}")
        return None, None, 0


def save_all_object_pointclouds(points_dict, output_dir, original_filename):
    """
    保存所有object的点云为PLY文件
    """
    print("\n" + "=" * 60)
    print("保存所有object的点云文件")
    print("=" * 60)
    
    base_name = os.path.splitext(os.path.basename(original_filename))[0]
    objects_dir = os.path.join(output_dir, f"{base_name}_objects")
    os.makedirs(objects_dir, exist_ok=True)
    
    saved_count = 0
    object_info = []
    
    for obj_val, points in points_dict.items():
        if len(points) == 0:
            continue
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        filename = os.path.join(objects_dir, f"object_{obj_val:03d}.ply")
        o3d.io.write_point_cloud(filename, pcd)
        
        centroid = np.mean(points, axis=0)
        y_min = np.min(points[:, 1])
        y_max = np.max(points[:, 1])
        
        object_info.append({
            'object_value': obj_val,
            'point_count': len(points),
            'y_min': y_min,
            'y_max': y_max,
            'centroid_x': centroid[0],
            'centroid_y': centroid[1],
            'centroid_z': centroid[2],
            'filename': filename
        })
        
        saved_count += 1
    
    if object_info:
        df = pd.DataFrame(object_info)
        csv_path = os.path.join(objects_dir, "objects_summary.csv")
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        
        print(f"\n已保存 {saved_count} 个object的点云文件到: {objects_dir}")
    
    return objects_dir, object_info


# ============================
# 改进的叶长计算函数（中轴线法）
# ============================
def calculate_leaf_length(points, n_sections=20):
    """
    计算叶片长度 - 使用中轴线法（更准确，适用于弯曲叶片）
    输入：叶片点云，分段数
    输出：叶片长度（点云空间）
    """
    if len(points) < 30:
        return 0.0
    
    try:
        # 计算主轴方向
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # PCA获取主轴
        U, S, Vt = np.linalg.svd(points_centered, full_matrices=False)
        length_axis = Vt[0]  # 第一主成分（长度方向）
        
        # 投影到长度轴
        projections = np.dot(points_centered, length_axis)
        
        # 分段数调整（确保不要太少）
        n_sections = min(n_sections, len(points) // 10)
        if n_sections < 2:
            # 点太少，使用简单投影法
            return np.max(projections) - np.min(projections)
        
        # 创建分段
        min_proj = np.min(projections)
        max_proj = np.max(projections)
        bins = np.linspace(min_proj, max_proj, n_sections + 1)
        
        # 收集中轴点
        axis_points = []
        
        for i in range(n_sections):
            # 获取当前段的点
            if i == n_sections - 1:
                mask = (projections >= bins[i]) & (projections <= bins[i+1])
            else:
                mask = (projections >= bins[i]) & (projections < bins[i+1])
            
            section_points = points[mask]
            
            if len(section_points) >= 3:
                # 计算截面中心（中轴点）
                section_center = np.mean(section_points, axis=0)
                axis_points.append(section_center)
        
        if len(axis_points) < 2:
            # 分段失败，使用简单投影法
            return np.max(projections) - np.min(projections)
        
        # 计算中轴线总长度
        axis_points = np.array(axis_points)
        axis_length = 0.0
        for i in range(len(axis_points) - 1):
            axis_length += np.linalg.norm(axis_points[i+1] - axis_points[i])
        
        return axis_length
        
    except Exception as e:
        print(f"  长度计算错误: {e}")
        return 0.0


# ============================
# 改进的叶宽计算函数（分段宽度法）
# ============================
def calculate_leaf_width(points, n_sections=15):
    """
    计算叶片宽度 - 分段宽度法（更准确，输出平均宽度和最大宽度）
    输入：叶片点云，分段数
    输出：平均宽度, 最大宽度
    """
    if len(points) < 30:
        return 0.0, 0.0
    
    try:
        # 计算主轴方向
        centroid = np.mean(points, axis=0)
        points_centered = points - centroid
        
        # PCA获取主轴
        U, S, Vt = np.linalg.svd(points_centered, full_matrices=False)
        length_axis = Vt[0]  # 第一主成分（长度方向）
        width_axis = Vt[1]   # 第二主成分（宽度方向）
        
        # 投影到长度轴
        length_projections = np.dot(points_centered, length_axis)
        
        # 分段数调整
        n_sections = min(n_sections, len(points) // 10)
        if n_sections < 2:
            # 点太少，使用整体宽度
            width_projections = np.dot(points_centered, width_axis)
            overall_width = np.max(width_projections) - np.min(width_projections)
            return overall_width, overall_width
        
        # 创建分段
        min_proj = np.min(length_projections)
        max_proj = np.max(length_projections)
        bins = np.linspace(min_proj, max_proj, n_sections + 1)
        
        # 存储每段的宽度
        section_widths = []
        
        for i in range(n_sections):
            # 获取当前段的点
            if i == n_sections - 1:
                mask = (length_projections >= bins[i]) & (length_projections <= bins[i+1])
            else:
                mask = (length_projections >= bins[i]) & (length_projections < bins[i+1])
            
            section_points = points[mask]
            
            if len(section_points) >= 3:
                # 计算该段的宽度
                section_centered = section_points - centroid
                width_projections = np.dot(section_centered, width_axis)
                section_width = np.max(width_projections) - np.min(width_projections)
                section_widths.append(section_width)
        
        if not section_widths:
            # 没有有效分段，使用整体宽度
            width_projections = np.dot(points_centered, width_axis)
            overall_width = np.max(width_projections) - np.min(width_projections)
            return overall_width, overall_width
        
        # 计算宽度统计
        avg_width = np.mean(section_widths)
        max_width = np.max(section_widths)
        
        return avg_width, max_width
        
    except Exception as e:
        print(f"  宽度计算错误: {e}")
        return 0.0, 0.0


def calculate_leaf_area_alpha_shape(points, alpha=0.015):
    """
    计算叶片面积
    """
    if len(points) < 30:
        return 0.0
    
    try:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        if len(points) > 2000:
            pcd = pcd.voxel_down_sample(voxel_size=0.003)
        
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            pcd, alpha=alpha)
        
        if len(mesh.triangles) > 0:
            return mesh.get_surface_area()
        else:
            return 0.0
            
    except Exception as e:
        return 0.0


def calibrate_leaf_measurements(reference_result, leaf_points, leaf_id, file_name):
    """
    使用参照物标定计算叶片真实尺寸
    修改：使用双重RANSAC拟合结果中的长度和面积进行标定
    """
    # 如果没有参照物结果，只返回3D测量值
    if reference_result is None:
        leaf_length_3d = calculate_leaf_length(leaf_points)
        leaf_avg_width_3d, leaf_max_width_3d = calculate_leaf_width(leaf_points)
        leaf_area_3d = calculate_leaf_area_alpha_shape(leaf_points)
        
        results = {
            '文件名': os.path.basename(file_name),
            '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            '叶片ID': leaf_id,
            '叶片点数': len(leaf_points),
            '参照物3D长度': None,
            '参照物拟合面积': None,
            '参照物真实长度': 9.5,
            '参照物真实面积': 47.5,
            '长度标定系数': None,
            '面积标定系数': None,
            '3D长度(中轴线法)': leaf_length_3d,
            '3D平均宽度(分段法)': leaf_avg_width_3d,
            '3D最大宽度(分段法)': leaf_max_width_3d,
            '3D面积': leaf_area_3d,
            '真实长度': None,
            '真实平均宽度': None,
            '真实最大宽度': None,
            '真实面积': None,
            '处理状态': '缺失参照物(220)'
        }
        return results
    
    REFERENCE_REAL_LENGTH = 9.5
    REFERENCE_REAL_AREA = 47.5
    
    # 使用双重RANSAC拟合得到的长度和面积
    REFERENCE_3D_LENGTH = reference_result['length_2d']
    REFERENCE_AREA = reference_result['rectangle_area_2d']
    
    length_scale = REFERENCE_REAL_LENGTH / REFERENCE_3D_LENGTH
    area_scale = REFERENCE_REAL_AREA / REFERENCE_AREA
    
    leaf_length_3d = calculate_leaf_length(leaf_points)
    leaf_length_real = leaf_length_3d * length_scale
    
    leaf_avg_width_3d, leaf_max_width_3d = calculate_leaf_width(leaf_points)
    leaf_avg_width_real = leaf_avg_width_3d * length_scale
    leaf_max_width_real = leaf_max_width_3d * length_scale
    
    leaf_area_3d = calculate_leaf_area_alpha_shape(leaf_points)
    leaf_area_real = leaf_area_3d * area_scale
    
    results = {
        '文件名': os.path.basename(file_name),
        '处理时间': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        '叶片ID': leaf_id,
        '叶片点数': len(leaf_points),
        '参照物3D长度': REFERENCE_3D_LENGTH,
        '参照物拟合面积': REFERENCE_AREA,
        '参照物真实长度': REFERENCE_REAL_LENGTH,
        '参照物真实面积': REFERENCE_REAL_AREA,
        '长度标定系数': length_scale,
        '面积标定系数': area_scale,
        '3D长度(中轴线法)': leaf_length_3d,
        '3D平均宽度(分段法)': leaf_avg_width_3d,
        '3D最大宽度(分段法)': leaf_max_width_3d,
        '3D面积': leaf_area_3d,
        '真实长度': leaf_length_real,
        '真实平均宽度': leaf_avg_width_real,
        '真实最大宽度': leaf_max_width_real,
        '真实面积': leaf_area_real,
        '处理状态': '成功'
    }
    
    return results


# ============================
# 分析object值≥300的点云与穗部的关系（修改版）
# ============================

def analyze_high_object_values(points_dict, stem_points, selected_ear_object, ear_points=None):
    """
    分析object值≥300的点云与穗部的关系
    
    逻辑：
    1. 找到大于j的所有i中最小的那个i → 得到 object A
    2. 找到所有小于 i_A 的i中最大的那个i → 得到 object C
    
    参数:
        points_dict: 所有object的点云字典
        stem_points: 主干点云（用于计算距离）
        selected_ear_object: 用于求穗位高的穗的object值（222-226中的一个）
        ear_points: 该穗的点云（可选，如果不提供则从points_dict中获取）
    
    返回:
        analysis_results: 包含分析结果的字典
        high_obj_info: 包含每个≥300的object的信息列表
    """
    print("\n" + "=" * 60)
    print("分析object值≥300的点云")
    print("=" * 60)
    
    # 获取穗部点云
    if ear_points is None and selected_ear_object is not None:
        ear_points = points_dict.get(selected_ear_object, None)
    
    if ear_points is None or len(ear_points) == 0:
        print("⚠️ 无法获取穗部点云，跳过分析")
        return None, None
    
    # 计算穗部点云中y轴最小的10%的点的y轴中位数j
    y_values_ear = ear_points[:, 1]
    n_ear = len(ear_points)
    k_ear = max(1, int(n_ear * 0.1))
    
    # 按y值排序，取最小的10%
    sorted_indices_ear = np.argsort(y_values_ear)
    bottom_10_indices = sorted_indices_ear[:k_ear]
    bottom_10_y = y_values_ear[bottom_10_indices]
    j = np.median(bottom_10_y)
    
    print(f"\n穗部object值: {selected_ear_object}")
    print(f"穗部点云总数: {len(ear_points)}")
    print(f"y轴最小的10%点数: {k_ear}")
    print(f"这些点的y轴中位数 j: {j:.6f}")
    
    # 查找object值≥300的点云
    high_object_values = [obj_val for obj_val in points_dict.keys() if obj_val >= 300]
    
    if not high_object_values:
        print("\n未找到object值≥300的点云")
        return None, None
    
    print(f"\n找到 {len(high_object_values)} 个object值≥300的点云: {sorted(high_object_values)}")
    
    # 对每个object值≥300的点云，计算y轴坐标最大的10%的点的y轴中位数i
    high_obj_info = []
    
    for obj_val in sorted(high_object_values):
        points = points_dict[obj_val]
        if len(points) == 0:
            continue
        
        print(f"\n处理 object={obj_val}, 点数={len(points)}")
        
        # 取y轴坐标最大的10%的点
        y_values_all = points[:, 1]  # 获取所有点的y坐标
        k = max(1, int(len(points) * 0.1))  # 10%的点数
        
        # 按y值从大到小排序，取最大的10%
        sorted_indices = np.argsort(y_values_all)[::-1]  # 降序排列
        top_10_indices = sorted_indices[:k]  # 取最大的10%
        top_10_points = points[top_10_indices]  # 获取这些点的坐标
        
        # 计算这些点的y轴中位数i
        y_values_top = top_10_points[:, 1]
        i = np.median(y_values_top)
        
        print(f"  y轴最大的10%点数: {len(top_10_points)}")
        print(f"  这些点的y轴中位数 i: {i:.6f}")
        
        high_obj_info.append({
            'object_value': obj_val,
            'point_count': len(points),
            'top_points_count': len(top_10_points),
            'y_median_i': i,
            'y_min': np.min(points[:, 1]),
            'y_max': np.max(points[:, 1])
        })
    
    if not high_obj_info:
        print("\n未找到有效的object值≥300的点云信息")
        return None, None
    
    # 第一步：找到大于j的所有i中最小的那个i
    greater_than_j = [info for info in high_obj_info if info['y_median_i'] > j]
    if greater_than_j:
        min_greater = min(greater_than_j, key=lambda x: x['y_median_i'])
        min_greater_obj = min_greater['object_value']
        min_greater_i = min_greater['y_median_i']
        print(f"\n步骤1: 大于j({j:.6f})的所有i中最小的i: {min_greater_i:.6f} (object={min_greater_obj})")
    else:
        min_greater_obj = None
        min_greater_i = None
        print(f"\n步骤1: 没有找到大于j({j:.6f})的i值")
    
    # 第二步：找到所有小于 min_greater_i 的i中最大的那个i
    max_less_obj = None
    max_less_i = None
    
    if min_greater_i is not None:
        less_than_min_greater = [info for info in high_obj_info 
                                  if info['y_median_i'] < min_greater_i]
        
        if less_than_min_greater:
            max_less = max(less_than_min_greater, key=lambda x: x['y_median_i'])
            max_less_obj = max_less['object_value']
            max_less_i = max_less['y_median_i']
            print(f"\n步骤2: 所有小于 {min_greater_i:.6f} 的i中最大的i: {max_less_i:.6f} (object={max_less_obj})")
        else:
            print(f"\n步骤2: 没有找到小于 {min_greater_i:.6f} 的i值")
    else:
        print(f"\n步骤2: 无法执行（步骤1未找到有效值）")
    
    analysis_results = {
        'ear_object_value': selected_ear_object,
        'ear_point_count': len(ear_points),
        'ear_bottom_10_percent_median_y': j,
        'greater_than_j_min_object': min_greater_obj,
        'greater_than_j_min_median_y': min_greater_i,
        'less_than_min_greater_max_object': max_less_obj,
        'less_than_min_greater_max_median_y': max_less_i,
        'high_object_info': high_obj_info
    }
    
    return analysis_results, high_obj_info


# ============================
# 叶片夹角计算函数
# ============================

def compute_pca_direction(points):
    """
    计算点云的PCA主方向
    """
    if len(points) < 10:
        return None
    
    centroid = np.mean(points, axis=0)
    centered = points - centroid
    
    pca = PCA(n_components=3)
    pca.fit(centered)
    
    return pca.components_[0]

def compute_angle_between_vectors(v1, v2):
    """
    计算两个向量之间的夹角（度）- 取锐角
    """
    v1_norm = v1 / np.linalg.norm(v1)
    v2_norm = v2 / np.linalg.norm(v2)
    
    cos_angle = np.abs(np.dot(v1_norm, v2_norm))
    cos_angle = np.clip(cos_angle, -1, 1)
    angle = np.arccos(cos_angle) * 180 / np.pi
    
    return angle

def find_nearest_leaf_points(leaf_points, stem_points, top_percent=30):
    """
    找到叶片点云中离主干最近的top_percent%的点
    """
    if len(leaf_points) == 0 or len(stem_points) == 0:
        return leaf_points
    
    stem_tree = KDTree(stem_points)
    distances, _ = stem_tree.query(leaf_points)
    
    k = max(1, int(len(leaf_points) * top_percent / 100))
    indices = np.argsort(distances)[:k]
    
    return leaf_points[indices]

def calculate_leaf_angle(leaf_points, stem_direction, stem_points=None, use_nearest_percent=30):
    """
    计算单个叶片与主干的夹角
    """
    if len(leaf_points) < 10:
        return None, None
    
    # 如果提供了主干点云，只使用离主干最近的点
    if stem_points is not None and use_nearest_percent > 0:
        leaf_points_used = find_nearest_leaf_points(leaf_points, stem_points, use_nearest_percent)
        if len(leaf_points_used) < 10:
            leaf_points_used = leaf_points
    else:
        leaf_points_used = leaf_points
    
    # 计算叶片方向
    leaf_direction = compute_pca_direction(leaf_points_used)
    
    if leaf_direction is None:
        return None, None
    
    # 计算夹角
    angle = compute_angle_between_vectors(stem_direction, leaf_direction)
    
    return angle, leaf_direction

def get_stem_direction_from_cloud(stem_ply_path):
    """
    从主干点云计算主干方向
    返回: stem_direction, stem_points, vertical_angle
    """
    print(f"\n📐 读取主干点云: {stem_ply_path}")
    
    # 读取主干点云
    stem_pcd = o3d.io.read_point_cloud(stem_ply_path)
    stem_points = np.asarray(stem_pcd.points)
    
    if len(stem_points) < 10:
        print("❌ 主干点数太少")
        return None, None, None
    
    # PCA计算主干方向
    stem_direction = compute_pca_direction(stem_points)
    
    if stem_direction is None:
        return None, None, None
    
    # 确保方向向上（Y轴正方向）
    if stem_direction[1] < 0:
        stem_direction = -stem_direction
    
    # 计算与垂直方向（Y轴）的夹角
    vertical_angle = np.arccos(np.abs(stem_direction[1])) * 180 / np.pi
    
    print(f"主干方向向量: [{stem_direction[0]:.4f}, {stem_direction[1]:.4f}, {stem_direction[2]:.4f}]")
    print(f"主干与垂直方向（Y轴）夹角: {vertical_angle:.2f}°")
    
    return stem_direction, stem_points, vertical_angle

# ============================
# 修改：新的主干提取函数（使用外部命令行工具）
# ============================

def get_python_executable():
    """
    获取当前运行的Python解释器路径
    """
    return sys.executable


def find_train_point_cloud(folder_path):
    """
    在数据文件夹下查找 train/point_cloud_main_clusters_merged.ply
    """
    train_ply = os.path.join(folder_path, "output_*", "train", "point_cloud_main_clusters_merged.ply")
    import glob
    files = glob.glob(train_ply)
    if files:
        return files[0]
    return None


def find_hybrid_ply(folder_path):
    """
    在数据文件夹下查找 hybrid_filtered_final.ply
    位置：数据文件夹/output_*/hybrid_filtered_final.ply
    """
    folder = Path(folder_path)
    
    # 查找所有 output_* 子目录下的 hybrid_filtered_final.ply
    hybrid_files = list(folder.glob("output_*/hybrid_filtered_final.ply"))
    
    if not hybrid_files:
        print(f"❌ 在 {folder_path} 中未找到 output_*/hybrid_filtered_final.ply")
        return None
    
    return str(hybrid_files[0])



def find_main_ply_files(folder_path):
    """
    在文件夹中自动查找所需的两个PLY文件
    - point_cloud_main_clusters_merged.ply
    - hybrid_filtered_final.ply（用于株高计算）
    """
    folder = Path(folder_path)
    
    print(f"\n正在搜索PLY文件: {folder_path}")
    
    # 查找 point_cloud_main_clusters_merged.ply
    main_ply_files = list(folder.rglob("point_cloud_main_clusters_merged.ply"))
    
    # 查找 hybrid_filtered_final.ply
    hybrid_ply_files = list(folder.rglob("hybrid_filtered_final.ply"))
    
    if not main_ply_files:
        print(f"❌ 错误: 未找到 point_cloud_main_clusters_merged.ply")
        return None, None, False
    
    if not hybrid_ply_files:
        print(f"❌ 错误: 未找到 hybrid_filtered_final.ply")
        return None, None, False
    
    main_ply = str(main_ply_files[0])
    hybrid_ply = str(hybrid_ply_files[0])
    
    print(f"找到主文件: {main_ply}")
    print(f"找到辅助文件: {hybrid_ply}")
    
    return main_ply, hybrid_ply, True



# def run_stem_extraction(folder_path, radius=0.15):
#     """
#     运行主干提取脚本（两步法）
#     步骤1：使用圆柱提取方法生成 stem.ply
#     步骤2：使用SLBC方法处理 stem.ply 生成 contracted_points_interpolated.ply
    
#     输入：
#         folder_path: 数据文件夹路径（如 /datashare/.../91227-1_frames）
#         radius: 圆柱半径（默认0.15）
    
#     输出：
#         最终的插值主干点云路径
#     """
#     print("\n" + "=" * 60)
#     print("第一步：运行圆柱形主干提取")
#     print("=" * 60)
    
#     # 查找必要的文件
#     train_ply = find_train_point_cloud(folder_path)
#     hybrid_ply = find_hybrid_ply(folder_path)
    
#     if not train_ply:
#         print(f"❌ 错误: 在文件夹 {folder_path} 中未找到 train/point_cloud_main_clusters_merged.ply")
#         return None
    
#     if not hybrid_ply:
#         print(f"❌ 错误: 在文件夹 {folder_path} 中未找到 hybrid_filtered_final.ply")
#         return None
    
#     print(f"找到叶片点云: {train_ply}")
#     print(f"找到完整点云: {hybrid_ply}")
    
#     # 输出文件路径
#     stem_ply = os.path.join(folder_path, "stem.ply")
    
#     # 主干提取脚本路径
#     stem_script = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型/找到主干方向.py"
    
#     if not os.path.exists(stem_script):
#         print(f"❌ 错误: 找不到主干提取脚本: {stem_script}")
#         return None
    
#     # 获取当前Python解释器路径
#     python_exe = get_python_executable()
#     print(f"使用Python解释器: {python_exe}")
    
#     # 运行第一步：圆柱形主干提取
#     print(f"\n运行圆柱形主干提取...")
#     print(f"输入叶片点云: {train_ply}")
#     print(f"输入完整点云: {hybrid_ply}")
#     print(f"输出主干点云: {stem_ply}")
#     print(f"圆柱半径: {radius}")
    
#     try:
#         result = subprocess.run(
#             [python_exe, stem_script, train_ply, hybrid_ply, stem_ply, "--radius", str(radius)],
#             capture_output=True, text=True, check=True
#         )
#         print(result.stdout)
#         if result.stderr:
#             print("警告输出:", result.stderr)
            
#     except subprocess.CalledProcessError as e:
#         print(f"❌ 主干提取脚本运行失败: {e}")
#         print("标准输出:", e.stdout)
#         print("错误输出:", e.stderr)
#         return None
    
#     # 检查输出文件
#     if not os.path.exists(stem_ply):
#         print(f"❌ 主干提取失败: 未找到输出文件 {stem_ply}")
#         return None
    
#     print(f"✅ 主干提取成功: {stem_ply}")
    
#     # ========== 第二步：运行SLBC处理 ==========
#     print("\n" + "=" * 60)
#     print("第二步：运行SLBC骨架提取（后处理）")
#     print("=" * 60)
    
#     # SLBC脚本路径
#     slbc_script = "/datashare/dir_liusha/pc-skeletor/pc-skeletor-main/example_tree.py"
    
#     if not os.path.exists(slbc_script):
#         print(f"❌ 错误: 找不到SLBC脚本: {slbc_script}")
#         return None
    
#     # 创建临时修改版的脚本
#     temp_script = os.path.join(folder_path, "temp_slbc_script.py")
    
#     try:
#         with open(slbc_script, 'r') as f:
#             script_content = f.read()
        
#         # 修改输入路径为生成的stem.ply
#         modified_content = script_content.replace(
#             'input_ply_path = "/datashare/dir_liusha/xibeinonglin/样本数据/91227-3_frames/stem.ply"',
#             f'input_ply_path = "{stem_ply}"'
#         )
        
#         with open(temp_script, 'w') as f:
#             f.write(modified_content)
        
#         print(f"运行SLBC脚本...")
#         print(f"输入: {stem_ply}")
        
#         # 运行SLBC脚本
#         result = subprocess.run([python_exe, temp_script], 
#                                capture_output=True, text=True, check=True)
#         print(result.stdout)
#         if result.stderr:
#             print("警告输出:", result.stderr)
        
#     except subprocess.CalledProcessError as e:
#         print(f"❌ SLBC脚本运行失败: {e}")
#         print("标准输出:", e.stdout)
#         print("错误输出:", e.stderr)
#         return None
#     except Exception as e:
#         print(f"❌ 发生错误: {e}")
#         traceback.print_exc()
#         return None
#     finally:
#         # 清理临时文件
#         if os.path.exists(temp_script):
#             os.remove(temp_script)
#             print(f"已清理临时脚本: {temp_script}")
    
#     # SLBC脚本会生成多个文件，我们需要的是插值后的点云
#     interpolated_path = os.path.join(folder_path, "contracted_points_interpolated.ply")
    
#     if os.path.exists(interpolated_path):
#         print(f"✅ SLBC处理成功: {interpolated_path}")
#         return interpolated_path
#     else:
#         # 如果没有插值文件，尝试使用清理后的文件
#         cleaned_path = os.path.join(folder_path, "contracted_points_cleaned.ply")
#         if os.path.exists(cleaned_path):
#             print(f"⚠️ 未找到插值文件，使用清理后的点云: {cleaned_path}")
#             return cleaned_path
        
#         contracted_path = os.path.join(folder_path, "contracted_points.ply")
#         if os.path.exists(contracted_path):
#             print(f"⚠️ 未找到插值文件，使用收缩点云: {contracted_path}")
#             return contracted_path
        
#         print(f"❌ SLBC处理失败: 未找到输出文件")
#         return None
def run_stem_extraction(folder_path, radius=0.15):
    """
    运行主干提取脚本（两步法）
    """
    print("\n" + "=" * 60)
    print("第一步：运行圆柱形主干提取")
    print("=" * 60)
    
    # 查找必要的文件
    train_ply = find_train_point_cloud(folder_path)
    hybrid_ply = find_hybrid_ply(folder_path)
    
    if not train_ply:
        print(f"❌ 错误: 在文件夹 {folder_path} 中未找到 train/point_cloud_main_clusters_merged.ply")
        return None
    
    if not hybrid_ply:
        print(f"❌ 错误: 在文件夹 {folder_path} 中未找到 hybrid_filtered_final.ply")
        return None
    
    print(f"找到叶片点云: {train_ply}")
    print(f"找到完整点云: {hybrid_ply}")
    
    # 输出文件路径
    stem_ply = os.path.join(folder_path, "stem.ply")
    
    # 主干提取脚本路径
    stem_script = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型/找到主干方向.py"
    
    if not os.path.exists(stem_script):
        print(f"❌ 错误: 找不到主干提取脚本: {stem_script}")
        return None
    
    # 获取当前Python解释器路径
    python_exe = get_python_executable()
    print(f"使用Python解释器: {python_exe}")
    
    # 运行第一步：圆柱形主干提取
    print(f"\n运行圆柱形主干提取...")
    print(f"输入叶片点云: {train_ply}")
    print(f"输入完整点云: {hybrid_ply}")
    print(f"输出主干点云: {stem_ply}")
    print(f"圆柱半径: {radius}")
    
    try:
        result = subprocess.run(
            [python_exe, stem_script, train_ply, hybrid_ply, stem_ply, "--radius", str(radius)],
            capture_output=True, text=True, check=True
        )
        print(result.stdout)
        if result.stderr:
            print("警告输出:", result.stderr)
            
    except subprocess.CalledProcessError as e:
        print(f"❌ 主干提取脚本运行失败: {e}")
        print("标准输出:", e.stdout)
        print("错误输出:", e.stderr)
        return None
    
    # 检查输出文件
    if not os.path.exists(stem_ply):
        print(f"❌ 主干提取失败: 未找到输出文件 {stem_ply}")
        return None
    
    print(f"✅ 主干提取成功: {stem_ply}")
    
    # ========== 第二步：运行SLBC处理 ==========
    print("\n" + "=" * 60)
    print("第二步：运行SLBC骨架提取（后处理）")
    print("=" * 60)
    
    # SLBC脚本路径
    slbc_script = "/datashare/dir_liusha/pc-skeletor/pc-skeletor-main/example_tree.py"
    
    if not os.path.exists(slbc_script):
        print(f"❌ 错误: 找不到SLBC脚本: {slbc_script}")
        return stem_ply  # 返回圆柱形主干提取的结果
    
    # 创建临时修改版的脚本
    temp_script = os.path.join(folder_path, "temp_slbc_script.py")
    
    try:
        with open(slbc_script, 'r') as f:
            script_content = f.read()
        
        # 获取 pc_skeletor 的路径
        pc_skeletor_path = "/datashare/dir_liusha/pc-skeletor/pc-skeletor-main"
        
        # 修改输入路径为生成的stem.ply
        modified_content = script_content.replace(
            'input_ply_path = "/datashare/dir_liusha/xibeinonglin/样本数据/91227-3_frames/stem.ply"',
            f'input_ply_path = "{stem_ply}"'
        )
        
        # 在脚本开头添加路径设置，确保能找到 pc_skeletor 模块
        path_setup = f'''
import sys
import os

# 添加 pc_skeletor 路径
pc_skeletor_path = "{pc_skeletor_path}"
if pc_skeletor_path not in sys.path:
    sys.path.insert(0, pc_skeletor_path)
    print(f"✅ 已添加 pc_skeletor 路径: {{pc_skeletor_path}}")

# 添加当前目录
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

print(f"Python 路径: {{sys.path[:3]}}")
print(f"当前工作目录: {{os.getcwd()}}")
print(f"stem.ply 存在: {{os.path.exists('{stem_ply}')}}")

# 尝试导入 pc_skeletor
try:
    from pc_skeletor import SLBC
    print("✅ pc_skeletor 导入成功")
except ImportError as e:
    print(f"❌ pc_skeletor 导入失败: {{e}}")
    print("尝试列出 pc_skeletor 路径内容:")
    if os.path.exists(pc_skeletor_path):
        print(f"  {pc_skeletor_path} 存在")
        files = os.listdir(pc_skeletor_path)
        print(f"  文件列表: {{files[:10]}}")
    else:
        print(f"  {pc_skeletor_path} 不存在")
    raise

'''
        
        # 找到原始脚本中的 import 语句位置，插入路径设置
        # 在文件开头插入路径设置
        modified_content = path_setup + modified_content
        
        with open(temp_script, 'w') as f:
            f.write(modified_content)
        
        print(f"运行SLBC脚本...")
        print(f"输入: {stem_ply}")
        print(f"pc_skeletor路径: {pc_skeletor_path}")
        
        # 运行SLBC脚本
        result = subprocess.run([python_exe, temp_script], 
                               capture_output=True, text=True, check=True)
        print(result.stdout)
        if result.stderr:
            print("警告输出:", result.stderr)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ SLBC脚本运行失败: {e}")
        print("标准输出:", e.stdout)
        print("错误输出:", e.stderr)
        # 如果SLBC失败，返回圆柱形主干提取的结果
        print("⚠️ 将使用圆柱形主干提取的结果继续处理")
        return stem_ply
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        traceback.print_exc()
        return stem_ply
    finally:
        # 清理临时文件
        if os.path.exists(temp_script):
            os.remove(temp_script)
            print(f"已清理临时脚本: {temp_script}")
    
    # SLBC脚本会生成多个文件，我们需要的是插值后的点云
    interpolated_path = os.path.join(folder_path, "contracted_points_interpolated.ply")
    
    if os.path.exists(interpolated_path):
        print(f"✅ SLBC处理成功: {interpolated_path}")
        return interpolated_path
    else:
        # 如果没有插值文件，尝试使用清理后的文件
        cleaned_path = os.path.join(folder_path, "contracted_points_cleaned.ply")
        if os.path.exists(cleaned_path):
            print(f"⚠️ 未找到插值文件，使用清理后的点云: {cleaned_path}")
            return cleaned_path
        
        contracted_path = os.path.join(folder_path, "contracted_points.ply")
        if os.path.exists(contracted_path):
            print(f"⚠️ 未找到插值文件，使用收缩点云: {contracted_path}")
            return contracted_path
        
        print(f"⚠️ SLBC处理未生成文件，使用圆柱形主干提取结果: {stem_ply}")
        return stem_ply


# ============================
# 修改后的主处理函数
# ============================

def process_single_data_folder(data_folder_path, eps=0.02, min_points=10, 
                               use_nearest_percent=30, skip_skeleton=False,
                               ransac_max_iterations=500, ransac_distance_threshold=0.01,
                               stem_radius=0.15):
    """
    处理单个数据文件夹
    参数:
        data_folder_path: 数据文件夹路径
        stem_radius: 圆柱形主干提取的半径
        ransac_max_iterations: RANSAC矩形拟合最大迭代次数
        ransac_distance_threshold: RANSAC矩形拟合距离阈值
    """
    print("\n" + "=" * 80)
    print(f"处理数据文件夹: {data_folder_path}")
    print("=" * 80)
    
    folder_name = os.path.basename(os.path.normpath(data_folder_path))
    
    # ========== 运行主干提取（新的两步法） ==========
    if not skip_skeleton:
        print("\n开始主干提取流程...")
        stem_cloud_path = run_stem_extraction(data_folder_path, radius=stem_radius)
        if stem_cloud_path is None:
            print("❌ 主干提取失败，无法继续")
            return None
    else:
        # 如果跳过，查找已有的文件
        stem_cloud_path = os.path.join(data_folder_path, "contracted_points_interpolated.ply")
        if not os.path.exists(stem_cloud_path):
            stem_cloud_path = os.path.join(data_folder_path, "contracted_points_cleaned.ply")
        if not os.path.exists(stem_cloud_path):
            stem_cloud_path = os.path.join(data_folder_path, "contracted_points.ply")
        
        if not os.path.exists(stem_cloud_path):
            print(f"❌ 找不到主干点云文件: {stem_cloud_path}")
            return None
        
        print(f"跳过主干提取，使用已有文件: {stem_cloud_path}")
    
    # ========== 计算主干方向 ==========
    stem_direction = None
    stem_points = None
    stem_vertical_angle = None  # 新增

    if stem_cloud_path and os.path.exists(stem_cloud_path):
        # stem_direction, stem_points = get_stem_direction_from_cloud(stem_cloud_path)
        stem_direction, stem_points, stem_vertical_angle = get_stem_direction_from_cloud(stem_cloud_path)
    else:
        print("⚠️ 警告: 无法获取主干点云，叶片夹角将无法计算")

    if stem_direction is None:
        print("⚠️ 警告: 无法计算主干方向，叶片夹角将无法计算")
    
    # ========== 查找表型分析所需的文件 ==========
    main_ply, hybrid_ply, success = find_main_ply_files(data_folder_path)
    
    if not success:
        print(f"❌ 错误: 在文件夹 {data_folder_path} 中未找到所需的表型分析文件")
        return None
    
    # 创建输出目录（在当前目录下，以文件夹名命名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"results_{folder_name}_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n表型分析结果将保存到: {output_dir}")
    
    # 解析主点云文件（修改：不启用聚类）
    points_dict, all_points = parse_ply_file_detailed(
        main_ply, enable_clustering=False, eps=eps, min_points=min_points
    )
    
    if not points_dict:
        print("错误: 无法解析点云文件")
        return None
    
    # 保存所有object的点云
    objects_dir, object_info = save_all_object_pointclouds(points_dict, output_dir, main_ply)
    
    # 解析辅助文件获取y轴信息（用于株高和穗位高）
    binary_y_min, binary_y_max, binary_point_count = parse_binary_ply_for_y(hybrid_ply)
    
    if binary_y_min is None or binary_y_max is None:
        print("错误: 无法解析辅助文件")
        return None
    
    # ========== 提取参照物（使用双重RANSAC矩形拟合） ==========
    reference_result = None
    missing_reference = False
    
    if 220 not in points_dict:
        print("⚠️ 警告: 点云中不存在object_value=220的参照物，叶片尺寸将无法标定")
        missing_reference = True
    else:
        reference_points = points_dict[220]
        # 使用双重RANSAC矩形拟合
        reference_result = fit_rectangle_ransac(
            reference_points,
            max_iterations=ransac_max_iterations,
            distance_threshold=ransac_distance_threshold,
            verbose=True
        )
        if reference_result is None:
            print("⚠️ 警告: 参照物双重RANSAC矩形拟合失败，叶片尺寸将无法标定")
            missing_reference = True
        else:
            # 保存拟合后的参照物点云
            save_fitted_reference_cloud(reference_result, output_dir, main_ply)
    
    # 计算标定系数（仅在参照物存在时计算）
    length_scale = None
    if reference_result is not None:
        REFERENCE_REAL_LENGTH = 9.5
        length_scale = REFERENCE_REAL_LENGTH / reference_result['length_2d']
    else:
        length_scale = None
    
    # ========== 计算株高 ==========
    plant_base_y = None
    has_base_221 = 221 in points_dict
    
    if has_base_221:
        plant_base_y = np.min(points_dict[221][:, 1])
        print(f"使用object=221作为基部: y_min={plant_base_y:.6f}")
    else:
        plant_base_y = binary_y_min
        print(f"⚠️ 未找到object=221，使用辅助文件y_min作为基部: {plant_base_y:.6f}")
    
    plant_height_measured = binary_y_max - plant_base_y
    plant_height_real = plant_height_measured * length_scale if length_scale is not None else None
    
    # ========== 计算穗位高 ==========
    print("\n" + "=" * 60)
    print("计算穗位高（多object值分析）")
    print("=" * 60)
    
    ear_object_values = [222, 223, 224, 225, 226]
    ear_stats = []
    missing_ear = True
    
    for obj_val in ear_object_values:
        if obj_val in points_dict:
            points = points_dict[obj_val]
            if len(points) > 0:
                y_median = np.median(points[:, 1])
                y_max = np.max(points[:, 1])
                ear_stats.append({
                    'object_value': obj_val,
                    'point_count': len(points),
                    'y_median': y_median,
                    'y_max': y_max
                })
                print(f"  object {obj_val}: 点数={len(points):6d}, y中位数={y_median:.6f}, y最大值={y_max:.6f}")
                missing_ear = False
            else:
                print(f"  object {obj_val}: 存在但点数为0")
        else:
            print(f"  object {obj_val}: 不存在")
    
    selected_ear_object = None
    ear_position_y = None
    ear_height_measured = None
    ear_height_real = None
    ear_points = None
    
    if not missing_ear and ear_stats:
        min_median_obj = min(ear_stats, key=lambda x: x['y_median'])
        selected_ear_object = min_median_obj['object_value']
        ear_position_y = min_median_obj['y_max']
        ear_points = points_dict.get(selected_ear_object, None)
        
        print(f"\n✅ 选中的object值: {selected_ear_object}")
        print(f"  该object y中位数: {min_median_obj['y_median']:.6f} (最小)")
        print(f"  该object y最大值: {ear_position_y:.6f}")
        
        ear_height_measured = binary_y_max - ear_position_y
        ear_height_real = ear_height_measured * length_scale if length_scale is not None else None
        
        print(f"\n穗位高计算结果:")
        print(f"  辅助文件y最大值: {binary_y_max:.6f}")
        print(f"  选中的object y最大值: {ear_position_y:.6f}")
        print(f"  穗位高测量值: {ear_height_measured:.6f}")
        if ear_height_real is not None:
            print(f"  真实穗位高: {ear_height_real:.6f} cm")
        else:
            print(f"  真实穗位高: 无法计算（缺少参照物）")
    else:
        print("\n⚠️ 警告: 未找到任何穗部object值(222-226)的点云")
    
    # ========== 分析object值≥300的点云 ==========
    high_object_analysis, high_obj_info = analyze_high_object_values(
        points_dict, stem_points, selected_ear_object, ear_points
    )
    
    # ========== 处理所有叶片并计算夹角 ==========
    print("\n" + "=" * 60)
    print("开始处理所有叶片点云并计算夹角")
    print("=" * 60)
    
    NON_LEAF_OBJECTS = {220, 221, 222, 223, 224, 225, 226}
    
    all_leaf_results = []
    leaf_count = 0
    
    for obj_val, points in points_dict.items():
        if obj_val in NON_LEAF_OBJECTS:
            continue
        
        if len(points) < 30:
            print(f"\n跳过 object_value={obj_val}: 点数太少 ({len(points)})")
            continue
        
        print(f"\n处理叶片 {leaf_count + 1}: object_value={obj_val}, 点数={len(points)}")
        
        leaf_result = calibrate_leaf_measurements(reference_result, points, obj_val, main_ply)
        
        if leaf_result:
            if stem_direction is not None:
                angle, leaf_direction = calculate_leaf_angle(
                    points, stem_direction, stem_points, use_nearest_percent
                )
                if angle is not None:
                    leaf_result['叶片与主干夹角(°)'] = angle
                    leaf_result['叶片方向_x'] = leaf_direction[0] if leaf_direction is not None else None
                    leaf_result['叶片方向_y'] = leaf_direction[1] if leaf_direction is not None else None
                    leaf_result['叶片方向_z'] = leaf_direction[2] if leaf_direction is not None else None
                    print(f"  叶片夹角: {angle:.2f}°")
                else:
                    leaf_result['叶片与主干夹角(°)'] = None
                    leaf_result['叶片方向_x'] = None
                    leaf_result['叶片方向_y'] = None
                    leaf_result['叶片方向_z'] = None
            else:
                leaf_result['叶片与主干夹角(°)'] = None
                leaf_result['叶片方向_x'] = None
                leaf_result['叶片方向_y'] = None
                leaf_result['叶片方向_z'] = None
            
            leaf_result['叶片序号'] = leaf_count + 1
            leaf_result['object值'] = obj_val
            
            all_leaf_results.append(leaf_result)
            leaf_count += 1
            
            if leaf_result['处理状态'] == '成功':
                print(f"  长度: {leaf_result['真实长度']:.2f} cm")
                print(f"  平均宽度: {leaf_result['真实平均宽度']:.2f} cm")
                print(f"  最大宽度: {leaf_result['真实最大宽度']:.2f} cm")
                print(f"  面积: {leaf_result['真实面积']:.2f} cm²")
            else:
                print(f"  3D长度: {leaf_result['3D长度(中轴线法)']:.2f}")
                print(f"  3D平均宽度: {leaf_result['3D平均宽度(分段法)']:.2f}")
                print(f"  3D面积: {leaf_result['3D面积']:.2f}")
                print(f"  状态: {leaf_result['处理状态']}")
    
    print(f"\n叶片处理完成，共找到 {leaf_count} 个叶片")
    
    if high_object_analysis:
        print("\n" + "=" * 60)
        print("高object值(≥300)分析结果摘要")
        print("=" * 60)
        print(f"穗部object值: {high_object_analysis['ear_object_value']}")
        print(f"穗部y轴最小10%点的中位数j: {high_object_analysis['ear_bottom_10_percent_median_y']:.6f}")
        
        min_greater_obj = high_object_analysis['greater_than_j_min_object']
        max_less_obj = high_object_analysis['less_than_min_greater_max_object']
        
        if min_greater_obj:
            print(f"\n步骤1 - 大于j的最小i对应的object值: {min_greater_obj}")
        if max_less_obj:
            print(f"步骤2 - 小于该i的最大i对应的object值: {max_less_obj}")
    
    all_results = {
        'folder_name': folder_name,
        'folder_path': data_folder_path,
        'main_ply': main_ply,
        'hybrid_ply': hybrid_ply,
        'stem_cloud': stem_cloud_path,
        'stem_direction': stem_direction.tolist() if stem_direction is not None else None,
        'stem_vertical_angle': stem_vertical_angle,
        'missing_flags': {
            'missing_reference_220': missing_reference,
            'missing_base_221': not has_base_221,
            'missing_ear_all': missing_ear
        },
        'plant_height': {
            '测量值': plant_height_measured,
            '真实值': plant_height_real,
            '基部y': plant_base_y,
            '顶部y': binary_y_max,
            '基部来源': 'object_221' if has_base_221 else 'hybrid_file'
        } if plant_base_y is not None else None,
        'ear_height': {
            '测量值': ear_height_measured,
            '真实值': ear_height_real,
            '穗位y': ear_position_y,
            '顶部y': binary_y_max,
            '选中object值': selected_ear_object
        } if ear_position_y is not None else None,
        'ear_analysis': ear_stats,
        'high_object_analysis': high_object_analysis,
        'high_obj_info': high_obj_info,
        'leaves': all_leaf_results,
        'length_scale': length_scale,
        'reference_result': reference_result,
        'binary_file_info': {
            'y_min': binary_y_min,
            'y_max': binary_y_max,
            'point_count': binary_point_count
        },
        'output_dir': output_dir
    }
    
    return all_results


def save_results_to_excel(all_results_list, output_path=None):
    """
    将所有处理结果保存到一个Excel文件中
    """
    if not all_results_list:
        print("没有结果可保存")
        return None
    
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"all_plant_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 所有叶片结果汇总
        all_leaves = []
        for folder_result in all_results_list:
            if folder_result and 'leaves' in folder_result:
                for leaf in folder_result['leaves']:
                    leaf['文件夹'] = folder_result['folder_name']
                    leaf['文件夹路径'] = folder_result['folder_path']
                    missing = folder_result.get('missing_flags', {})
                    leaf['参照物状态'] = '缺失' if missing.get('missing_reference_220', False) else '正常'
                    leaf['基部状态'] = '缺失' if missing.get('missing_base_221', False) else '正常'
                    leaf['穗部状态'] = '缺失' if missing.get('missing_ear_all', False) else '正常'
                    if folder_result['stem_direction']:
                        leaf['主干方向_x'] = folder_result['stem_direction'][0]
                        leaf['主干方向_y'] = folder_result['stem_direction'][1]
                        leaf['主干方向_z'] = folder_result['stem_direction'][2]
                    all_leaves.append(leaf)
        
        if all_leaves:
            df_leaves = pd.DataFrame(all_leaves)
            cols = df_leaves.columns.tolist()
            angle_cols = ['叶片与主干夹角(°)', '叶片方向_x', '叶片方向_y', '叶片方向_z', 
                         '主干方向_x', '主干方向_y', '主干方向_z']
            other_cols = [c for c in cols if c not in angle_cols]
            new_cols = other_cols[:5] + angle_cols + other_cols[5:]
            new_cols = [c for c in new_cols if c in df_leaves.columns]
            df_leaves = df_leaves[new_cols]
            df_leaves.to_excel(writer, sheet_name='所有叶片汇总', index=False)
            print(f"叶片汇总: {len(all_leaves)} 条记录")
        
        # 高object值分析结果表
        high_obj_data = []
        for folder_result in all_results_list:
            if folder_result and folder_result.get('high_obj_info'):
                high_analysis = folder_result.get('high_object_analysis')
                folder_name = folder_result['folder_name']
                
                if high_analysis is not None:
                    greater_obj = high_analysis.get('greater_than_j_min_object')
                    max_less_obj = high_analysis.get('less_than_min_greater_max_object')
                else:
                    greater_obj = None
                    max_less_obj = None
                
                for info in folder_result['high_obj_info']:
                    obj_val = info['object_value']
                    mark_type = ''
                    if obj_val == greater_obj:
                        mark_type = '大于j的最小i对应的object'
                    elif obj_val == max_less_obj:
                        mark_type = '小于该i的最大i对应的object'
                    
                    high_obj_data.append({
                        '文件夹': folder_name,
                        '文件夹路径': folder_result['folder_path'],
                        'object值': obj_val,
                        '点数': info['point_count'],
                        'y轴最大10%点数': info['top_points_count'],
                        'y轴最大10%点y轴中位数(i)': info['y_median_i'],
                        'y_min': info['y_min'],
                        'y_max': info['y_max'],
                        '穗部object值': high_analysis.get('ear_object_value') if high_analysis else None,
                        '穗部y轴最小10%点中位数(j)': high_analysis.get('ear_bottom_10_percent_median_y') if high_analysis else None,
                        '标记说明': mark_type
                    })
        
        if high_obj_data:
            df_high_obj = pd.DataFrame(high_obj_data)
            df_high_obj.to_excel(writer, sheet_name='高object值分析(≥300)', index=False)
            print(f"高object值分析: {len(high_obj_data)} 条记录")
        
        # 株高结果表
        plant_height_data = []
        for folder_result in all_results_list:
            if folder_result:
                missing = folder_result.get('missing_flags', {})
                if folder_result.get('plant_height') is not None:
                    ph = folder_result['plant_height']
                    plant_height_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
                        '基部状态': '缺失' if missing.get('missing_base_221', False) else '正常',
                        '基部来源': ph.get('基部来源', 'unknown'),
                        '基部y': ph['基部y'],
                        '顶部y': ph['顶部y'],
                        '株高测量值': ph['测量值'],
                        '真实株高': ph['真实值'] if ph.get('真实值') is not None else 'N/A'
                    })
                else:
                    plant_height_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
                        '基部状态': '缺失' if missing.get('missing_base_221', False) else '正常',
                        '基部来源': 'N/A',
                        '基部y': 'N/A',
                        '顶部y': 'N/A',
                        '株高测量值': 'N/A',
                        '真实株高': 'N/A'
                    })
        
        if plant_height_data:
            pd.DataFrame(plant_height_data).to_excel(writer, sheet_name='株高结果', index=False)
            print(f"株高结果: {len(plant_height_data)} 条记录")
        
        # 穗位高结果表
        ear_height_data = []
        for folder_result in all_results_list:
            if folder_result:
                missing = folder_result.get('missing_flags', {})
                if folder_result.get('ear_height') is not None:
                    eh = folder_result['ear_height']
                    ear_height_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
                        '穗部状态': '缺失' if missing.get('missing_ear_all', False) else '正常',
                        '选中object值': eh.get('选中object值', 'N/A'),
                        '穗位y': eh['穗位y'] if eh.get('穗位y') is not None else 'N/A',
                        '顶部y': eh['顶部y'],
                        '穗位高测量值': eh['测量值'] if eh.get('测量值') is not None else 'N/A',
                        '真实穗位高': eh['真实值'] if eh.get('真实值') is not None else 'N/A'
                    })
                else:
                    ear_height_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        '参照物状态': '缺失' if missing.get('missing_reference_220', False) else '正常',
                        '穗部状态': '缺失' if missing.get('missing_ear_all', False) else '正常',
                        '选中object值': 'N/A',
                        '穗位y': 'N/A',
                        '顶部y': 'N/A',
                        '穗位高测量值': 'N/A',
                        '真实穗位高': 'N/A'
                    })
        
        if ear_height_data:
            pd.DataFrame(ear_height_data).to_excel(writer, sheet_name='穗位高结果', index=False)
            print(f"穗位高结果: {len(ear_height_data)} 条记录")
        
        # 穗部object详细分析表
        ear_analysis_data = []
        for folder_result in all_results_list:
            if folder_result and folder_result.get('ear_analysis'):
                for ear_stat in folder_result['ear_analysis']:
                    ear_analysis_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        'object值': ear_stat['object_value'],
                        '点数': ear_stat['point_count'],
                        'y中位数': ear_stat['y_median'],
                        'y最大值': ear_stat['y_max']
                    })
        
        if ear_analysis_data:
            pd.DataFrame(ear_analysis_data).to_excel(writer, sheet_name='穗部object分析', index=False)
            print(f"穗部object分析: {len(ear_analysis_data)} 条记录")
        
        # 文件夹统计
        folder_stats = []
        for folder_result in all_results_list:
            if folder_result:
                missing = folder_result.get('missing_flags', {})
                high_analysis = folder_result.get('high_object_analysis')
                folder_stats.append({
                    '文件夹': folder_result['folder_name'],
                    '文件夹路径': folder_result['folder_path'],
                    '参照物(220)': '缺失' if missing.get('missing_reference_220', False) else '存在',
                    '基部(221)': '缺失' if missing.get('missing_base_221', False) else '存在',
                    '穗部(222-226)': '缺失' if missing.get('missing_ear_all', False) else '存在',
                    '叶片数量': len(folder_result.get('leaves', [])),
                    '长度标定系数': folder_result['length_scale'] if folder_result.get('length_scale') is not None else 'N/A',
                    '参照物3D长度': folder_result['reference_result']['length_2d'] if folder_result.get('reference_result') else 'N/A',
                    '参照物拟合面积': folder_result['reference_result']['rectangle_area_2d'] if folder_result.get('reference_result') else 'N/A',
                    '参照物内点比例': f"{folder_result['reference_result']['inlier_ratio']*100:.1f}%" if folder_result.get('reference_result') else 'N/A',
                    '辅助文件y_min': folder_result['binary_file_info']['y_min'],
                    '辅助文件y_max': folder_result['binary_file_info']['y_max'],
                    '主干点云存在': os.path.exists(folder_result['stem_cloud']) if folder_result['stem_cloud'] else False,
                    '主干与垂直方向夹角(°)': folder_result.get('stem_vertical_angle', 'N/A'),
                    '大于j的最小i对应的object值': high_analysis.get('greater_than_j_min_object') if high_analysis else None,
                    '小于该i的最大i对应的object值': high_analysis.get('less_than_min_greater_max_object') if high_analysis else None
                })
        
        if folder_stats:
            df_stats = pd.DataFrame(folder_stats)
            df_stats.to_excel(writer, sheet_name='文件夹统计', index=False)
            print(f"文件夹统计: {len(folder_stats)} 个")
        
        # 关键object值摘要表
        summary_data = []
        for folder_result in all_results_list:
            if folder_result:
                high_analysis = folder_result.get('high_object_analysis')
                if high_analysis:
                    summary_data.append({
                        '文件夹': folder_result['folder_name'],
                        '文件夹路径': folder_result['folder_path'],
                        '穗部object值': high_analysis.get('ear_object_value'),
                        '穗部y轴最小10%点中位数(j)': high_analysis.get('ear_bottom_10_percent_median_y'),
                        '大于j的最小i对应的object值': high_analysis.get('greater_than_j_min_object'),
                        '该object的i值': high_analysis.get('greater_than_j_min_median_y'),
                        '小于该i的最大i对应的object值': high_analysis.get('less_than_min_greater_max_object'),
                        '该object的i值': high_analysis.get('less_than_min_greater_max_median_y')
                    })
        
        if summary_data:
            pd.DataFrame(summary_data).to_excel(writer, sheet_name='关键object值摘要', index=False)
            print(f"关键object值摘要: {len(summary_data)} 条记录")
    
    print(f"\n✅ 所有结果已保存到: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description='完整的叶片表型参数提取系统 - 改进版（使用外部命令行主干提取）')
    
    parser.add_argument('folders', nargs='+', 
                       help='要处理的数据文件夹路径')
    parser.add_argument('--eps', type=float, default=0.02, help='DBSCAN聚类半径（已禁用，保留参数兼容性）')
    parser.add_argument('--min_points', type=int, default=10, help='DBSCAN最小簇点数（已禁用，保留参数兼容性）')
    parser.add_argument('--excel', '-e', type=str, default=None, help='Excel输出路径')
    parser.add_argument('--nearest_percent', type=int, default=30, 
                       help='使用离主干最近的百分比点计算叶片方向')
    parser.add_argument('--skip_skeleton', action='store_true',
                       help='跳过主干提取，使用已有的文件')
    parser.add_argument('--stem_radius', type=float, default=0.15,
                       help='圆柱形主干提取的半径（默认0.15）')
    parser.add_argument('--ransac_iter', type=int, default=500,
                       help='RANSAC矩形拟合最大迭代次数（默认500）')
    parser.add_argument('--ransac_thresh', type=float, default=0.01,
                       help='RANSAC矩形拟合距离阈值（默认0.01）')
    
    args = parser.parse_args()
    
    # 展开文件夹列表
    expanded_folders = []
    for pattern in args.folders:
        expanded_folders.extend(glob.glob(pattern))
    
    expanded_folders = [f for f in expanded_folders if os.path.isdir(f)]
    
    if not expanded_folders:
        print("错误: 没有找到匹配的文件夹")
        sys.exit(1)
    
    print("=" * 80)
    print("完整的叶片表型参数提取系统 - 改进版（使用外部命令行主干提取）")
    print("=" * 80)
    print(f"当前Python解释器: {get_python_executable()}")
    print(f"注意: 聚类功能已禁用，直接在原始点云上处理")
    print(f"叶片夹角: 使用离主干最近的 {args.nearest_percent}% 点")
    print(f"主干提取: {'跳过' if args.skip_skeleton else '自动运行（两步法）'}")
    print(f"  步骤1: 圆柱形主干提取 (半径={args.stem_radius})")
    print(f"  步骤2: SLBC骨架提取和后处理")
    print(f"参照物拟合: 双重RANSAC矩形拟合方法")
    print(f"  RANSAC迭代次数: {args.ransac_iter}")
    print(f"  RANSAC距离阈值: {args.ransac_thresh}")
    print(f"高object值分析: object值≥300的点云，计算与穗部的关系")
    print(f"  步骤1: 找到大于j的最小i对应的object")
    print(f"  步骤2: 找到小于该i的最大i对应的object")
    print(f"待处理数据文件夹: {len(expanded_folders)}")
    print("=" * 80)
    
    print("\n找到的数据文件夹:")
    for folder in expanded_folders:
        print(f"  - {folder}")
    
    all_results_list = []
    
    for i, folder_path in enumerate(expanded_folders, 1):
        print(f"\n[{i}/{len(expanded_folders)}] 处理: {folder_path}")
        
        try:
            results = process_single_data_folder(
                folder_path,
                eps=args.eps,
                min_points=args.min_points,
                use_nearest_percent=args.nearest_percent,
                skip_skeleton=args.skip_skeleton,
                ransac_max_iterations=args.ransac_iter,
                ransac_distance_threshold=args.ransac_thresh,
                stem_radius=args.stem_radius
            )
            
            if results:
                all_results_list.append(results)
                print(f"\n✅ {folder_path} 处理成功")
            else:
                print(f"\n❌ {folder_path} 处理失败")
        except Exception as e:
            print(f"\n❌ {folder_path} 处理失败: {e}")
            traceback.print_exc()
    
    if all_results_list:
        save_results_to_excel(all_results_list, args.excel)
        print(f"\n处理完成! 成功处理 {len(all_results_list)}/{len(expanded_folders)} 个文件夹")
    else:
        print("\n没有成功处理任何文件夹")


if __name__ == "__main__":
    main()


