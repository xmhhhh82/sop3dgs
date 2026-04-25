import os
import numpy as np
import open3d as o3d
import argparse
import sys

def get_single_ply_center(ply_file_path, sampling_rate=0.1):
    """
    从单个PLY文件计算中心点
    :param ply_file_path: PLY文件路径
    :param sampling_rate: 采样率，0.1表示使用10%的点来计算中心点
    :return: 中心点坐标
    """
    print(f"读取PLY文件: {ply_file_path}")
    
    try:
        pcd = o3d.io.read_point_cloud(ply_file_path)
        points = np.asarray(pcd.points)
        
        if len(points) == 0:
            raise ValueError("PLY文件没有点云数据")
        
        # 如果需要，进行采样
        if sampling_rate < 1.0:
            sample_size = max(1, int(len(points) * sampling_rate))
            indices = np.random.choice(len(points), sample_size, replace=False)
            points = points[indices]
            print(f"采样后使用 {sample_size} 个点计算中心")
        
        # 计算中心点
        center_point = np.mean(points, axis=0)
        
        print(f"总点数: {len(points)}")
        print(f"中心点坐标: ({center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f})")
        
        return center_point
        
    except Exception as e:
        print(f"读取文件 {ply_file_path} 时出错: {e}")
        raise

def remove_outliers_statistical(pcd, nb_neighbors=20, std_ratio=2.0):
    """
    使用统计方法去除离群点
    :param pcd: 输入点云
    :param nb_neighbors: 邻居数量
    :param std_ratio: 标准差倍数
    :return: 去噪后的点云, 离群点索引
    """
    print("使用统计方法去除离群点...")
    print(f"参数: nb_neighbors={nb_neighbors}, std_ratio={std_ratio}")
    
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors=nb_neighbors, std_ratio=std_ratio)
    
    inlier_pcd = pcd.select_by_index(ind)
    outlier_pcd = pcd.select_by_index(ind, invert=True)
    
    print(f"保留点数: {len(inlier_pcd.points)}")
    print(f"去除点数: {len(outlier_pcd.points)}")
    
    return inlier_pcd, outlier_pcd

def remove_outliers_dbscan(pcd, eps=0.05, min_points=10):
    """
    使用DBSCAN聚类方法去除离群点
    :param pcd: 输入点云
    :param eps: DBSCAN的搜索半径
    :param min_points: 最小点数
    :return: 去噪后的点云, 离群点索引
    """
    print("使用DBSCAN方法去除离群点...")
    print(f"参数: eps={eps}, min_points={min_points}")
    
    # 使用DBSCAN进行聚类
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug):
        labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points, print_progress=True))
    
    # 找出最大的簇（假设主物体是最大的簇）
    if labels.max() < 0:  # 所有点都是噪声
        print("警告: DBSCAN没有找到任何聚类，所有点都被标记为噪声")
        # 返回原始点云作为主聚类
        main_indices = np.where(labels >= 0)[0]  # 没有聚类，返回空
        other_indices = np.where(labels < 0)[0]
    else:
        max_label = labels.max()
        print(f"找到 {max_label + 1} 个聚类")
        
        # 统计每个簇的点数
        cluster_sizes = []
        for i in range(max_label + 1):
            cluster_size = np.sum(labels == i)
            cluster_sizes.append(cluster_size)
            print(f"聚类 {i}: {cluster_size} 个点")
        
        # 选择点数最多的簇
        main_cluster_label = np.argmax(cluster_sizes)
        print(f"选择主聚类: {main_cluster_label} (包含 {cluster_sizes[main_cluster_label]} 个点)")
        
        # 提取主聚类的点
        main_indices = np.where(labels == main_cluster_label)[0]
        other_indices = np.where(labels != main_cluster_label)[0]
    
    inlier_pcd = pcd.select_by_index(main_indices)
    outlier_pcd = pcd.select_by_index(other_indices)
    
    print(f"保留点数: {len(inlier_pcd.points)}")
    print(f"去除点数: {len(outlier_pcd.points)}")
    
    return inlier_pcd, outlier_pcd

def find_files_in_folder(folder_path):
    """
    在给定的文件夹中自动查找目标物体文件和场景点云文件
    :param folder_path: 文件夹路径（如91227-1_frames）
    :return: (object_file_path, scene_file_path, output_file_path, base_name)
    """
    # 获取文件夹名称作为基础名称
    folder_name = os.path.basename(folder_path)
    
    # 如果文件夹名以_frames结尾，去掉后缀
    if folder_name.endswith('_frames'):
        base_name = folder_name[:-7]  # 去掉 '_frames'
    else:
        base_name = folder_name
    
    print(f"\n处理文件夹: {folder_path}")
    print(f"基础名称: {base_name}")
    
    # 构建可能的路径
    # 目标物体文件: output_[base_name]/train/point_cloud_non_bg_with_values.ply
    object_file = os.path.join(folder_path, f"output_{base_name}", "train", "point_cloud_non_bg_with_values.ply")
    
    # 场景点云文件: output_[base_name]/point_cloud/iteration_30000/point_cloud.ply
    scene_file = os.path.join(folder_path, f"output_{base_name}", "point_cloud", "iteration_30000", "point_cloud.ply")
    
    # 输出文件: output_[base_name]/hybrid_filtered_final.ply
    output_file = os.path.join(folder_path, f"output_{base_name}", "hybrid_filtered_final.ply")
    
    # 检查文件是否存在
    object_exists = os.path.exists(object_file)
    scene_exists = os.path.exists(scene_file)
    
    if not object_exists:
        print(f"警告: 目标物体文件不存在: {object_file}")
    if not scene_exists:
        print(f"警告: 场景点云文件不存在: {scene_file}")
    
    return object_file, scene_file, output_file, base_name, object_exists and scene_exists

def hybrid_filter_pipeline_single_ply(object_ply_file, scene_ply_file, radius=2.0, 
                                     statistical_params=None, dbscan_params=None,
                                     output_path=None, sampling_rate=0.1):
    """
    混合过滤管道（使用单个PLY文件作为目标参考）：
    球体过滤 + 统计方法去噪 + DBSCAN聚类
    
    :param object_ply_file: 目标物体的PLY文件路径（用于确定中心点）
    :param scene_ply_file: 完整场景的PLY文件路径（需要过滤的点云）
    :param radius: 球体过滤半径（米）
    :param statistical_params: 统计去噪参数
    :param dbscan_params: DBSCAN聚类参数
    :param output_path: 输出文件路径
    :param sampling_rate: 计算中心点时的采样率
    :return: 过滤后的点云, 中心点, 输出文件路径
    """
    if statistical_params is None:
        statistical_params = {'nb_neighbors': 10, 'std_ratio': 0.01}
    if dbscan_params is None:
        dbscan_params = {'eps': 0.05, 'min_points': 10}
    
    print("=" * 60)
    print("开始混合过滤管道处理（单PLY参考模式）")
    print("=" * 60)
    print(f"目标参考文件: {object_ply_file}")
    print(f"场景点云文件: {scene_ply_file}")
    
    # 1. 从单个PLY文件获取中心点
    print("\n步骤1: 从目标PLY文件计算中心点...")
    center_point = get_single_ply_center(object_ply_file, sampling_rate)
    
    # 2. 球体过滤
    print("\n步骤2: 执行球体过滤...")
    pcd = o3d.io.read_point_cloud(scene_ply_file)
    original_points = len(pcd.points)
    points = np.asarray(pcd.points)
    
    distances = np.linalg.norm(points - center_point, axis=1)
    mask = distances <= radius
    
    sphere_filtered_pcd = o3d.geometry.PointCloud()
    sphere_filtered_pcd.points = o3d.utility.Vector3dVector(points[mask])
    
    # 保留颜色信息
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        sphere_filtered_pcd.colors = o3d.utility.Vector3dVector(colors[mask])
    
    sphere_points = len(sphere_filtered_pcd.points)
    print(f"球体过滤结果: {original_points} -> {sphere_points} 个点")
    print(f"球体过滤去除率: {100*(1-sphere_points/original_points):.1f}%")
    
    # 3. 统计方法去除离群点
    print("\n步骤3: 使用统计方法去除离群点...")
    statistical_pcd, statistical_outliers = remove_outliers_statistical(
        sphere_filtered_pcd, 
        **statistical_params
    )
    statistical_points = len(statistical_pcd.points)
    print(f"统计去噪结果: {sphere_points} -> {statistical_points} 个点")
    print(f"统计去噪去除率: {100*(1-statistical_points/sphere_points):.1f}%")
    
    # 4. DBSCAN聚类获取主聚类
    print("\n步骤4: 使用DBSCAN聚类获取主聚类...")
    final_pcd, dbscan_outliers = remove_outliers_dbscan(
        statistical_pcd,
        **dbscan_params
    )
    final_points = len(final_pcd.points)
    print(f"DBSCAN聚类结果: {statistical_points} -> {final_points} 个点")
    print(f"DBSCAN去除率: {100*(1-final_points/statistical_points):.1f}%")
    
    # 5. 保存最终结果到指定路径
    if output_path is None:
        # 如果没有指定输出路径，生成默认文件名
        timestamp = np.datetime64('now').astype('datetime64[s]').astype(str).replace(':', '').replace('-', '').replace('T', '_')
        output_path = f"hybrid_filtered_final_{timestamp}.ply"
    else:
        # 确保输出目录存在
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    # 保存最终结果
    o3d.io.write_point_cloud(output_path, final_pcd)
    print(f"\n保存最终结果到: {output_path}")
    
    # 文件大小信息
    if os.path.exists(output_path):
        file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
        print(f"文件大小: {file_size_mb:.2f} MB")
    
    # 6. 统计信息
    print("\n" + "=" * 60)
    print("处理完成!")
    print("=" * 60)
    print(f"中心点来源: {object_ply_file}")
    print(f"中心点坐标: ({center_point[0]:.6f}, {center_point[1]:.6f}, {center_point[2]:.6f})")
    print(f"过滤半径: {radius}")
    
    print(f"\n统计参数:")
    print(f"  统计方法: nb_neighbors={statistical_params['nb_neighbors']}, std_ratio={statistical_params['std_ratio']}")
    print(f"  DBSCAN: eps={dbscan_params['eps']}, min_points={dbscan_params['min_points']}")
    
    print(f"\n点数统计:")
    print(f"  原始场景点数: {original_points}")
    print(f"  球体过滤后: {sphere_points}")
    print(f"  统计去噪后: {statistical_points}")
    print(f"  DBSCAN聚类后: {final_points}")
    
    print(f"\n去除比例:")
    print(f"  球体过滤: {100*(1-sphere_points/original_points):.1f}%")
    print(f"  统计去噪: {100*(1-statistical_points/sphere_points):.1f}%")
    print(f"  DBSCAN聚类: {100*(1-final_points/statistical_points):.1f}%")
    print(f"  总计去除: {100*(1-final_points/original_points):.1f}%")
    
    return final_pcd, center_point, output_path

def process_single_folder(folder_path, radius=2.0, 
                         statistical_params=None, 
                         dbscan_params=None,
                         sampling_rate=0.1):
    """
    处理单个文件夹
    """
    # 自动查找文件
    object_file, scene_file, output_file, base_name, files_exist = find_files_in_folder(folder_path)
    
    if not files_exist:
        print(f"错误: 文件夹 {folder_path} 中缺少必要的文件，跳过处理")
        return None
    
    print(f"\n开始处理: {base_name}")
    print(f"目标物体文件: {object_file}")
    print(f"场景点云文件: {scene_file}")
    print(f"输出文件: {output_file}")
    
    try:
        # 执行混合过滤管道
        final_pcd, center, output = hybrid_filter_pipeline_single_ply(
            object_ply_file=object_file,
            scene_ply_file=scene_file,
            radius=radius,
            statistical_params=statistical_params,
            dbscan_params=dbscan_params,
            output_path=output_file,
            sampling_rate=sampling_rate
        )
        
        print(f"\n✓ 成功处理: {base_name}")
        return {
            'folder': folder_path,
            'base_name': base_name,
            'output': output,
            'center': center
        }
        
    except Exception as e:
        print(f"\n✗ 处理 {base_name} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """
    主函数：接受命令行参数
    用法：python script.py 文件夹路径1 文件夹路径2 ...
    或：python script.py --folders 文件夹路径1 文件夹路径2 ...
    """
    parser = argparse.ArgumentParser(description='点云混合过滤处理工具')
    parser.add_argument('folders', nargs='+', help='要处理的文件夹路径（一个或多个）')
    parser.add_argument('--radius', type=float, default=2.0, help='球体过滤半径（米），默认2.0')
    parser.add_argument('--nb_neighbors', type=int, default=10, help='统计去噪的邻居数量，默认10')
    parser.add_argument('--std_ratio', type=float, default=0.01, help='统计去噪的标准差倍数，默认0.01')
    parser.add_argument('--eps', type=float, default=0.05, help='DBSCAN聚类半径，默认0.05')
    parser.add_argument('--min_points', type=int, default=10, help='DBSCAN最小点数，默认10')
    parser.add_argument('--sampling_rate', type=float, default=0.1, help='中心点计算采样率，默认0.1')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("点云混合过滤处理工具")
    print("=" * 80)
    print(f"处理文件夹数量: {len(args.folders)}")
    print(f"过滤半径: {args.radius}米")
    print(f"统计去噪参数: nb_neighbors={args.nb_neighbors}, std_ratio={args.std_ratio}")
    print(f"DBSCAN参数: eps={args.eps}, min_points={args.min_points}")
    print(f"采样率: {args.sampling_rate}")
    print("=" * 80)
    
    # 设置参数
    statistical_params = {
        'nb_neighbors': args.nb_neighbors,
        'std_ratio': args.std_ratio
    }
    dbscan_params = {
        'eps': args.eps,
        'min_points': args.min_points
    }
    
    # 处理每个文件夹
    results = []
    for folder_path in args.folders:
        # 检查文件夹是否存在
        if not os.path.exists(folder_path):
            print(f"\n错误: 文件夹不存在: {folder_path}")
            continue
        
        result = process_single_folder(
            folder_path=folder_path,
            radius=args.radius,
            statistical_params=statistical_params,
            dbscan_params=dbscan_params,
            sampling_rate=args.sampling_rate
        )
        
        if result:
            results.append(result)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("处理总结")
    print("=" * 80)
    print(f"成功处理: {len(results)}/{len(args.folders)} 个文件夹")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['base_name']}:")
        print(f"   输出文件: {result['output']}")
        print(f"   中心点: ({result['center'][0]:.6f}, {result['center'][1]:.6f}, {result['center'][2]:.6f})")

def interactive_mode():
    """
    交互式模式：让用户输入文件夹路径
    """
    print("=" * 80)
    print("点云混合过滤处理工具 - 交互式模式")
    print("=" * 80)
    print("请输入要处理的文件夹路径（每行一个，输入空行结束）：")
    
    folders = []
    while True:
        folder = input().strip()
        if not folder:
            break
        if os.path.exists(folder):
            folders.append(folder)
        else:
            print(f"警告: 文件夹不存在，已跳过: {folder}")
    
    if not folders:
        print("没有输入有效的文件夹路径")
        return
    
    print(f"\n将处理 {len(folders)} 个文件夹:")
    for folder in folders:
        print(f"  {folder}")
    
    # 获取参数
    try:
        radius = float(input("\n请输入过滤半径（米，默认2.0）: ") or "2.0")
        nb_neighbors = int(input("请输入统计去噪邻居数量（默认10）: ") or "10")
        std_ratio = float(input("请输入统计去噪标准差倍数（默认0.01）: ") or "0.01")
        eps = float(input("请输入DBSCAN聚类半径（默认0.05）: ") or "0.05")
        min_points = int(input("请输入DBSCAN最小点数（默认10）: ") or "10")
        sampling_rate = float(input("请输入中心点计算采样率（默认0.1）: ") or "0.1")
    except ValueError:
        print("输入参数错误，使用默认值")
        radius, nb_neighbors, std_ratio, eps, min_points, sampling_rate = 2.0, 10, 0.01, 0.05, 10, 0.1
    
    statistical_params = {
        'nb_neighbors': nb_neighbors,
        'std_ratio': std_ratio
    }
    dbscan_params = {
        'eps': eps,
        'min_points': min_points
    }
    
    # 处理文件夹
    results = []
    for folder_path in folders:
        result = process_single_folder(
            folder_path=folder_path,
            radius=radius,
            statistical_params=statistical_params,
            dbscan_params=dbscan_params,
            sampling_rate=sampling_rate
        )
        
        if result:
            results.append(result)
    
    # 打印总结
    print("\n" + "=" * 80)
    print("处理总结")
    print("=" * 80)
    print(f"成功处理: {len(results)}/{len(folders)} 个文件夹")
    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['base_name']}:")
        print(f"   输出文件: {result['output']}")
        print(f"   中心点: ({result['center'][0]:.6f}, {result['center'][1]:.6f}, {result['center'][2]:.6f})")

if __name__ == "__main__":
    # 如果没有命令行参数，进入交互式模式
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()



# 单个文件夹：python /datashare/dir_liusha/xibeinonglin/1_15_提取表型/主干提取.py /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames
# 多个文件夹：python /datashare/dir_liusha/xibeinonglin/1_15_提取表型/主干提取.py /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames  /datashare/dir_liusha/xibeinonglin/样本数据/DK517F-1_frames