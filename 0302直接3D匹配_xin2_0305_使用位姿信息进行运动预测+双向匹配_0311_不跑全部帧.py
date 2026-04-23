# # 单个文件跑
# import os
# import numpy as np
# import cv2
# from pathlib import Path
# from collections import defaultdict
# import json
# from tqdm import tqdm
# import re
# import random
# from scipy.spatial.distance import cdist
# import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
# from datetime import datetime
# from scipy.ndimage import binary_erosion, binary_dilation

# class CenterPoint3DMatcher:
#     """
#     以第一帧为初始帧的3D空间实例匹配器
#     核心功能：数据驱动的多证据融合匹配
#     - 运动连续性预测
#     - 双向空间匹配
#     - 掩码形状相似性
#     """
    
#     def __init__(self, config):
#         """
#         config: {
#             'depth_dir': 深度图目录,
#             'camera_json': 相机参数文件,
#             'mask_dir': 分割掩码目录,
#             'output_dir': 输出目录,
#             'target_class': 目标类别 (默认3),
#             'fixed_class_ids': 固定类别ID映射 {class_id: global_id},
#             'depth_scale': 深度缩放因子 (默认1000),
#             'depth_format': 深度图格式 ('16bit' 或 'npy'),
#             'max_depth': 最大有效深度 (默认10.0米),
#             'spatial_threshold': 空间距离阈值 (默认0.3米),
#             'motion_threshold': 运动预测阈值 (默认0.5米),
#             'motion_weight': 运动连续性权重 (默认0.4),
#             'bidirectional_weight': 双向匹配权重 (默认0.4),
#             'shape_weight': 形状相似性权重 (默认0.2),
#             'coherence_threshold': 协同阈值 (默认0.2),
#             'confidence_threshold': 置信度阈值 (默认0.5),
#             'first_frame': 指定第一帧 (可选),
#             'colormap': 颜色映射方式 ('random', 'tab20', 'hsv') 默认'tab20',
            
#             # 运动相关配置
#             'use_motion_prior': True,         # 是否使用运动先验
            
#             # 前K个最小深度的配置
#             'top_k_min_depths': 5000,
#             'use_top_k_depths': True,
            
#             # 像素深度保存配置
#             'pixel_depth_save': {...},
            
#             'id_font_scale': 0.4,
#             'depth_font_scale': 0.3,
#             'text_color': (255, 255, 255),
#             'unmatched_color': (128, 128, 128),
#             'min_instance_area': 5,
            
#             # 新增：只处理前N帧
#             'max_frames': 10,  # 只处理前10帧
#             'first_frame_candidates': [1,2,3,4,5],  # 候选初始帧
#         }
#         """
#         self.config = config
#         self.depth_dir = Path(config['depth_dir'])
#         self.camera_json = config['camera_json']
#         self.mask_dir = Path(config['mask_dir'])
#         self.output_dir = Path(config['output_dir'])
        
#         # 新增：目标类别和固定类别ID
#         self.target_class = config.get('target_class', 3)
#         self.fixed_class_ids = config.get('fixed_class_ids', {
#             0: 220,  # 类别0固定为220
#             1: 221,  # 类别1固定为221
#             2: 222   # 类别2固定为222
#         })
        
#         # 新增：候选初始帧
#         self.first_frame_candidates = config.get('first_frame_candidates', [1, 2, 3, 4, 5])
        
#         # 参数配置
#         self.depth_scale = config.get('depth_scale', 1000.0)
#         self.depth_format = config.get('depth_format', '16bit')
#         self.max_depth = config.get('max_depth', 10.0)
#         self.spatial_threshold = config.get('spatial_threshold', 0.3)
#         self.motion_threshold = config.get('motion_threshold', 0.5)
#         self.motion_weight = config.get('motion_weight', 0.4)        # 运动连续性权重
#         self.bidirectional_weight = config.get('bidirectional_weight', 0.4)  # 双向匹配权重
#         self.shape_weight = config.get('shape_weight', 0.2)          # 形状相似性权重
#         self.coherence_threshold = config.get('coherence_threshold', 0.2)
#         self.confidence_threshold = config.get('confidence_threshold', 0.5)
#         self.colormap = config.get('colormap', 'tab20')
#         self.min_instance_area = config.get('min_instance_area', 5)
        
#         # 新增：只处理前N帧
#         self.max_frames = config.get('max_frames', 10)
        
#         # 运动相关配置
#         self.use_motion_prior = config.get('use_motion_prior', True)
        
#         # ========== 前K个最小深度的配置 ==========
#         self.top_k_min_depths = config.get('top_k_min_depths', 2000)
#         self.use_top_k_depths = config.get('use_top_k_depths', True)
        
#         # 像素深度保存配置
#         self.pixel_depth_config = config.get('pixel_depth_save', {})
#         self.pixel_depth_enabled = self.pixel_depth_config.get('enabled', False)
#         self.frames_to_save = self.pixel_depth_config.get('frames_to_save', [])
#         self.depth_decimals = self.pixel_depth_config.get('depth_decimals', 3)
#         self.pixel_min_depth = self.pixel_depth_config.get('min_depth', 0.0)
#         self.pixel_max_depth = self.pixel_depth_config.get('max_depth', 10.0)
#         self.include_coordinates = self.pixel_depth_config.get('include_coordinates', True)
#         self.separate_files = self.pixel_depth_config.get('separate_files', True)
#         self.save_visualization = self.pixel_depth_config.get('save_visualization', True)
        
#         # 文字绘制参数
#         self.id_font_scale = config.get('id_font_scale', 0.4)
#         self.depth_font_scale = config.get('depth_font_scale', 0.3)
#         self.text_color = config.get('text_color', (255, 255, 255))
#         self.unmatched_color = config.get('unmatched_color', (128, 128, 128))
        
#         # 可以手动指定第一帧，但会被覆盖如果设置了自动选择
#         self.first_frame = config.get('first_frame', None)
        
#         # 数据结构
#         self.cameras = {}           # 相机参数 {frame_num: {...}}
#         self.depth_maps = {}         # 深度图 {frame_num: array}
#         self.depth_maps_meters = {}  # 深度图（米为单位）{frame_num: array}
#         self.masks = {}              # 分割掩码 {frame_num: array} - 只包含目标类别的实例
#         self.mask_contours = {}      # 掩码轮廓 {frame_num: {instance_id: contour}}
#         self.mask_moments = {}       # 掩码矩 {frame_num: {instance_id: moments}}
        
#         # 类别信息
#         self.frame_class_info = defaultdict(dict)  # {frame_num: {local_id: class_id}}
        
#         # 实例中心点信息
#         self.instance_centers_2d = defaultdict(dict)  # {frame_num: {instance_id: (x, y)}}
#         self.instance_depths = defaultdict(dict)      # {frame_num: {instance_id: depth_median}}
#         self.instance_stats = defaultdict(dict)       # {frame_num: {instance_id: stats}}
        
#         # 保存每个实例的所有有效深度值
#         self.instance_all_depths = defaultdict(dict)  # {frame_num: {instance_id: depths_list}}
        
#         # 3D中心点
#         self.instance_centers_3d = defaultdict(dict)  # {frame_num: {instance_id: [x, y, z]}}
        
#         # 全局ID管理
#         self.global_instance_counter = 223  # 从223开始，因为220-222已被固定类别占用
#         self.global_instance_map = {}  # (frame_num, local_id) -> global_id
#         self.global_centers_3d = {}    # global_id -> 3D中心点
#         self.global_stats = {}          # global_id -> 统计信息
#         self.global_masks = {}           # global_id -> 历史掩码 {frame_num: mask}
        
#         # ========== 运动跟踪相关 ==========
#         self.motion_trajectories = defaultdict(list)  # global_id -> [(frame_num, position), ...]
        
#         # 颜色映射
#         self.global_id_colors = {}      # global_id -> (B, G, R)
        
#         # 匹配结果统计
#         self.matching_stats = {
#             'total_matches': 0,
#             'cooperative_matches': 0,    # 协同匹配数
#             'motion_matches': 0,          # 纯运动匹配数
#             'bidirectional_matches': 0,   # 纯双向匹配数
#             'shape_matches': 0,           # 纯形状匹配数
#             'high_confidence': 0,
#             'low_confidence': 0,
#             'unmatched_instances': 0,
#             'rejected_matches': 0
#         }
        
#         # 帧掩码数量统计
#         self.frame_mask_counts = {}  # frame_num -> mask_count (目标类实例数)
#         self.frame_original_counts = {}  # 原始实例总数
#         self.frame_matched_counts = {}   # 匹配成功的实例数
#         self.frame_instances_with_depth = {}  # 有有效深度的实例数
        
#         # 创建输出目录
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # 彩色掩码目录
#         self.color_masks_dir = self.output_dir / "color_masks"
#         self.color_masks_dir.mkdir(exist_ok=True)
        
#         # 带信息的彩色掩码目录
#         self.color_masks_with_info_dir = self.output_dir / "color_masks_with_info"
#         self.color_masks_with_info_dir.mkdir(exist_ok=True)
        
#         # 像素深度数据目录
#         self.pixel_depth_data_dir = self.output_dir / "pixel_depth_data"
#         self.pixel_depth_data_dir.mkdir(exist_ok=True)
        
#         # 可视化图像目录
#         self.saved_frames_viz_dir = self.output_dir / "saved_frames_visualization"
#         self.saved_frames_viz_dir.mkdir(exist_ok=True)
        
#         # ========== 深度统计目录 ==========
#         self.depth_stats_dir = self.output_dir / "depth_statistics"
#         self.depth_stats_dir.mkdir(exist_ok=True)
        
#         # ========== 中心点3D坐标保存目录 ==========
#         self.centers_3d_dir = self.output_dir / "centers_3d_info"
#         self.centers_3d_dir.mkdir(exist_ok=True)
        
#         # ========== 匹配详细信息目录 ==========
#         self.matching_info_dir = self.output_dir / "matching_info"
#         self.matching_info_dir.mkdir(exist_ok=True)
        
#         print("="*60)
#         print("🚀 初始化数据驱动3D匹配器")
#         print("="*60)
#         print(f"📊 配置参数:")
#         print(f"   目标类别: {self.target_class} (进行匹配的类别)")
#         print(f"   固定类别ID映射: {self.fixed_class_ids}")
#         print(f"   候选初始帧: {self.first_frame_candidates}")
#         print(f"   最小实例面积: {self.min_instance_area} 像素")
#         print(f"   空间阈值: {self.spatial_threshold} 米")
#         print(f"   运动阈值: {self.motion_threshold} 米")
#         print(f"   运动权重: {self.motion_weight}")
#         print(f"   双向权重: {self.bidirectional_weight}")
#         print(f"   形状权重: {self.shape_weight}")
#         print(f"   协同阈值: {self.coherence_threshold}")
#         print(f"   最大深度: {self.max_depth} 米")
#         print(f"   处理帧数限制: 前 {self.max_frames} 帧")
        
#         if self.use_top_k_depths:
#             print(f"   深度计算方法: 使用前 {self.top_k_min_depths} 个最小深度值的中位数")
#         else:
#             print(f"   深度计算方法: 使用所有有效深度的中位数")
        
#         if self.pixel_depth_enabled:
#             print("\n📝 像素深度保存配置:")
#             print(f"   启用: {self.pixel_depth_enabled}")
#             print(f"   要保存的帧: {self.frames_to_save}")
#         print("="*60)
        
#         # 加载数据
#         self.load_camera_parameters()
#         self.load_depth_maps()
#         self.load_masks_with_class_info()
        
#         # 提取掩码轮廓和矩
#         self.extract_mask_features()
        
#         # 计算所有实例的中心点和深度中位数
#         self.compute_instance_centers()
        
#         # 将2D中心点转换为3D
#         self.convert_to_3d_centers()
        
#         # 初始化固定类别的全局点
#         self.initialize_fixed_class_points()
        
#         # 自动选择最佳初始帧
#         self.select_best_first_frame()
    
#     # ==================== 选择最佳初始帧 ====================
    
#     def select_best_first_frame(self):
#         """从候选帧中选择实例数量最大的作为初始帧"""
#         print("\n🔍 选择最佳初始帧...")
#         print(f"   候选帧: {self.first_frame_candidates}")
        
#         candidate_stats = []
        
#         for frame_num in self.first_frame_candidates:
#             if frame_num in self.instance_centers_3d:
#                 # 统计目标类实例数量
#                 target_instances = [inst_id for inst_id in self.instance_centers_3d[frame_num].keys()
#                                    if inst_id <= self.frame_mask_counts.get(frame_num, 0)]
#                 instance_count = len(target_instances)
                
#                 if instance_count > 0:
#                     candidate_stats.append({
#                         'frame': frame_num,
#                         'instance_count': instance_count,
#                         'has_depth': frame_num in self.instance_depths,
#                         'has_camera': frame_num in self.cameras
#                     })
#                     print(f"     帧 {frame_num:04d}: {instance_count} 个目标类实例")
#                 else:
#                     print(f"     帧 {frame_num:04d}: 没有目标类实例")
#             else:
#                 print(f"     帧 {frame_num:04d}: 没有3D中心点数据")
        
#         if not candidate_stats:
#             print("   ⚠️ 候选帧中都没有目标类实例，将使用第一个有实例的帧")
#             # 找第一个有实例的帧
#             for frame_num in sorted(self.instance_centers_3d.keys()):
#                 if frame_num in self.instance_centers_3d:
#                     target_instances = [inst_id for inst_id in self.instance_centers_3d[frame_num].keys()
#                                        if inst_id <= self.frame_mask_counts.get(frame_num, 0)]
#                     if target_instances:
#                         self.first_frame = frame_num
#                         print(f"   选择帧 {frame_num:04d} 作为初始帧 ({len(target_instances)} 个实例)")
#                         return
        
#         # 按实例数量排序，选择最大的
#         candidate_stats.sort(key=lambda x: -x['instance_count'])
#         best_frame = candidate_stats[0]['frame']
#         self.first_frame = best_frame
        
#         print(f"\n✅ 选择帧 {best_frame:04d} 作为初始帧 (实例数: {candidate_stats[0]['instance_count']})")
        
#         # 如果还有并列第二的，也显示出来
#         if len(candidate_stats) > 1 and candidate_stats[1]['instance_count'] == candidate_stats[0]['instance_count']:
#             print(f"   ⚠️ 注意: 帧 {candidate_stats[1]['frame']:04d} 也有相同的实例数")
    
#     # ==================== 颜色生成模块 ====================
    
#     def generate_colormap(self, num_colors):
#         """为全局ID生成颜色映射"""
#         colors = {}
        
#         if self.colormap == 'tab20':
#             import matplotlib.cm as cm
#             cmap = cm.get_cmap('tab20')
#             for i in range(1, num_colors + 1):
#                 rgba = cmap((i-1) % 20 / 20)
#                 colors[i] = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        
#         elif self.colormap == 'hsv':
#             for i in range(1, num_colors + 1):
#                 hue = (i-1) / max(num_colors, 1)
#                 rgb = plt.cm.hsv(hue)[:3]
#                 colors[i] = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
#         else:  # 'random'
#             for i in range(1, num_colors + 1):
#                 random.seed(i)
#                 colors[i] = (random.randint(0, 255), 
#                            random.randint(0, 255), 
#                            random.randint(0, 255))
        
#         return colors
    
#     def get_instance_color(self, global_id):
#         """获取全局ID对应的颜色"""
#         if global_id not in self.global_id_colors:
#             max_current = max(self.global_id_colors.keys()) if self.global_id_colors else 0
#             if global_id > max_current:
#                 new_colors = self.generate_colormap(global_id)
#                 self.global_id_colors.update(new_colors)
        
#         return self.global_id_colors.get(global_id, (128, 128, 128))
    
#     # ==================== 数据加载模块 ====================
    
#     def load_camera_parameters(self):
#         """加载相机参数"""
#         print("\n📷 加载相机参数...")
        
#         with open(self.camera_json, 'r') as f:
#             camera_data = json.load(f)
        
#         for cam_info in camera_data:
#             img_name = cam_info.get('img_name', '')
#             frame_str = Path(img_name).stem
            
#             match = re.search(r'(\d+)', frame_str)
#             if match:
#                 frame_num = int(match.group(1))
                
#                 # 新增：只保留前max_frames帧的相机参数
#                 if frame_num <= self.max_frames:
#                     fx = cam_info['fx']
#                     fy = cam_info['fy']
#                     width = cam_info['width']
#                     height = cam_info['height']
#                     cx = width / 2
#                     cy = height / 2
                    
#                     intrinsics = np.array([
#                         [fx, 0, cx],
#                         [0, fy, cy],
#                         [0, 0, 1]
#                     ])
                    
#                     rotation = np.array(cam_info['rotation'])
#                     position = np.array(cam_info['position'])
                    
#                     self.cameras[frame_num] = {
#                         'intrinsics': intrinsics,
#                         'rotation': rotation,
#                         'position': position,
#                         'width': width,
#                         'height': height,
#                         'fx': fx,
#                         'fy': fy,
#                         'cx': cx,
#                         'cy': cy,
#                         'frame': frame_num
#                     }
        
#         print(f"✅ 已加载 {len(self.cameras)} 帧相机参数 (前{self.max_frames}帧)")

#     def load_depth_maps(self):
#         """加载深度图并转换为米"""
#         print("\n📊 加载深度图...")
        
#         depth_files = []
#         if self.depth_format == '16bit':
#             depth_files.extend(sorted(Path(self.depth_dir).glob("*.png")))
#             depth_files.extend(sorted(Path(self.depth_dir).glob("*.tif")))
#         elif self.depth_format == 'npy':
#             depth_files.extend(sorted(Path(self.depth_dir).glob("*.npy")))
        
#         print(f"找到 {len(depth_files)} 个深度文件")
        
#         for depth_file in tqdm(depth_files, desc="加载深度图"):
#             frame_str = depth_file.stem
#             if frame_str.startswith('depth_'):
#                 frame_str = frame_str.replace('depth_', '')
            
#             match = re.search(r'(\d+)', frame_str)
#             if match:
#                 frame_num = int(match.group(1))
                
#                 # 新增：只保留前max_frames帧的深度图
#                 if frame_num <= self.max_frames:
#                     try:
#                         if self.depth_format == 'npy':
#                             depth_data = np.load(str(depth_file))
#                             self.depth_maps[frame_num] = depth_data
#                         else:
#                             depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
#                             if depth_img is not None:
#                                 if len(depth_img.shape) == 3:
#                                     depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
#                                 self.depth_maps[frame_num] = depth_img
#                     except Exception as e:
#                         print(f"警告: 加载深度文件 {depth_file} 时出错: {e}")
        
#         # 将深度图转换为米
#         print("\n🔄 转换深度图到米...")
#         for frame_num, depth_map in tqdm(self.depth_maps.items(), desc="转换深度"):
#             if self.depth_format == '16bit':
#                 depth_m = depth_map.astype(np.float32) / self.depth_scale
#             else:
#                 depth_m = depth_map.astype(np.float32)
            
#             # 标记无效深度
#             depth_m[(depth_map <= 0) | (depth_m >= self.max_depth)] = np.nan
            
#             self.depth_maps_meters[frame_num] = depth_m
        
#         print(f"✅ 成功加载 {len(self.depth_maps)} 帧深度图 (前{self.max_frames}帧)")
    
#     def extract_frame_number(self, filename):
#         """从文件名中提取帧号"""
#         match = re.search(r'mask_(\d+)', filename)
#         if match:
#             return int(match.group(1))
        
#         match = re.search(r'frame_(\d+)', filename)
#         if match:
#             return int(match.group(1))
        
#         match = re.search(r'(\d+)', filename)
#         if match:
#             return int(match.group(1))
        
#         return None

#     def load_masks_with_class_info(self):
#         """加载分割掩码，同时读取类别信息，只保留目标类别的实例"""
#         print("\n🎭 加载分割掩码和类别信息...")
#         print(f"   目标类别: {self.target_class}")
#         print(f"   固定类别: {list(self.fixed_class_ids.keys())} (ID映射: {self.fixed_class_ids})")
        
#         mask_files = list(self.mask_dir.glob("*.png"))
#         mask_files.extend(self.mask_dir.glob("*.tif"))
        
#         print(f"找到 {len(mask_files)} 个掩码文件")
        
#         # 首先找到所有对应的class_info文件
#         class_info_files = list(self.mask_dir.glob("class_info_*.json"))
#         class_info_map = {}
#         for info_file in class_info_files:
#             # 从class_info_xxx.json中提取基础名称
#             base_name = info_file.stem.replace('class_info_', '')
#             class_info_map[base_name] = info_file
        
#         print(f"找到 {len(class_info_map)} 个类别信息文件")
        
#         frame_file_map = {}
#         for mask_file in mask_files:
#             frame_num = self.extract_frame_number(mask_file.stem)
#             if frame_num is not None:
#                 # 新增：只保留前max_frames帧的掩码
#                 if frame_num <= self.max_frames:
#                     frame_file_map[frame_num] = mask_file
#             else:
#                 print(f"警告: 无法从文件名提取帧号: {mask_file.name}")
        
#         print(f"成功解析 {len(frame_file_map)} 个帧号 (前{self.max_frames}帧)")
        
#         sorted_frames = sorted(frame_file_map.keys())
#         if sorted_frames:
#             print(f"排序后的帧号范围: {sorted_frames[0]} - {sorted_frames[-1]}")
        
#         # 不再在这里设置第一帧，将在select_best_first_frame中设置
        
#         total_instances = 0
#         target_instances = 0
#         fixed_instances = {0: 0, 1: 0, 2: 0}
        
#         for frame_num in tqdm(sorted_frames, desc="加载掩码"):
#             mask_file = frame_file_map[frame_num]
            
#             # 查找对应的类别信息文件
#             base_name = mask_file.stem.replace('mask_', '')
#             if base_name in class_info_map:
#                 with open(class_info_map[base_name], 'r') as f:
#                     class_data = json.load(f)
                
#                 # 创建类别映射：instance_id -> class_id
#                 instance_to_class = {}
#                 for inst in class_data['instances']:
#                     instance_to_class[inst['instance_id']] = inst['class_id']
#             else:
#                 print(f"警告: 帧 {frame_num} 没有对应的类别信息文件")
#                 continue
            
#             mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
#             if mask is None:
#                 print(f"警告: 无法读取 {mask_file}")
#                 continue
                
#             if len(mask.shape) == 3:
#                 mask = mask[:, :, 0]
            
#             original_unique = np.unique(mask)
#             original_unique = original_unique[original_unique != 0]
#             original_count = len(original_unique)
            
#             self.frame_original_counts[frame_num] = original_count
            
#             # 根据类别信息分离实例
#             all_instance_ids = np.unique(mask)
#             all_instance_ids = all_instance_ids[all_instance_ids != 0]
            
#             # 分类实例
#             target_class_instances = []  # 目标类别的实例ID
#             fixed_class_instances = defaultdict(list)  # 固定类别的实例ID
            
#             for inst_id in all_instance_ids:
#                 if inst_id in instance_to_class:
#                     class_id = instance_to_class[inst_id]
#                     if class_id == self.target_class:
#                         target_class_instances.append(inst_id)
#                     elif class_id in self.fixed_class_ids:
#                         fixed_class_instances[class_id].append(inst_id)
#                 else:
#                     print(f"  警告: 实例 {inst_id} 在帧 {frame_num} 中没有类别信息")
            
#             # 创建新掩码，只包含目标类别的实例
#             new_mask = np.zeros_like(mask)
            
#             # 首先添加固定类别的实例（保留原始ID）
#             for class_id, inst_list in fixed_class_instances.items():
#                 fixed_global_id = self.fixed_class_ids[class_id]
#                 for inst_id in inst_list:
#                     new_mask[mask == inst_id] = fixed_global_id
#                     # 保存类别信息
#                     self.frame_class_info[frame_num][fixed_global_id] = class_id
#                     fixed_instances[class_id] += 1
#                     total_instances += 1
            
#             # 然后添加目标类别的实例（重新编号从1开始）
#             target_local_id = 1
#             for inst_id in target_class_instances:
#                 new_mask[mask == inst_id] = target_local_id
#                 # 保存类别信息
#                 self.frame_class_info[frame_num][target_local_id] = self.target_class
#                 target_local_id += 1
#                 target_instances += 1
#                 total_instances += 1
            
#             self.masks[frame_num] = new_mask
#             self.frame_mask_counts[frame_num] = target_local_id - 1  # 只统计目标类别的实例数
            
#             if len(self.masks) <= 5 or frame_num in self.first_frame_candidates:
#                 print(f"  帧 {frame_num:04d}: 目标类实例 {target_local_id-1}个, "
#                       f"固定类实例 C0:{len(fixed_class_instances[0])} "
#                       f"C1:{len(fixed_class_instances[1])} "
#                       f"C2:{len(fixed_class_instances[2])} "
#                       f"(原始总数: {original_count})")
        
#         print(f"\n✅ 成功加载 {len(self.masks)} 帧掩码 (前{self.max_frames}帧)")
#         print(f"   总实例数: {total_instances}")
#         print(f"   目标类别 {self.target_class} 实例数: {target_instances}")
#         print(f"   固定类别0实例数: {fixed_instances[0]}")
#         print(f"   固定类别1实例数: {fixed_instances[1]}")
#         print(f"   固定类别2实例数: {fixed_instances[2]}")
    
#     def initialize_fixed_class_points(self):
#         """初始化固定类别的全局点"""
#         print("\n🌟 初始化固定类别的全局点...")
        
#         for class_id, global_id in self.fixed_class_ids.items():
#             # 为每个固定类别创建一个虚拟的全局点
#             self.global_centers_3d[global_id] = {
#                 'point': [0, 0, 0],  # 虚拟点
#                 'frames': [],
#                 'instances': [],
#                 'distances': [],
#                 'num_views': 0,
#                 'confidence': 1.0,
#                 'is_fixed': True,
#                 'class_id': class_id
#             }
            
#             self.global_stats[global_id] = {
#                 'avg_depth': 0,
#                 'total_area': 0,
#                 'num_views': 0,
#                 'class_id': class_id
#             }
            
#             print(f"   初始化固定ID {global_id} (类别{class_id})")
        
#         # 更新颜色映射
#         self.global_id_colors.update(self.generate_colormap(max(self.fixed_class_ids.values())))
#         print(f"✅ 固定类别全局点初始化完成")
    
#     def extract_mask_features(self):
#         """提取掩码的轮廓和矩"""
#         print("\n🔍 提取掩码特征...")
        
#         for frame_num, mask in tqdm(self.masks.items(), desc="提取特征"):
#             self.mask_contours[frame_num] = {}
#             self.mask_moments[frame_num] = {}
            
#             instance_ids = np.unique(mask)
#             instance_ids = instance_ids[instance_ids != 0]
            
#             for instance_id in instance_ids:
#                 mask_binary = (mask == instance_id).astype(np.uint8) * 255
                
#                 # 提取轮廓
#                 contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                 if contours:
#                     self.mask_contours[frame_num][instance_id] = contours[0]
                
#                 # 计算矩
#                 moments = cv2.moments(mask_binary)
#                 self.mask_moments[frame_num][instance_id] = moments
        
#         print("✅ 掩码特征提取完成")
    
#     # ==================== 形状相似性计算 ====================
    
#     def compute_shape_similarity(self, mask1, mask2):
#         """
#         计算两个掩码的形状相似度
#         返回 0-1 之间的分数
#         """
#         # 方法1：Hu矩（旋转不变性）
#         moments1 = cv2.moments(mask1.astype(np.uint8) * 255)
#         moments2 = cv2.moments(mask2.astype(np.uint8) * 255)
        
#         if moments1['m00'] == 0 or moments2['m00'] == 0:
#             return 0.0
        
#         hu1 = cv2.HuMoments(moments1).flatten()
#         hu2 = cv2.HuMoments(moments2).flatten()
        
#         # 计算Hu矩的相似度
#         hu_diff = np.sum(np.abs(np.log(np.abs(hu1) + 1e-10) - np.log(np.abs(hu2) + 1e-10)))
#         hu_similarity = 1.0 / (1.0 + hu_diff)
        
#         # 方法2：面积相似度
#         area1 = np.sum(mask1 > 0)
#         area2 = np.sum(mask2 > 0)
#         area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
#         # 方法3：轮廓相似度（如果轮廓存在）
#         contour_similarity = 0.5
#         try:
#             # 计算轮廓的圆形度、伸长度等特征
#             pass
#         except:
#             pass
        
#         # 加权融合
#         shape_score = 0.5 * hu_similarity + 0.5 * area_ratio
        
#         return shape_score
    
#     # ==================== 前K个最小深度计算方法 ====================
    
#     def compute_top_k_depth_median(self, depths, k=None):
#         """
#         计算前K个最小深度的中位数
#         """
#         if k is None:
#             k = self.top_k_min_depths
        
#         if len(depths) == 0:
#             return None, 0, None, None
        
#         sorted_depths = np.sort(depths)
#         k_actual = min(k, len(sorted_depths))
#         top_k_depths = sorted_depths[:k_actual]
        
#         median_depth = np.median(top_k_depths)
#         min_depth = np.min(top_k_depths)
#         max_of_top_k = np.max(top_k_depths)
        
#         return median_depth, k_actual, min_depth, max_of_top_k
    
#     # ==================== 核心计算模块 ====================
    
#     def compute_instance_centers(self):
#         """
#         计算每个实例的中心点和深度中位数
#         只计算目标类别的实例
#         """
#         print("\n🎯 计算实例中心点和深度中位数...")
#         print(f"   只计算目标类别 {self.target_class} 的实例")
#         print(f"   最小实例面积: {self.min_instance_area} 像素")
        
#         if self.use_top_k_depths:
#             print(f"   深度计算方法: 使用前 {self.top_k_min_depths} 个最小深度值的中位数")
#         else:
#             print(f"   深度计算方法: 使用所有有效深度的中位数")
        
#         total_instances = 0
#         frames_with_depth = 0
#         instances_skipped_small_area = 0
#         instances_skipped_no_depth = 0
        
#         # 统计信息
#         depth_stats = {
#             'total_instances': 0,
#             'instances_with_top_k': 0,
#             'instances_using_all': 0,
#             'avg_k_used': 0,
#             'avg_pixel_count': 0,
#             'avg_min_depth': 0,
#             'avg_max_of_top_k': 0
#         }
        
#         depth_stats_file = self.depth_stats_dir / "depth_calculation_stats.txt"
        
#         for frame_num, mask in tqdm(self.masks.items(), desc="处理帧"):
#             self.frame_instances_with_depth[frame_num] = 0
            
#             if frame_num not in self.depth_maps_meters:
#                 print(f"  警告: 帧 {frame_num} 没有对应的深度图")
#                 continue
            
#             if frame_num not in self.cameras:
#                 print(f"  警告: 帧 {frame_num} 没有对应的相机参数")
#                 continue
            
#             depth_map = self.depth_maps_meters[frame_num]
            
#             if depth_map.shape != mask.shape:
#                 print(f"  警告: 帧 {frame_num} 深度图尺寸 {depth_map.shape} 与掩码尺寸 {mask.shape} 不匹配")
#                 continue
            
#             frames_with_depth += 1
            
#             instance_ids = np.unique(mask)
#             instance_ids = instance_ids[instance_ids != 0]
            
#             for instance_id in instance_ids:
#                 # 只处理目标类别的实例（即ID小于等于frame_mask_counts[frame_num]的实例）
#                 if instance_id <= self.frame_mask_counts.get(frame_num, 0):
#                     # 这是目标类别的实例
#                     pass
#                 else:
#                     # 这是固定类别的实例，跳过计算
#                     continue
                
#                 mask_binary = (mask == instance_id)
                
#                 y_indices, x_indices = np.where(mask_binary)
                
#                 if len(y_indices) < self.min_instance_area:
#                     instances_skipped_small_area += 1
#                     continue
                
#                 center_x = int(np.mean(x_indices))
#                 center_y = int(np.mean(y_indices))
                
#                 depths_in_mask = depth_map[mask_binary]
#                 valid_depths = depths_in_mask[~np.isnan(depths_in_mask)]
                
#                 has_valid_depth = False
                
#                 if len(valid_depths) == 0:
#                     # 尝试使用中心点深度
#                     if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
#                         center_depth = depth_map[center_y, center_x]
#                         if not np.isnan(center_depth):
#                             valid_depths = np.array([center_depth])
#                             has_valid_depth = True
#                 else:
#                     has_valid_depth = True
                
#                 if not has_valid_depth:
#                     self.instance_centers_2d[frame_num][int(instance_id)] = (center_x, center_y)
#                     self.instance_stats[frame_num][int(instance_id)] = {
#                         'area': len(y_indices),
#                         'has_depth': False,
#                         'num_valid_depths': 0,
#                         'class_id': self.target_class
#                     }
#                     instances_skipped_no_depth += 1
#                     continue
                
#                 # 保存所有有效深度值
#                 self.instance_all_depths[frame_num][int(instance_id)] = valid_depths.tolist()
                
#                 # 计算深度中位数
#                 if self.use_top_k_depths and len(valid_depths) > 0:
#                     depth_median, k_used, min_depth, max_of_top_k = self.compute_top_k_depth_median(
#                         valid_depths, self.top_k_min_depths
#                     )
                    
#                     if depth_median is not None:
#                         depth_stats['instances_with_top_k'] += 1
#                         depth_stats['avg_k_used'] += k_used
#                         depth_std = np.std(valid_depths[:k_used] if k_used < len(valid_depths) else valid_depths)
#                     else:
#                         depth_median = float(np.median(valid_depths))
#                         depth_std = float(np.std(valid_depths))
#                         depth_stats['instances_using_all'] += 1
#                         k_used = len(valid_depths)
#                         min_depth = float(np.min(valid_depths))
#                         max_of_top_k = float(np.max(valid_depths[:min(self.top_k_min_depths, len(valid_depths))]))
#                 else:
#                     depth_median = float(np.median(valid_depths))
#                     depth_std = float(np.std(valid_depths))
#                     k_used = len(valid_depths)
#                     min_depth = float(np.min(valid_depths))
#                     max_of_top_k = float(np.max(valid_depths))
#                     depth_stats['instances_using_all'] += 1
                
#                 # 更新统计
#                 depth_stats['total_instances'] += 1
#                 depth_stats['avg_pixel_count'] += len(valid_depths)
#                 depth_stats['avg_min_depth'] += min_depth
#                 depth_stats['avg_max_of_top_k'] += max_of_top_k
                
#                 self.instance_centers_2d[frame_num][int(instance_id)] = (center_x, center_y)
#                 self.instance_depths[frame_num][int(instance_id)] = depth_median
#                 self.instance_stats[frame_num][int(instance_id)] = {
#                     'area': len(y_indices),
#                     'depth_median': depth_median,
#                     'depth_std': depth_std,
#                     'num_valid_depths': len(valid_depths),
#                     'has_depth': True,
#                     'depth_calculation_method': 'top_k' if (self.use_top_k_depths and k_used <= self.top_k_min_depths) else 'all',
#                     'top_k_used': k_used if self.use_top_k_depths else len(valid_depths),
#                     'top_k_ratio': (k_used / len(valid_depths)) if len(valid_depths) > 0 else 0,
#                     'min_depth': min_depth,
#                     'max_of_top_k': max_of_top_k,
#                     'center_point': (center_x, center_y),
#                     'class_id': self.target_class
#                 }
                
#                 self.frame_instances_with_depth[frame_num] += 1
#                 total_instances += 1
        
#         # 计算平均值
#         if depth_stats['total_instances'] > 0:
#             depth_stats['avg_k_used'] /= depth_stats['total_instances']
#             depth_stats['avg_pixel_count'] /= depth_stats['total_instances']
#             depth_stats['avg_min_depth'] /= depth_stats['total_instances']
#             depth_stats['avg_max_of_top_k'] /= depth_stats['total_instances']
        
#         print(f"\n✅ 计算完成:")
#         print(f"   处理帧数: {frames_with_depth}/{len(self.masks)}")
#         print(f"   目标类实例数 (有有效深度): {total_instances}")
#         print(f"   因面积过小跳过的实例: {instances_skipped_small_area}")
#         print(f"   因无有效深度跳过的实例: {instances_skipped_no_depth}")
        
#         print(f"\n📊 深度计算统计:")
#         print(f"   使用前K个最小深度的实例: {depth_stats['instances_with_top_k']}")
#         print(f"   使用全部深度的实例: {depth_stats['instances_using_all']}")
#         print(f"   平均使用的像素数: {depth_stats['avg_k_used']:.1f}")
#         print(f"   实例平均总像素数: {depth_stats['avg_pixel_count']:.1f}")
#         print(f"   平均最小深度: {depth_stats['avg_min_depth']:.3f}m")
#         print(f"   平均前K个最大深度: {depth_stats['avg_max_of_top_k']:.3f}m")
        
#         # 保存深度统计信息到文件
#         with open(depth_stats_file, 'w') as f:
#             f.write("="*60 + "\n")
#             f.write("深度计算统计报告\n")
#             f.write("="*60 + "\n\n")
            
#             f.write(f"配置参数:\n")
#             f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
#             if self.use_top_k_depths:
#                 f.write(f"  K值: {self.top_k_min_depths}\n")
#             f.write(f"  最小实例面积: {self.min_instance_area} 像素\n\n")
            
#             f.write(f"总体统计:\n")
#             f.write(f"  总实例数: {total_instances}\n")
#             f.write(f"  使用前K个最小深度的实例: {depth_stats['instances_with_top_k']}\n")
#             f.write(f"  使用全部深度的实例: {depth_stats['instances_using_all']}\n")
#             f.write(f"  平均使用的像素数: {depth_stats['avg_k_used']:.1f}\n")
#             f.write(f"  实例平均总像素数: {depth_stats['avg_pixel_count']:.1f}\n")
#             f.write(f"  平均最小深度: {depth_stats['avg_min_depth']:.3f}m\n")
#             f.write(f"  平均前K个最大深度: {depth_stats['avg_max_of_top_k']:.3f}m\n")
        
#         print(f"\n✅ 深度统计已保存: {depth_stats_file}")
    
#     def convert_to_3d_centers(self):
#         """将2D中心点转换为3D坐标"""
#         print("\n🔄 转换2D中心点到3D坐标...")
        
#         total_converted = 0
#         total_skipped = 0
        
#         for frame_num in tqdm(self.instance_centers_2d.keys(), desc="转换"):
#             if frame_num not in self.cameras:
#                 total_skipped += len(self.instance_centers_2d[frame_num])
#                 continue
            
#             cam = self.cameras[frame_num]
#             K = cam['intrinsics']
#             R = cam['rotation']
#             C = cam['position']
            
#             K_inv = np.linalg.inv(K)
            
#             for instance_id, (x, y) in self.instance_centers_2d[frame_num].items():
#                 if instance_id not in self.instance_depths[frame_num]:
#                     total_skipped += 1
#                     continue
                
#                 depth = self.instance_depths[frame_num][instance_id]
                
#                 point_2d_homo = np.array([x, y, 1.0])
                
#                 ray_camera = K_inv @ point_2d_homo
#                 ray_camera = ray_camera / np.linalg.norm(ray_camera)
                
#                 point_camera = ray_camera * depth
                
#                 point_world = R @ point_camera + C
                
#                 self.instance_centers_3d[frame_num][instance_id] = point_world.tolist()
#                 total_converted += 1
        
#         print(f"✅ 转换完成:")
#         print(f"   成功转换: {total_converted} 个实例")
#         print(f"   跳过 (无深度或相机参数): {total_skipped} 个实例")
    
#     # ==================== 初始化第一帧 ====================
    
#     def initialize_with_first_frame(self):
#         """用第一帧初始化全局ID（只处理目标类别的实例）"""
#         if self.first_frame is None:
#             print("❌ 没有找到第一帧")
#             return None
        
#         print(f"\n🌟 用第一帧初始化全局ID: {self.first_frame:04d}")
        
#         if self.first_frame not in self.instance_centers_3d:
#             print(f"❌ 第一帧 {self.first_frame} 没有3D中心点")
#             return None
        
#         instance_count = 0
#         for instance_id in self.instance_centers_3d[self.first_frame].keys():
#             # 只处理目标类别的实例
#             if instance_id > self.frame_mask_counts.get(self.first_frame, 0):
#                 continue
                
#             key = (self.first_frame, instance_id)
#             if key not in self.global_instance_map:
#                 global_id = self.global_instance_counter
#                 self.global_instance_counter += 1
                
#                 self.global_instance_map[key] = global_id
                
#                 point = self.instance_centers_3d[self.first_frame][instance_id]
#                 stats = self.instance_stats[self.first_frame][instance_id]
                
#                 self.global_centers_3d[global_id] = {
#                     'point': point,
#                     'frames': [self.first_frame],
#                     'instances': [(self.first_frame, instance_id)],
#                     'distances': [],
#                     'num_views': 1,
#                     'confidence': 1.0,
#                     'is_fixed': False,
#                     'class_id': self.target_class
#                 }
                
#                 # 保存掩码
#                 if self.first_frame in self.masks and instance_id in np.unique(self.masks[self.first_frame]):
#                     mask = (self.masks[self.first_frame] == instance_id)
#                     self.global_masks[global_id] = {self.first_frame: mask}
                
#                 self.global_stats[global_id] = {
#                     'avg_depth': stats['depth_median'],
#                     'total_area': stats['area'],
#                     'num_views': 1,
#                     'class_id': self.target_class
#                 }
                
#                 # 初始化运动轨迹
#                 self.motion_trajectories[global_id].append((self.first_frame, point))
                
#                 instance_count += 1
#                 print(f"   初始ID {global_id:3d}: 帧{self.first_frame:04d}:{instance_id} "
#                       f"(深度: {stats['depth_median']:.3f}m)")
        
#         self.frame_matched_counts[self.first_frame] = instance_count
        
#         # 更新颜色映射（包括固定类别）
#         max_id = max(max(self.fixed_class_ids.values()), self.global_instance_counter - 1)
#         self.global_id_colors.update(self.generate_colormap(max_id))
        
#         print(f"✅ 初始化完成: 分配了 {instance_count} 个全局ID (目标类实例)")
        
#         return self.first_frame
    
#     # ==================== 数据驱动的运动预测 ====================
    
#     def predict_position_by_motion(self, global_id, target_frame):
#         """
#         数据驱动的运动预测
#         使用加权移动平均，不做物理假设
#         """
#         if not self.use_motion_prior:
#             return None, 0
        
#         # 跳过固定类别的点
#         if global_id in self.fixed_class_ids.values():
#             return None, 0
        
#         if global_id not in self.motion_trajectories:
#             return None, 0
        
#         trajectory = self.motion_trajectories[global_id]
#         if len(trajectory) < 2:
#             return None, 0
        
#         # 获取最近几帧
#         last_frame, last_pos = trajectory[-1]
        
#         # 方法1：线性预测（基于最近两帧）
#         if len(trajectory) >= 2:
#             prev_frame, prev_pos = trajectory[-2]
#             velocity = np.array(last_pos) - np.array(prev_pos)
#             linear_pred = np.array(last_pos) + velocity
        
#         # 方法2：加权移动平均（考虑更多历史）
#         if len(trajectory) >= 3:
#             weights = [0.5, 0.3, 0.2]  # 越近权重越大
#             weighted_sum = np.zeros(3)
#             weight_total = 0
            
#             for i, (_, pos) in enumerate(reversed(trajectory[-3:])):
#                 weighted_sum += weights[i] * np.array(pos)
#                 weight_total += weights[i]
            
#             wma_pred = weighted_sum / weight_total
            
#             # 融合两种预测
#             if len(trajectory) >= 3:
#                 predicted_pos = 0.7 * linear_pred + 0.3 * wma_pred
#             else:
#                 predicted_pos = linear_pred
#         else:
#             predicted_pos = linear_pred
        
#         # 置信度：基于轨迹长度和运动平滑度
#         length_conf = min(0.9, len(trajectory) * 0.1)
        
#         # 运动平滑度（连续两帧位移的一致性）
#         if len(trajectory) >= 3:
#             v1 = np.array(trajectory[-1][1]) - np.array(trajectory[-2][1])
#             v2 = np.array(trajectory[-2][1]) - np.array(trajectory[-3][1])
            
#             # 计算方向一致性（余弦相似度）
#             if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
#                 cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#                 smooth_conf = max(0.5, (cos_sim + 1) / 2)
#             else:
#                 smooth_conf = 0.7
#         else:
#             smooth_conf = 0.7
        
#         confidence = length_conf * smooth_conf
        
#         return predicted_pos.tolist(), confidence
    
#     def get_motion_predictions(self, frame_num, global_ids):
#         """
#         获取所有全局点的运动预测（跳过固定类别）
#         """
#         predictions = []
        
#         for global_id in global_ids:
#             # 跳过固定类别的点
#             if global_id in self.fixed_class_ids.values():
#                 continue
                
#             predicted_pos, confidence = self.predict_position_by_motion(global_id, frame_num)
#             if predicted_pos is not None:
#                 predictions.append({
#                     'global_id': global_id,
#                     'predicted_pos': predicted_pos,
#                     'confidence': confidence
#                 })
        
#         # 按置信度排序
#         predictions.sort(key=lambda x: -x['confidence'])
        
#         return predictions
    
#     # ==================== 协同匹配模块（核心） ====================
    
#     def match_with_coherence(self, frame_num):
#         """
#         协同匹配：多证据融合
#         只匹配目标类别的实例
#         """
#         if frame_num not in self.instance_centers_3d:
#             return {}
        
#         # 只获取目标类别的实例
#         current_instances = {}
#         for inst_id, point in self.instance_centers_3d[frame_num].items():
#             if inst_id <= self.frame_mask_counts.get(frame_num, 0):
#                 current_instances[inst_id] = point
        
#         if not current_instances:
#             return {}
        
#         # 获取全局点（跳过固定类别）
#         global_points = []
#         global_ids = []
#         for gid, info in self.global_centers_3d.items():
#             if gid in self.fixed_class_ids.values():
#                 continue
#             global_points.append(info['point'])
#             global_ids.append(gid)
        
#         if not global_points:
#             return {}
        
#         global_points = np.array(global_points)
#         frame_points = np.array(list(current_instances.values()))
#         frame_ids = list(current_instances.keys())
        
#         # 计算距离矩阵
#         distances = cdist(frame_points, global_points)
        
#         print(f"\n  🔍 协同匹配: 帧 {frame_num:04d} ({len(frame_ids)}目标类实例) <-> 全局点 ({len(global_ids)}个)")
        
#         # ========== 第1步：收集所有证据 ==========
#         match_evidence = defaultdict(list)  # (local_id, global_id) -> [证据列表]
        
#         # 证据1：运动连续性预测
#         motion_predictions = self.get_motion_predictions(frame_num, global_ids)
#         for pred in motion_predictions:
#             global_id = pred['global_id']
#             predicted_pos = pred['predicted_pos']
#             confidence = pred['confidence']
            
#             for i, local_id in enumerate(frame_ids):
#                 dist = np.linalg.norm(np.array(predicted_pos) - frame_points[i])
#                 if dist < self.motion_threshold * 2:
#                     # 运动得分：置信度 * 距离衰减
#                     motion_score = confidence * (1.0 / (1.0 + dist * 5))
#                     match_evidence[(local_id, global_id)].append({
#                         'type': 'motion',
#                         'confidence': confidence,
#                         'distance': dist,
#                         'score': motion_score,
#                         'details': f'运动预测(置信度:{confidence:.2f}, 距离:{dist:.3f}m)'
#                     })
        
#         # 证据2：双向空间匹配
#         for i, local_id in enumerate(frame_ids):
#             for j, global_id in enumerate(global_ids):
#                 dist = distances[i, j]
#                 if dist < self.spatial_threshold * 1.5:
#                     is_forward_nn = (np.argmin(distances[i]) == j)
#                     is_backward_nn = (np.argmin(distances[:, j]) == i)
                    
#                     if is_forward_nn and is_backward_nn:
#                         bid_score = 1.0 / (1.0 + dist * 5) * 1.5
#                         evidence_type = 'perfect_bidirectional'
#                         details = f'完美双向(距离:{dist:.3f}m)'
#                     elif is_forward_nn:
#                         bid_score = 1.0 / (1.0 + dist * 5) * 1.0
#                         evidence_type = 'forward_nn'
#                         details = f'正向最近邻(距离:{dist:.3f}m)'
#                     elif is_backward_nn:
#                         bid_score = 1.0 / (1.0 + dist * 5) * 1.0
#                         evidence_type = 'backward_nn'
#                         details = f'反向最近邻(距离:{dist:.3f}m)'
#                     else:
#                         bid_score = 1.0 / (1.0 + dist * 5) * 0.6
#                         evidence_type = 'distance'
#                         details = f'距离证据(距离:{dist:.3f}m)'
                    
#                     match_evidence[(local_id, global_id)].append({
#                         'type': 'bidirectional',
#                         'subtype': evidence_type,
#                         'distance': dist,
#                         'score': bid_score,
#                         'details': details
#                     })
        
#         # 证据3：形状相似性
#         for i, local_id in enumerate(frame_ids):
#             # 当前帧的掩码
#             current_mask = (self.masks[frame_num] == local_id)
            
#             for j, global_id in enumerate(global_ids):
#                 # 获取这个全局点最近一帧的掩码
#                 last_frame, last_local_id = self.get_last_occurrence(global_id)
#                 if last_frame in self.masks and last_local_id in np.unique(self.masks[last_frame]):
#                     last_mask = (self.masks[last_frame] == last_local_id)
                    
#                     # 计算形状相似度
#                     shape_score = self.compute_shape_similarity(current_mask, last_mask)
                    
#                     if shape_score > 0.5:  # 阈值
#                         match_evidence[(local_id, global_id)].append({
#                             'type': 'shape',
#                             'score': shape_score,
#                             'details': f'形状相似(得分:{shape_score:.2f})'
#                         })
        
#         # 证据4：历史匹配
#         for (local_id, global_id) in list(match_evidence.keys()):
#             key = (frame_num, local_id)
#             if key in self.global_instance_map and self.global_instance_map[key] == global_id:
#                 match_evidence[(local_id, global_id)].append({
#                     'type': 'history',
#                     'score': 0.8,
#                     'details': '历史匹配'
#                 })
        
#         # ========== 第2步：计算综合得分 ==========
#         match_candidates = []
        
#         for (local_id, global_id), evidences in match_evidence.items():
#             total_score = 0
#             motion_count = 0
#             bidirectional_count = 0
#             shape_count = 0
#             evidence_details = []
            
#             for ev in evidences:
#                 if ev['type'] == 'motion':
#                     total_score += ev['score'] * self.motion_weight
#                     motion_count += 1
#                 elif ev['type'] == 'bidirectional':
#                     total_score += ev['score'] * self.bidirectional_weight
#                     bidirectional_count += 1
#                 elif ev['type'] == 'shape':
#                     total_score += ev['score'] * self.shape_weight
#                     shape_count += 1
#                 elif ev['type'] == 'history':
#                     total_score += ev['score'] * 0.1  # 历史权重低
                
#                 evidence_details.append(ev['details'])
            
#             # 协同效应：多种证据加分
#             evidence_types = (motion_count > 0) + (bidirectional_count > 0) + (shape_count > 0)
#             if evidence_types >= 2:
#                 total_score *= 1.2
#                 match_type = '协同匹配'
#             elif motion_count > 0:
#                 match_type = '运动匹配'
#             elif bidirectional_count > 0:
#                 match_type = '双向匹配'
#             elif shape_count > 0:
#                 match_type = '形状匹配'
#             else:
#                 match_type = '历史匹配'
            
#             match_candidates.append({
#                 'local_id': local_id,
#                 'global_id': global_id,
#                 'score': total_score,
#                 'match_type': match_type,
#                 'motion_count': motion_count,
#                 'bidirectional_count': bidirectional_count,
#                 'shape_count': shape_count,
#                 'evidence_count': len(evidences),
#                 'evidence_details': evidence_details
#             })
        
#         # 按得分排序
#         match_candidates.sort(key=lambda x: -x['score'])
        
#         # ========== 第3步：解决冲突 ==========
#         final_matches = {}
#         used_globals = set()
#         used_locals = set()
#         matching_details = []
        
#         # 先处理协同匹配
#         for cand in match_candidates:
#             if cand['match_type'] == '协同匹配':
#                 if cand['local_id'] not in used_locals and cand['global_id'] not in used_globals:
#                     final_matches[cand['local_id']] = {
#                         'global_id': cand['global_id'],
#                         'confidence': cand['score'],
#                         'match_type': cand['match_type'],
#                         'evidence_count': cand['evidence_count']
#                     }
#                     used_locals.add(cand['local_id'])
#                     used_globals.add(cand['global_id'])
#                     self.matching_stats['cooperative_matches'] += 1
                    
#                     print(f"    🤝 协同匹配: 本地{cand['local_id']} <-> 全局{cand['global_id']} "
#                           f"(得分:{cand['score']:.3f})")
        
#         # 再处理单一证据的
#         for cand in match_candidates:
#             if cand['local_id'] not in used_locals and cand['global_id'] not in used_globals:
#                 # 检查本地冲突
#                 conflict = False
#                 for selected_local in final_matches:
#                     i1 = frame_ids.index(selected_local)
#                     i2 = frame_ids.index(cand['local_id'])
#                     local_dist = np.linalg.norm(frame_points[i1] - frame_points[i2])
                    
#                     if local_dist < self.coherence_threshold:
#                         conflict = True
#                         break
                
#                 if not conflict:
#                     final_matches[cand['local_id']] = {
#                         'global_id': cand['global_id'],
#                         'confidence': cand['score'],
#                         'match_type': cand['match_type'],
#                         'evidence_count': cand['evidence_count']
#                     }
#                     used_locals.add(cand['local_id'])
#                     used_globals.add(cand['global_id'])
                    
#                     if cand['match_type'] == '运动匹配':
#                         self.matching_stats['motion_matches'] += 1
#                     elif cand['match_type'] == '双向匹配':
#                         self.matching_stats['bidirectional_matches'] += 1
#                     elif cand['match_type'] == '形状匹配':
#                         self.matching_stats['shape_matches'] += 1
                    
#                     print(f"    { '🌀' if cand['match_type']=='运动匹配' else '🔍' } {cand['match_type']}: 本地{cand['local_id']} <-> 全局{cand['global_id']} "
#                           f"(得分:{cand['score']:.3f})")
        
#         # 处理未匹配的实例
#         for local_id in frame_ids:
#             if local_id not in used_locals:
#                 stats = self.instance_stats[frame_num].get(local_id, {})
                
#                 i = frame_ids.index(local_id)
#                 min_dist = np.min(distances[i]) if len(distances[i]) > 0 else float('inf')
#                 min_dist_global = global_ids[np.argmin(distances[i])] if min_dist < float('inf') else None
                
#                 # 检查是否有证据
#                 has_evidence = False
#                 for (lid, gid), evidences in match_evidence.items():
#                     if lid == local_id:
#                         has_evidence = True
#                         break
                
#                 if min_dist < self.spatial_threshold:
#                     if has_evidence:
#                         reason = f"有证据但被竞争 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
#                     else:
#                         reason = f"距离近但证据不足 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
#                 else:
#                     reason = f"距离太远 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
                
#                 matching_details.append({
#                     'frame': frame_num,
#                     'local_id': local_id,
#                     'global_id': None,
#                     'match_status': 'UNMATCHED',
#                     'reason': reason,
#                     'depth_method': stats.get('depth_calculation_method', 'unknown'),
#                     'top_k_used': stats.get('top_k_used', 0),
#                     'total_pixels': stats.get('num_valid_depths', 0)
#                 })
                
#                 print(f"    ⚠️ 未匹配: 本地{local_id} ({reason})")
        
#         self.save_matching_details(frame_num, matching_details)
        
#         print(f"    最终接受: {len(final_matches)}/{len(frame_ids)} 个匹配")
#         print(f"    协同匹配: {self.matching_stats['cooperative_matches']} | 运动匹配: {self.matching_stats['motion_matches']} | 双向匹配: {self.matching_stats['bidirectional_matches']} | 形状匹配: {self.matching_stats['shape_matches']}")
        
#         self.frame_matched_counts[frame_num] = len(final_matches)
        
#         return final_matches
    
#     def get_last_occurrence(self, global_id):
#         """获取全局点最近一次出现的帧和本地ID"""
#         if global_id in self.global_centers_3d:
#             instances = self.global_centers_3d[global_id]['instances']
#             if instances:
#                 return instances[-1]  # (frame_num, local_id)
#         return None, None
    
#     def update_global_point(self, global_id, new_point, frame_num, local_id, distance):
#         """更新全局点的3D位置和轨迹"""
#         # 跳过固定类别的点
#         if global_id in self.fixed_class_ids.values():
#             return new_point
        
#         info = self.global_centers_3d[global_id]
#         current_point = np.array(info['point'])
#         current_views = info['num_views']
        
#         new_avg_point = (current_point * current_views + new_point) / (current_views + 1)
        
#         old_confidence = info.get('confidence', 0.5)
#         new_confidence = 1.0 / (1.0 + np.mean(info['distances'] + [distance]) if info['distances'] else distance)
#         info['confidence'] = (old_confidence * current_views + new_confidence) / (current_views + 1)
        
#         info['point'] = new_avg_point.tolist()
#         info['frames'].append(frame_num)
#         info['instances'].append((frame_num, local_id))
#         info['distances'].append(distance)
#         info['num_views'] = current_views + 1
        
#         # 保存掩码
#         if frame_num in self.masks and local_id in np.unique(self.masks[frame_num]):
#             mask = (self.masks[frame_num] == local_id)
#             if global_id not in self.global_masks:
#                 self.global_masks[global_id] = {}
#             self.global_masks[global_id][frame_num] = mask
        
#         # 更新运动轨迹
#         self.motion_trajectories[global_id].append((frame_num, new_point.tolist()))
#         if len(self.motion_trajectories[global_id]) > 20:
#             self.motion_trajectories[global_id].pop(0)
        
#         stats = self.instance_stats[frame_num][local_id]
#         self.global_stats[global_id]['num_views'] += 1
#         self.global_stats[global_id]['total_area'] += stats['area']
#         self.global_stats[global_id]['avg_depth'] = (
#             (self.global_stats[global_id]['avg_depth'] * current_views + stats['depth_median']) / 
#             (current_views + 1)
#         )
        
#         return new_avg_point
    
#     # ==================== 像素深度保存模块 ====================
    
#     def save_pixel_depths_to_txt(self):
#         """保存指定帧的像素深度数据"""
#         if not self.pixel_depth_enabled:
#             return
        
#         print("\n📝 保存像素深度数据到txt文件...")
#         print(f"   要保存的帧: {self.frames_to_save}")
        
#         valid_frames = []
#         for frame_num in self.frames_to_save:
#             # 新增：只保存前max_frames帧内的像素深度
#             if frame_num <= self.max_frames and frame_num in self.masks and frame_num in self.depth_maps_meters:
#                 valid_frames.append(frame_num)
#             else:
#                 print(f"   ⚠️ 帧 {frame_num} 超出处理范围或不存在，跳过")
        
#         if not valid_frames:
#             print("   ⚠️ 没有有效的帧可保存")
#             return
        
#         total_instances_saved = 0
#         total_pixels_saved = 0
        
#         for frame_num in tqdm(valid_frames, desc="保存像素深度"):
#             mask = self.masks[frame_num]
#             depth_map = self.depth_maps_meters[frame_num]
            
#             instance_ids = np.unique(mask)
#             instance_ids = instance_ids[instance_ids != 0]
            
#             frame_pixel_count = 0
#             frame_instance_count = 0
            
#             if self.separate_files:
#                 for instance_id in instance_ids:
#                     mask_binary = (mask == instance_id)
#                     y_indices, x_indices = np.where(mask_binary)
                    
#                     valid_pixels = []
#                     valid_depths = []
#                     for y, x in zip(y_indices, x_indices):
#                         depth_val = depth_map[y, x]
#                         if not np.isnan(depth_val) and self.pixel_min_depth <= depth_val <= self.pixel_max_depth:
#                             valid_pixels.append((y, x, depth_val))
#                             valid_depths.append(depth_val)
                    
#                     if not valid_pixels:
#                         continue
                    
#                     valid_pixels.sort(key=lambda p: p[2])
                    
#                     if self.use_top_k_depths and len(valid_depths) > 0:
#                         depth_median_top_k, k_used, depth_min, depth_max_top_k = self.compute_top_k_depth_median(
#                             valid_depths, self.top_k_min_depths
#                         )
#                         depth_median = depth_median_top_k
#                     else:
#                         depth_median = np.median(valid_depths)
#                         depth_min = np.min(valid_depths)
#                         depth_max_top_k = np.max(valid_depths)
#                         k_used = len(valid_depths)
                    
#                     depth_mean = np.mean(valid_depths)
#                     depth_std = np.std(valid_depths)
                    
#                     key = (frame_num, int(instance_id))
#                     global_id = self.global_instance_map.get(key, instance_id if instance_id in self.fixed_class_ids.values() else None)
                    
#                     stats = self.instance_stats[frame_num].get(int(instance_id), {})
#                     depth_method = stats.get('depth_calculation_method', 'unknown')
                    
#                     class_id = self.frame_class_info[frame_num].get(int(instance_id), 'unknown')
                    
#                     filename = f"frame{frame_num:04d}_instance{instance_id}_global{global_id if global_id else 'unmatched'}_class{class_id}.txt"
#                     filepath = self.pixel_depth_data_dir / filename
                    
#                     with open(filepath, 'w') as f:
#                         f.write(f"# ========== 实例深度数据 ==========\n")
#                         f.write(f"# Frame: {frame_num}\n")
#                         f.write(f"# Instance ID (local): {instance_id}\n")
#                         f.write(f"# Global ID: {global_id if global_id else 'unmatched'}\n")
#                         f.write(f"# Class ID: {class_id}\n")
#                         f.write(f"# Pixel count: {len(valid_pixels)}\n")
#                         f.write(f"# Depth method: {depth_method}\n")
#                         f.write(f"# Top K used: {k_used}/{len(valid_pixels)}\n")
#                         f.write(f"# Depth median: {depth_median:.{self.depth_decimals}f} m\n")
#                         f.write(f"# Depth mean: {depth_mean:.{self.depth_decimals}f} m\n")
#                         f.write(f"# Depth std: {depth_std:.{self.depth_decimals}f} m\n")
#                         f.write(f"# Depth range: {depth_min:.{self.depth_decimals}f} - {depth_max_top_k:.{self.depth_decimals}f} m\n")
#                         f.write("#" + "-"*50 + "\n")
                        
#                         for y, x, depth_val in valid_pixels:
#                             if self.include_coordinates:
#                                 f.write(f"{y:4d} {x:4d} {depth_val:.{self.depth_decimals}f}\n")
#                             else:
#                                 f.write(f"{depth_val:.{self.depth_decimals}f}\n")
                    
#                     frame_pixel_count += len(valid_pixels)
#                     frame_instance_count += 1
#                     total_pixels_saved += len(valid_pixels)
            
#             if self.save_visualization:
#                 self.save_frame_visualization(frame_num)
            
#             print(f"   帧 {frame_num:04d}: 保存了 {frame_instance_count} 个实例, {frame_pixel_count} 个像素")
        
#         print(f"✅ 像素深度数据保存完成: {total_instances_saved} 实例, {total_pixels_saved} 像素")
    
#     def save_frame_visualization(self, frame_num):
#         """保存指定帧的可视化图像"""
#         if frame_num not in self.masks:
#             return
        
#         mask = self.masks[frame_num]
#         h, w = mask.shape
        
#         color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
#         for local_id in np.unique(mask):
#             if local_id == 0:
#                 continue
            
#             key = (frame_num, int(local_id))
#             if key in self.global_instance_map:
#                 global_id = self.global_instance_map[key]
#                 color = self.get_instance_color(global_id)
#                 color_mask[mask == local_id] = color
#             elif local_id in self.fixed_class_ids.values():
#                 # 固定类别的实例
#                 color = self.get_instance_color(local_id)
#                 color_mask[mask == local_id] = color
#             else:
#                 color_mask[mask == local_id] = self.unmatched_color
        
#         font = cv2.FONT_HERSHEY_SIMPLEX
        
#         for local_id in np.unique(mask):
#             if local_id == 0:
#                 continue
            
#             if frame_num in self.instance_centers_2d and local_id in self.instance_centers_2d[frame_num]:
#                 center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                
#                 depth_value = None
#                 if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
#                     depth_value = self.instance_depths[frame_num][local_id]
                
#                 key = (frame_num, int(local_id))
#                 global_id = self.global_instance_map.get(key)
                
#                 class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
                
#                 texts = []
#                 if global_id is not None:
#                     texts.append(f"ID:{global_id}")
#                 elif local_id in self.fixed_class_ids.values():
#                     texts.append(f"ID:{local_id}(C{class_id})")
#                 if depth_value is not None:
#                     texts.append(f"{depth_value:.2f}m")
                
#                 if not texts:
#                     continue
                
#                 full_text = " ".join(texts)
                
#                 font_scale = self.id_font_scale if (global_id or local_id in self.fixed_class_ids.values()) else self.depth_font_scale
#                 thickness = 1
                
#                 (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                
#                 text_x = int(center_x - text_width / 2)
#                 text_y = int(center_y + text_height / 2)
                
#                 if 0 <= text_x < w and 0 <= text_y < h:
#                     padding = 2
#                     cv2.rectangle(color_mask, 
#                                 (text_x - padding, text_y - text_height - padding),
#                                 (text_x + text_width + padding, text_y + padding),
#                                 (0, 0, 0), -1)
                    
#                     cv2.putText(color_mask, full_text, (text_x, text_y),
#                               font, font_scale, self.text_color, thickness)
        
#         info_text = f"Frame {frame_num:04d}"
#         if self.use_top_k_depths:
#             info_text += f" (Top-{self.top_k_min_depths} Depth)"
#         cv2.putText(color_mask, info_text, (10, 30),
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
#         output_path = self.saved_frames_viz_dir / f"frame_{frame_num:04d}_visualization.png"
#         cv2.imwrite(str(output_path), color_mask)
    
#     # ==================== 保存中心点3D坐标信息 ====================
    
#     def save_all_centers_3d_info(self):
#         """保存所有掩码中心点的3D坐标信息"""
#         print("\n📌 保存所有掩码中心点的3D坐标信息...")
        
#         for frame_num in tqdm(sorted(self.instance_centers_3d.keys()), desc="保存中心点3D坐标"):
#             if frame_num not in self.instance_centers_3d or frame_num not in self.instance_centers_2d:
#                 continue
            
#             centers_3d = self.instance_centers_3d[frame_num]
#             centers_2d = self.instance_centers_2d[frame_num]
#             depths = self.instance_depths.get(frame_num, {})
#             stats = self.instance_stats.get(frame_num, {})
#             all_depths = self.instance_all_depths.get(frame_num, {})
            
#             if not centers_3d:
#                 continue
            
#             filename = self.centers_3d_dir / f"frame_{frame_num:04d}_centers_3d.txt"
            
#             with open(filename, 'w') as f:
#                 f.write("#" + "="*100 + "\n")
#                 f.write(f"# 帧 {frame_num:04d} 的所有掩码中心点3D坐标信息\n")
#                 f.write("#" + "="*100 + "\n\n")
                
#                 f.write("# local_id,global_id,class_id,center_2d_x,center_2d_y,depth_median,point_3d_x,point_3d_y,point_3d_z,depth_method,top_k_used,total_pixels\n")
#                 f.write("-"*100 + "\n")
                
#                 for local_id in sorted(centers_3d.keys()):
#                     if local_id not in centers_2d:
#                         continue
                    
#                     center_x, center_y = centers_2d[local_id]
#                     depth_median = depths.get(local_id, -1)
#                     point_3d = centers_3d[local_id]
                    
#                     key = (frame_num, local_id)
#                     global_id = self.global_instance_map.get(key, local_id if local_id in self.fixed_class_ids.values() else -1)
                    
#                     class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
                    
#                     stat = stats.get(local_id, {})
#                     depth_method = stat.get('depth_calculation_method', 'unknown')
#                     top_k_used = stat.get('top_k_used', 0)
#                     total_pixels = stat.get('num_valid_depths', 0)
                    
#                     f.write(f"{local_id},{global_id},{class_id},{center_x:.2f},{center_y:.2f},{depth_median:.6f},"
#                            f"{point_3d[0]:.6f},{point_3d[1]:.6f},{point_3d[2]:.6f},"
#                            f"{depth_method},{top_k_used},{total_pixels}\n")
        
#         summary_file = self.centers_3d_dir / "all_frames_centers_3d_summary.csv"
#         with open(summary_file, 'w') as f:
#             f.write("frame_num,local_id,global_id,class_id,center_2d_x,center_2d_y,depth_median,point_3d_x,point_3d_y,point_3d_z,depth_method,top_k_used,total_pixels\n")
            
#             for frame_num in sorted(self.instance_centers_3d.keys()):
#                 if frame_num not in self.instance_centers_2d:
#                     continue
                
#                 centers_3d = self.instance_centers_3d[frame_num]
#                 centers_2d = self.instance_centers_2d[frame_num]
#                 depths = self.instance_depths.get(frame_num, {})
#                 stats = self.instance_stats.get(frame_num, {})
                
#                 for local_id in sorted(centers_3d.keys()):
#                     if local_id not in centers_2d:
#                         continue
                    
#                     center_x, center_y = centers_2d[local_id]
#                     depth_median = depths.get(local_id, -1)
#                     point_3d = centers_3d[local_id]
#                     key = (frame_num, local_id)
#                     global_id = self.global_instance_map.get(key, local_id if local_id in self.fixed_class_ids.values() else -1)
#                     class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
#                     stat = stats.get(local_id, {})
#                     depth_method = stat.get('depth_calculation_method', 'unknown')
#                     top_k_used = stat.get('top_k_used', 0)
#                     total_pixels = stat.get('num_valid_depths', 0)
                    
#                     f.write(f"{frame_num},{local_id},{global_id},{class_id},{center_x:.2f},{center_y:.2f},{depth_median:.6f},"
#                            f"{point_3d[0]:.6f},{point_3d[1]:.6f},{point_3d[2]:.6f},"
#                            f"{depth_method},{top_k_used},{total_pixels}\n")
        
#         print(f"✅ 所有帧的中心点3D坐标已保存到: {self.centers_3d_dir}")
    
#     # ==================== 保存匹配详细信息 ====================
    
#     def save_matching_details(self, frame_num, matching_details):
#         """保存当前帧的匹配详细信息"""
#         filename = self.matching_info_dir / f"frame_{frame_num:04d}_matching_info.txt"
        
#         with open(filename, 'w') as f:
#             f.write("="*120 + "\n")
#             f.write(f"帧 {frame_num:04d} 匹配详细信息\n")
#             f.write("="*120 + "\n\n")
            
#             f.write(f"空间阈值: {self.spatial_threshold} 米\n")
#             f.write(f"运动阈值: {self.motion_threshold} 米\n")
#             f.write(f"协同阈值: {self.coherence_threshold} 米\n\n")
            
#             total_instances = len(matching_details)
#             unmatched_count = sum(1 for d in matching_details if d['match_status'] == 'UNMATCHED')
            
#             f.write(f"实例总数: {total_instances}\n")
#             f.write(f"未匹配  : {unmatched_count}\n\n")
            
#             f.write("-"*120 + "\n")
#             f.write("详细匹配信息:\n")
#             f.write("-"*120 + "\n\n")
            
#             for detail in matching_details:
#                 if detail['match_status'] == 'UNMATCHED':
#                     f.write(f"⚠️ 未匹配:\n")
#                     f.write(f"   本地ID: {detail['local_id']}\n")
#                     f.write(f"   原因: {detail['reason']}\n")
#                     f.write(f"   深度方法: {detail['depth_method']}\n")
#                     f.write(f"   使用像素: {detail.get('top_k_used', 0)}/{detail.get('total_pixels', 0)}\n")
#                     f.write("\n")
            
#             f.write("="*120 + "\n")
        
#         print(f"   ✅ 匹配详细信息已保存: {filename}")
    
#     # ==================== 主运行流程 ====================
    
#     def run_matching(self):
#         """运行完整的匹配流程"""
#         print("\n" + "="*60)
#         print("🚀 开始数据驱动协同匹配")
#         print("="*60)
        
#         first_frame = self.initialize_with_first_frame()
#         if first_frame is None:
#             print("❌ 初始化失败")
#             return False
        
#         all_frames = sorted(list(self.instance_centers_3d.keys()))
#         max_frame = max(all_frames)
        
#         # 按照时间顺序重组帧序列
#         # 从第一帧开始，先处理后面的帧直到最后一帧
#         forward_frames = [f for f in all_frames if f > first_frame]
#         # 然后从第一帧开始，处理前面的帧（循环到开头）
#         backward_frames = [f for f in all_frames if f < first_frame]
        
#         # 按照时间顺序重组：初始帧 → 后面的帧 → 前面的帧（循环）
#         ordered_frames = [first_frame] + forward_frames + backward_frames
        
#         print(f"\n📊 处理顺序 (时间循环):")
#         print(f"   第一帧 (初始): {first_frame:04d}")
#         if forward_frames:
#             print(f"   后续帧 (递增): {forward_frames}")
#         if backward_frames:
#             print(f"   循环帧 (从头开始): {backward_frames}")
#         print(f"   完整顺序: {ordered_frames}")
#         print(f"   总处理帧数: {len(all_frames)}/{self.max_frames}")
        
#         processed_frames = {first_frame}
#         self.total_matches = 0
#         self.unmatched_total = 0
        
#         # 按照时间顺序处理所有帧
#         for current_frame in tqdm(ordered_frames[1:], desc="按时间顺序处理帧"):
#             print(f"\n{'='*50}")
#             print(f"📌 处理帧: {current_frame:04d} (时间顺序中的第 {ordered_frames.index(current_frame)+1} 帧)")
#             print(f"{'='*50}")
            
#             matches = self.match_with_coherence(current_frame)
            
#             frame_matches = len(matches)
#             frame_unmatched = len([k for k in self.instance_centers_3d[current_frame].keys() 
#                                 if k <= self.frame_mask_counts.get(current_frame, 0)]) - frame_matches
            
#             if matches:
#                 for local_id, match_info in matches.items():
#                     global_id = match_info['global_id']
#                     distance = match_info.get('distance', 0.3)
#                     point = np.array(self.instance_centers_3d[current_frame][local_id])
                    
#                     key = (current_frame, local_id)
#                     self.global_instance_map[key] = global_id
                    
#                     self.update_global_point(global_id, point, current_frame, local_id, distance)
                    
#                     self.total_matches += 1
#                     if match_info['confidence'] > self.confidence_threshold:
#                         self.matching_stats['high_confidence'] += 1
#                     else:
#                         self.matching_stats['low_confidence'] += 1
                
#                 print(f"\n  ✅ 匹配成功: {frame_matches} 个实例")
#                 if frame_unmatched > 0:
#                     print(f"  ⚠️ 未匹配: {frame_unmatched} 个实例")
#                     self.unmatched_total += frame_unmatched
#             else:
#                 print(f"\n  ⚠️ 没有找到任何匹配")
#                 target_count = len([k for k in self.instance_centers_3d[current_frame].keys() 
#                                 if k <= self.frame_mask_counts.get(current_frame, 0)])
#                 self.unmatched_total += target_count
            
#             processed_frames.add(current_frame)
#             print(f"     已处理 {len(processed_frames)}/{len(all_frames)} 帧")
#             print(f"     当前全局实例数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}")
        
#         self.matching_stats['unmatched_instances'] = self.unmatched_total
#         self.matching_stats['total_matches'] = self.total_matches
        
#         print("\n" + "="*60)
#         print("✅ 匹配完成！")
#         print("="*60)
#         print(f"第一帧: {first_frame:04d} (从候选帧 {self.first_frame_candidates} 中选择)")
#         print(f"处理顺序: {ordered_frames}")
#         print(f"   → 从初始帧向后处理到最后一帧")
#         print(f"   → 然后从头开始处理到初始帧之前的帧")
#         print(f"固定类别ID: 220(类别0), 221(类别1), 222(类别2)")
#         print(f"目标类实例数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}")
#         print(f"总匹配次数: {self.total_matches}")
#         print(f"协同匹配: {self.matching_stats['cooperative_matches']}")
#         print(f"运动匹配: {self.matching_stats['motion_matches']}")
#         print(f"双向匹配: {self.matching_stats['bidirectional_matches']}")
#         print(f"形状匹配: {self.matching_stats['shape_matches']}")
#         print(f"高质量匹配: {self.matching_stats['high_confidence']}")
#         print(f"低质量匹配: {self.matching_stats['low_confidence']}")
#         print(f"未匹配实例总数: {self.unmatched_total}")
        
#         self.save_results()
#         self.save_all_centers_3d_info()
#         self.verify_matching_results()
        
#         return True
    
#     # ==================== 结果保存模块 ====================
    
#     # def save_results(self):
#     #     """保存所有结果"""
#     #     print("\n💾 保存结果...")
        
#     #     mapping_file = self.output_dir / "id_mapping.json"
        
#     #     mapping_data = {
#     #         'first_frame': self.first_frame,
#     #         'first_frame_candidates': self.first_frame_candidates,
#     #         'fixed_class_ids': self.fixed_class_ids,
#     #         'target_class': self.target_class,
#     #         'global_instances': {},
#     #         'frame_mappings': defaultdict(dict),
#     #         'stats': self.matching_stats,
#     #         'color_map': {str(k): v for k, v in self.global_id_colors.items()},
#     #         'depth_config': {
#     #             'use_top_k_depths': self.use_top_k_depths,
#     #             'top_k_min_depths': self.top_k_min_depths
#     #         },
#     #         'processing_info': {
#     #             'max_frames': self.max_frames,
#     #             'processed_frames': len(self.masks)
#     #         }
#     #     }
        
#     #     for (frame, local_id), global_id in self.global_instance_map.items():
#     #         mapping_data['frame_mappings'][str(frame)][str(local_id)] = global_id
        
#     #     for global_id, info in self.global_centers_3d.items():
#     #         class_id = info.get('class_id', 'unknown')
#     #         mapping_data['global_instances'][str(global_id)] = {
#     #             'point_3d': info['point'],
#     #             'frames': info['frames'],
#     #             'instances': [(f, i) for f, i in info['instances']],
#     #             'num_views': info['num_views'],
#     #             'confidence': info.get('confidence', 0.5),
#     #             'avg_distance': np.mean(info['distances']) if info['distances'] else 0,
#     #             'color': self.global_id_colors.get(global_id, [128,128,128]),
#     #             'class_id': class_id,
#     #             'is_fixed': info.get('is_fixed', False)
#     #         }
        
#     #     mapping_data['summary'] = {
#     #         'total_frames': len(self.masks),
#     #         'total_instances': len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0),
#     #         'global_ids': len(self.global_centers_3d),
#     #         'fixed_global_ids': list(self.fixed_class_ids.values()),
#     #         'target_global_ids': [gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()],
#     #         'unmatched_instances': self.matching_stats['unmatched_instances'],
#     #         'rejected_matches': self.matching_stats['rejected_matches'],
#     #         'spatial_threshold': self.spatial_threshold,
#     #         'motion_threshold': self.motion_threshold,
#     #         'coherence_threshold': self.coherence_threshold,
#     #         'min_instance_area': self.min_instance_area,
#     #         'depth_calculation': 'top_k' if self.use_top_k_depths else 'all',
#     #         'top_k_value': self.top_k_min_depths if self.use_top_k_depths else None,
#     #         'max_frames': self.max_frames,
#     #         'processed_frames': len(self.masks)
#     #     }
        
#     #     with open(mapping_file, 'w') as f:
#     #         json.dump(mapping_data, f, indent=2)
        
#     #     print(f"✅ ID映射已保存: {mapping_file}")
        
#     #     # 生成灰度掩码
#     #     unified_mask_dir = self.output_dir / "unified_masks"
#     #     unified_mask_dir.mkdir(exist_ok=True)
        
#     #     print("\n🎨 生成灰度统一ID掩码...")
        
#     #     for frame_num, mask in tqdm(self.masks.items(), desc="生成灰度掩码"):
#     #         new_mask = np.zeros_like(mask)
            
#     #         for local_id in np.unique(mask):
#     #             if local_id == 0:
#     #                 continue
                
#     #             if local_id in self.fixed_class_ids.values():
#     #                 # 固定类别的实例
#     #                 new_mask[mask == local_id] = local_id
#     #             else:
#     #                 key = (frame_num, int(local_id))
#     #                 if key in self.global_instance_map:
#     #                     global_id = self.global_instance_map[key]
#     #                     new_mask[mask == local_id] = global_id
            
#     #         output_path = unified_mask_dir / f"unified_mask_{frame_num:04d}.png"
#     #         cv2.imwrite(str(output_path), new_mask)
        
#     #     print(f"✅ 灰度掩码已保存到: {unified_mask_dir}")
        
#     #     # 生成彩色掩码
#     #     print("\n🎨 生成彩色统一ID掩码...")
        
#     #     for frame_num, mask in tqdm(self.masks.items(), desc="生成彩色掩码"):
#     #         h, w = mask.shape
#     #         color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
#     #         for local_id in np.unique(mask):
#     #             if local_id == 0:
#     #                 continue
                
#     #             if local_id in self.fixed_class_ids.values():
#     #                 # 固定类别的实例
#     #                 color = self.get_instance_color(local_id)
#     #                 color_mask[mask == local_id] = color
#     #             else:
#     #                 key = (frame_num, int(local_id))
#     #                 if key in self.global_instance_map:
#     #                     global_id = self.global_instance_map[key]
#     #                     color = self.get_instance_color(global_id)
#     #                     color_mask[mask == local_id] = color
#     #                 else:
#     #                     color_mask[mask == local_id] = self.unmatched_color
            
#     #         output_path = self.color_masks_dir / f"mask_{frame_num:04d}.png"
#     #         cv2.imwrite(str(output_path), color_mask)
        
#     #     print(f"✅ 彩色掩码已保存到: {self.color_masks_dir}")
        
#     #     # 生成带信息的彩色掩码
#     #     print("\n🎨 生成带ID和深度信息的彩色掩码...")
        
#     #     for frame_num, mask in tqdm(self.masks.items(), desc="生成带信息掩码"):
#     #         h, w = mask.shape
#     #         color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
#     #         for local_id in np.unique(mask):
#     #             if local_id == 0:
#     #                 continue
                
#     #             if local_id in self.fixed_class_ids.values():
#     #                 color = self.get_instance_color(local_id)
#     #                 color_mask[mask == local_id] = color
#     #             else:
#     #                 key = (frame_num, int(local_id))
#     #                 if key in self.global_instance_map:
#     #                     global_id = self.global_instance_map[key]
#     #                     color = self.get_instance_color(global_id)
#     #                     color_mask[mask == local_id] = color
#     #                 else:
#     #                     color_mask[mask == local_id] = self.unmatched_color
            
#     #         for local_id in np.unique(mask):
#     #             if local_id == 0:
#     #                 continue
                
#     #             if frame_num in self.instance_centers_2d and local_id in self.instance_centers_2d[frame_num]:
#     #                 center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                    
#     #                 depth_value = None
#     #                 if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
#     #                     depth_value = self.instance_depths[frame_num][local_id]
                    
#     #                 if local_id in self.fixed_class_ids.values():
#     #                     global_id = local_id
#     #                     class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
#     #                     texts = [f"ID:{global_id}(C{class_id})"]
#     #                 else:
#     #                     key = (frame_num, int(local_id))
#     #                     global_id = self.global_instance_map.get(key)
#     #                     if global_id is None:
#     #                         continue
#     #                     texts = [f"ID:{global_id}"]
                    
#     #                 if depth_value is not None:
#     #                     texts.append(f"{depth_value:.2f}m")
                    
#     #                 if not texts:
#     #                     continue
                    
#     #                 full_text = " ".join(texts)
                    
#     #                 font = cv2.FONT_HERSHEY_SIMPLEX
#     #                 font_scale = self.id_font_scale
#     #                 thickness = 1
                    
#     #                 (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                    
#     #                 text_x = int(center_x - text_width / 2)
#     #                 text_y = int(center_y + text_height / 2)
                    
#     #                 if 0 <= text_x < w and 0 <= text_y < h:
#     #                     padding = 2
#     #                     cv2.rectangle(color_mask, 
#     #                                 (text_x - padding, text_y - text_height - padding),
#     #                                 (text_x + text_width + padding, text_y + padding),
#     #                                 (0, 0, 0), -1)
                        
#     #                     cv2.putText(color_mask, full_text, (text_x, text_y),
#     #                               font, font_scale, self.text_color, thickness)
            
#     #         output_path = self.color_masks_with_info_dir / f"mask_with_info_{frame_num:04d}.png"
#     #         cv2.imwrite(str(output_path), color_mask)
        
#     #     print(f"✅ 带信息彩色掩码已保存到: {self.color_masks_with_info_dir}")
        
#     #     self.save_pixel_depths_to_txt()
        
#     #     legend_file = self.output_dir / "color_legend.png"
#     #     self.save_color_legend(legend_file)
        
#     #     centers_3d_file = self.output_dir / "centers_3d.txt"
#     #     with open(centers_3d_file, 'w') as f:
#     #         f.write("# global_id x y z num_views avg_depth confidence first_frame class_id is_fixed color_rgb depth_method top_k_used\n")
#     #         for global_id, info in self.global_centers_3d.items():
#     #             point = info['point']
#     #             stats = self.global_stats.get(global_id, {})
#     #             avg_depth = stats.get('avg_depth', 0)
#     #             confidence = info.get('confidence', 0.5)
#     #             first_frame = info['frames'][0] if info['frames'] else 0
#     #             color = self.global_id_colors.get(global_id, (128,128,128))
#     #             class_id = info.get('class_id', 'unknown')
#     #             is_fixed = info.get('is_fixed', False)
                
#     #             if not is_fixed and info['frames']:
#     #                 first_inst = info['instances'][0]
#     #                 frame_num, local_id = first_inst
#     #                 depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown')
#     #                 top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0)
#     #             else:
#     #                 depth_method = 'fixed'
#     #                 top_k_used = 0
                
#     #             f.write(f"{global_id} {point[0]:.3f} {point[1]:.3f} {point[2]:.3f} "
#     #                    f"{info['num_views']} {avg_depth:.3f} {confidence:.2f} {first_frame:04d} "
#     #                    f"{class_id} {is_fixed} ({color[0]},{color[1]},{color[2]}) "
#     #                    f"{depth_method} {top_k_used}\n")
        
#     #     print(f"✅ 3D中心点已保存: {centers_3d_file}")
        
#     #     report_file = self.output_dir / "matching_report.txt"
#     #     with open(report_file, 'w') as f:
#     #         f.write("="*60 + "\n")
#     #         f.write("数据驱动3D匹配报告\n")
#     #         f.write("="*60 + "\n\n")
            
#     #         f.write(f"配置参数:\n")
#     #         f.write(f"  目标类别: {self.target_class}\n")
#     #         f.write(f"  固定类别ID: 220(类别0), 221(类别1), 222(类别2)\n")
#     #         f.write(f"  候选初始帧: {self.first_frame_candidates}\n")
#     #         f.write(f"  选中的初始帧: {self.first_frame}\n")
#     #         f.write(f"  空间阈值: {self.spatial_threshold}米\n")
#     #         f.write(f"  运动阈值: {self.motion_threshold}米\n")
#     #         f.write(f"  协同阈值: {self.coherence_threshold}米\n")
#     #         f.write(f"  运动权重: {self.motion_weight}\n")
#     #         f.write(f"  双向权重: {self.bidirectional_weight}\n")
#     #         f.write(f"  形状权重: {self.shape_weight}\n")
#     #         f.write(f"  置信度阈值: {self.confidence_threshold}\n")
#     #         f.write(f"  最小实例面积: {self.min_instance_area}像素\n")
#     #         f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
#     #         if self.use_top_k_depths:
#     #             f.write(f"  K值: {self.top_k_min_depths}\n")
#     #         f.write(f"  处理帧数限制: 前 {self.max_frames} 帧\n\n")
            
#     #         f.write(f"数据统计:\n")
#     #         f.write(f"  总帧数: {len(self.masks)}\n")
#     #         f.write(f"  第一帧: {self.first_frame:04d}\n")
#     #         f.write(f"  固定类全局ID: 220, 221, 222\n")
#     #         f.write(f"  目标类全局ID数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}\n")
#     #         f.write(f"  总实例数: {len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0)}\n")
#     #         f.write(f"  未匹配实例数: {self.matching_stats['unmatched_instances']}\n\n")
            
#     #         f.write(f"匹配统计:\n")
#     #         f.write(f"  总匹配次数: {self.matching_stats['total_matches']}\n")
#     #         f.write(f"  协同匹配: {self.matching_stats['cooperative_matches']}\n")
#     #         f.write(f"  运动匹配: {self.matching_stats['motion_matches']}\n")
#     #         f.write(f"  双向匹配: {self.matching_stats['bidirectional_matches']}\n")
#     #         f.write(f"  形状匹配: {self.matching_stats['shape_matches']}\n")
#     #         f.write(f"  高质量匹配: {self.matching_stats['high_confidence']}\n")
#     #         f.write(f"  低质量匹配: {self.matching_stats['low_confidence']}\n")
        
#     #     print(f"✅ 报告已保存: {report_file}")


#     def save_results(self):
#         """保存所有结果"""
#         print("\n💾 保存结果...")
        
#         mapping_file = self.output_dir / "id_mapping.json"
        
#         mapping_data = {
#             'first_frame': self.first_frame,
#             'first_frame_candidates': self.first_frame_candidates,
#             'fixed_class_ids': self.fixed_class_ids,
#             'target_class': self.target_class,
#             'global_instances': {},
#             'frame_mappings': defaultdict(dict),
#             'stats': self.matching_stats,
#             'color_map': {str(k): v for k, v in self.global_id_colors.items()},
#             'depth_config': {
#                 'use_top_k_depths': self.use_top_k_depths,
#                 'top_k_min_depths': self.top_k_min_depths
#             },
#             'processing_info': {
#                 'max_frames': self.max_frames,
#                 'processed_frames': len(self.masks)
#             }
#         }
        
#         for (frame, local_id), global_id in self.global_instance_map.items():
#             mapping_data['frame_mappings'][str(frame)][str(local_id)] = global_id
        
#         for global_id, info in self.global_centers_3d.items():
#             class_id = info.get('class_id', 'unknown')
#             mapping_data['global_instances'][str(global_id)] = {
#                 'point_3d': info['point'],
#                 'frames': info['frames'],
#                 'instances': [(f, i) for f, i in info['instances']],
#                 'num_views': info['num_views'],
#                 'confidence': info.get('confidence', 0.5),
#                 'avg_distance': np.mean(info['distances']) if info['distances'] else 0,
#                 'color': self.global_id_colors.get(global_id, [128,128,128]),
#                 'class_id': class_id,
#                 'is_fixed': info.get('is_fixed', False)
#             }
        
#         mapping_data['summary'] = {
#             'total_frames': len(self.masks),
#             'total_instances': len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0),
#             'global_ids': len(self.global_centers_3d),
#             'fixed_global_ids': list(self.fixed_class_ids.values()),
#             'target_global_ids': [gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()],
#             'unmatched_instances': self.matching_stats['unmatched_instances'],
#             'rejected_matches': self.matching_stats['rejected_matches'],
#             'spatial_threshold': self.spatial_threshold,
#             'motion_threshold': self.motion_threshold,
#             'coherence_threshold': self.coherence_threshold,
#             'min_instance_area': self.min_instance_area,
#             'depth_calculation': 'top_k' if self.use_top_k_depths else 'all',
#             'top_k_value': self.top_k_min_depths if self.use_top_k_depths else None,
#             'max_frames': self.max_frames,
#             'processed_frames': len(self.masks)
#         }
        
#         with open(mapping_file, 'w') as f:
#             json.dump(mapping_data, f, indent=2)
        
#         print(f"✅ ID映射已保存: {mapping_file}")
        
#         # 生成灰度掩码
#         unified_mask_dir = self.output_dir / "unified_masks"
#         unified_mask_dir.mkdir(exist_ok=True)
        
#         print("\n🎨 生成灰度统一ID掩码...")
        
#         for frame_num, mask in tqdm(self.masks.items(), desc="生成灰度掩码"):
#             new_mask = np.zeros_like(mask)
            
#             for local_id in np.unique(mask):
#                 if local_id == 0:
#                     continue
                
#                 if local_id in self.fixed_class_ids.values():
#                     # 固定类别的实例
#                     new_mask[mask == local_id] = local_id
#                 else:
#                     key = (frame_num, int(local_id))
#                     if key in self.global_instance_map:
#                         global_id = self.global_instance_map[key]
#                         new_mask[mask == local_id] = global_id
            
#             output_path = unified_mask_dir / f"unified_mask_{frame_num:04d}.png"
#             cv2.imwrite(str(output_path), new_mask)
        
#         print(f"✅ 灰度掩码已保存到: {unified_mask_dir}")
        
#         # 生成彩色掩码
#         print("\n🎨 生成彩色统一ID掩码...")
        
#         for frame_num, mask in tqdm(self.masks.items(), desc="生成彩色掩码"):
#             h, w = mask.shape
#             color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
#             for local_id in np.unique(mask):
#                 if local_id == 0:
#                     continue
                
#                 if local_id in self.fixed_class_ids.values():
#                     # 固定类别的实例
#                     color = self.get_instance_color(local_id)
#                     color_mask[mask == local_id] = color
#                 else:
#                     key = (frame_num, int(local_id))
#                     if key in self.global_instance_map:
#                         global_id = self.global_instance_map[key]
#                         color = self.get_instance_color(global_id)
#                         color_mask[mask == local_id] = color
#                     else:
#                         color_mask[mask == local_id] = self.unmatched_color
            
#             output_path = self.color_masks_dir / f"mask_{frame_num:04d}.png"
#             cv2.imwrite(str(output_path), color_mask)
        
#         print(f"✅ 彩色掩码已保存到: {self.color_masks_dir}")
        
#         # ========== 修改开始：生成带信息的彩色掩码（固定类别也显示ID信息） ==========
#         print("\n🎨 生成带ID和深度信息的彩色掩码（所有类别都显示ID）...")
        
#         for frame_num, mask in tqdm(self.masks.items(), desc="生成带信息掩码"):
#             h, w = mask.shape
#             color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
#             # 第一步：为所有实例上色
#             for local_id in np.unique(mask):
#                 if local_id == 0:
#                     continue
                
#                 if local_id in self.fixed_class_ids.values():
#                     # 固定类别的实例 - 上色
#                     color = self.get_instance_color(local_id)
#                     color_mask[mask == local_id] = color
#                 else:
#                     key = (frame_num, int(local_id))
#                     if key in self.global_instance_map:
#                         global_id = self.global_instance_map[key]
#                         color = self.get_instance_color(global_id)
#                         color_mask[mask == local_id] = color
#                     else:
#                         color_mask[mask == local_id] = self.unmatched_color
            
#             # 第二步：为所有实例添加文字信息（包括固定类别）
#             for local_id in np.unique(mask):
#                 if local_id == 0:
#                     continue
                
#                 # 获取中心点坐标（固定类别需要临时计算）
#                 if local_id in self.fixed_class_ids.values():
#                     # 固定类别 - 临时计算中心点
#                     y_indices, x_indices = np.where(mask == local_id)
#                     if len(y_indices) == 0:
#                         continue
#                     center_x = int(np.mean(x_indices))
#                     center_y = int(np.mean(y_indices))
                    
#                     # 固定类别没有深度信息
#                     depth_value = None
                    
#                     # 固定类别的文字
#                     class_map = {220: '0', 221: '1', 222: '2'}
#                     texts = [f"ID:{local_id}(C{class_map[local_id]})"]
#                     font_scale = self.id_font_scale
                    
#                 else:
#                     # 目标类别 - 使用已有的中心点
#                     if frame_num not in self.instance_centers_2d or local_id not in self.instance_centers_2d[frame_num]:
#                         continue
                        
#                     center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                    
#                     depth_value = None
#                     if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
#                         depth_value = self.instance_depths[frame_num][local_id]
                    
#                     key = (frame_num, int(local_id))
#                     global_id = self.global_instance_map.get(key)
#                     if global_id is None:
#                         continue
                    
#                     texts = [f"ID:{global_id}"]
#                     font_scale = self.id_font_scale
                
#                 # 添加深度信息（如果有）
#                 if depth_value is not None:
#                     texts.append(f"{depth_value:.2f}m")
                
#                 if not texts:
#                     continue
                
#                 full_text = " ".join(texts)
                
#                 # 绘制文字
#                 font = cv2.FONT_HERSHEY_SIMPLEX
#                 thickness = 1
                
#                 (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                
#                 text_x = int(center_x - text_width / 2)
#                 text_y = int(center_y + text_height / 2)
                
#                 if 0 <= text_x < w and 0 <= text_y < h:
#                     padding = 2
#                     cv2.rectangle(color_mask, 
#                                 (text_x - padding, text_y - text_height - padding),
#                                 (text_x + text_width + padding, text_y + padding),
#                                 (0, 0, 0), -1)
                    
#                     cv2.putText(color_mask, full_text, (text_x, text_y),
#                             font, font_scale, self.text_color, thickness)
            
#             output_path = self.color_masks_with_info_dir / f"mask_with_info_{frame_num:04d}.png"
#             cv2.imwrite(str(output_path), color_mask)
        
#         print(f"✅ 带信息彩色掩码已保存到: {self.color_masks_with_info_dir}（所有类别都显示ID）")
#         # ========== 修改结束 ==========
        
#         self.save_pixel_depths_to_txt()
        
#         legend_file = self.output_dir / "color_legend.png"
#         self.save_color_legend(legend_file)
        
#         centers_3d_file = self.output_dir / "centers_3d.txt"
#         with open(centers_3d_file, 'w') as f:
#             f.write("# global_id x y z num_views avg_depth confidence first_frame class_id is_fixed color_rgb depth_method top_k_used\n")
#             for global_id, info in self.global_centers_3d.items():
#                 point = info['point']
#                 stats = self.global_stats.get(global_id, {})
#                 avg_depth = stats.get('avg_depth', 0)
#                 confidence = info.get('confidence', 0.5)
#                 first_frame = info['frames'][0] if info['frames'] else 0
#                 color = self.global_id_colors.get(global_id, (128,128,128))
#                 class_id = info.get('class_id', 'unknown')
#                 is_fixed = info.get('is_fixed', False)
                
#                 if not is_fixed and info['frames']:
#                     first_inst = info['instances'][0]
#                     frame_num, local_id = first_inst
#                     depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown')
#                     top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0)
#                 else:
#                     depth_method = 'fixed'
#                     top_k_used = 0
                
#                 f.write(f"{global_id} {point[0]:.3f} {point[1]:.3f} {point[2]:.3f} "
#                     f"{info['num_views']} {avg_depth:.3f} {confidence:.2f} {first_frame:04d} "
#                     f"{class_id} {is_fixed} ({color[0]},{color[1]},{color[2]}) "
#                     f"{depth_method} {top_k_used}\n")
        
#         print(f"✅ 3D中心点已保存: {centers_3d_file}")
        
#         report_file = self.output_dir / "matching_report.txt"
#         with open(report_file, 'w') as f:
#             f.write("="*60 + "\n")
#             f.write("数据驱动3D匹配报告\n")
#             f.write("="*60 + "\n\n")
            
#             f.write(f"配置参数:\n")
#             f.write(f"  目标类别: {self.target_class}\n")
#             f.write(f"  固定类别ID: 220(类别0), 221(类别1), 222(类别2)\n")
#             f.write(f"  候选初始帧: {self.first_frame_candidates}\n")
#             f.write(f"  选中的初始帧: {self.first_frame}\n")
#             f.write(f"  空间阈值: {self.spatial_threshold}米\n")
#             f.write(f"  运动阈值: {self.motion_threshold}米\n")
#             f.write(f"  协同阈值: {self.coherence_threshold}米\n")
#             f.write(f"  运动权重: {self.motion_weight}\n")
#             f.write(f"  双向权重: {self.bidirectional_weight}\n")
#             f.write(f"  形状权重: {self.shape_weight}\n")
#             f.write(f"  置信度阈值: {self.confidence_threshold}\n")
#             f.write(f"  最小实例面积: {self.min_instance_area}像素\n")
#             f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
#             if self.use_top_k_depths:
#                 f.write(f"  K值: {self.top_k_min_depths}\n")
#             f.write(f"  处理帧数限制: 前 {self.max_frames} 帧\n\n")
            
#             f.write(f"数据统计:\n")
#             f.write(f"  总帧数: {len(self.masks)}\n")
#             f.write(f"  第一帧: {self.first_frame:04d}\n")
#             f.write(f"  固定类全局ID: 220, 221, 222\n")
#             f.write(f"  目标类全局ID数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}\n")
#             f.write(f"  总实例数: {len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0)}\n")
#             f.write(f"  未匹配实例数: {self.matching_stats['unmatched_instances']}\n\n")
            
#             f.write(f"匹配统计:\n")
#             f.write(f"  总匹配次数: {self.matching_stats['total_matches']}\n")
#             f.write(f"  协同匹配: {self.matching_stats['cooperative_matches']}\n")
#             f.write(f"  运动匹配: {self.matching_stats['motion_matches']}\n")
#             f.write(f"  双向匹配: {self.matching_stats['bidirectional_matches']}\n")
#             f.write(f"  形状匹配: {self.matching_stats['shape_matches']}\n")
#             f.write(f"  高质量匹配: {self.matching_stats['high_confidence']}\n")
#             f.write(f"  低质量匹配: {self.matching_stats['low_confidence']}\n")
        
#         print(f"✅ 报告已保存: {report_file}")




    
#     def save_color_legend(self, output_path):
#         """保存颜色图例"""
#         if not self.global_centers_3d:
#             return
        
#         num_ids = len(self.global_centers_3d)
#         legend_height = max(30 * num_ids, 100)
#         legend_width = 600
        
#         legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
#         # 先显示固定类别
#         fixed_ids = sorted([gid for gid in self.global_centers_3d.keys() if gid in self.fixed_class_ids.values()])
#         dynamic_ids = sorted([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])
        
#         i = 0
#         for global_id in fixed_ids + dynamic_ids:
#             y_start = i * 30 + 10
#             y_end = y_start + 20
            
#             color = self.global_id_colors.get(global_id, (128,128,128))
            
#             cv2.rectangle(legend, (20, y_start), (60, y_end), color, -1)
#             cv2.rectangle(legend, (20, y_start), (60, y_end), (0,0,0), 1)
            
#             info = self.global_centers_3d[global_id]
#             class_id = info.get('class_id', '?')
#             is_fixed = info.get('is_fixed', False)
            
#             if is_fixed:
#                 text = f"ID {global_id:3d} (C{class_id}): FIXED CLASS"
#             else:
#                 first_inst = info['instances'][0] if info['instances'] else (0,0)
#                 frame_num, local_id = first_inst
#                 depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown') if frame_num in self.instance_stats else 'unknown'
#                 top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0) if frame_num in self.instance_stats else 0
#                 total_pixels = self.instance_stats[frame_num].get(local_id, {}).get('num_valid_depths', 0) if frame_num in self.instance_stats else 0
                
#                 method_text = f" ({depth_method}:{top_k_used}/{total_pixels})" if self.use_top_k_depths else ""
#                 text = f"ID {global_id:3d} (C{class_id}): {info['num_views']}帧, 置信度:{info.get('confidence',0.5):.2f}{method_text}"
            
#             cv2.putText(legend, text, (70, y_end-2), 
#                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
#             i += 1
        
#         title = f"初始帧: {self.first_frame:04d} | 固定ID: 220(C0), 221(C1), 222(C2) | 目标类: {self.target_class} | 深度计算方法: {'前K个最小深度 (K=' + str(self.top_k_min_depths) + ')' if self.use_top_k_depths else '全部深度'}"
#         cv2.putText(legend, title, (20, legend_height-10), 
#                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
#         cv2.imwrite(str(output_path), legend)
#         print(f"✅ 颜色图例已保存: {output_path}")
    
#     # ==================== 验证模块 ====================
    
#     def verify_matching_results(self):
#         """验证匹配结果"""
#         print("\n" + "="*60)
#         print("🔍 验证匹配结果")
#         print("="*60)
        
#         all_frames = sorted(self.frame_matched_counts.keys())
        
#         print("\n每帧匹配情况:")
#         print("-"*100)
#         print(f"{'帧号':<8} {'原始掩码':<10} {'有深度':<10} {'目标类':<10} {'匹配数':<10} {'匹配率(%)':<10} {'未匹配':<10}")
#         print("-"*100)
        
#         total_original = 0
#         total_with_depth = 0
#         total_target = 0
#         total_matched = 0
        
#         for frame_num in all_frames:
#             original = self.frame_original_counts.get(frame_num, 0)
#             with_depth = self.frame_instances_with_depth.get(frame_num, 0)
#             target_count = self.frame_mask_counts.get(frame_num, 0)
#             matched = self.frame_matched_counts.get(frame_num, 0)
#             unmatched = with_depth - matched
#             match_rate = (matched / target_count * 100) if target_count > 0 else 0
            
#             total_original += original
#             total_with_depth += with_depth
#             total_target += target_count
#             total_matched += matched
            
#             marker = " ⚠️" if match_rate < 50 and target_count > 0 else ""
            
#             print(f"{frame_num:04d}    {original:<10} {with_depth:<10} {target_count:<10} {matched:<10} {match_rate:<10.1f}{marker} {unmatched:<10}")
        
#         print("-"*100)
#         print(f"{'总计':<8} {total_original:<10} {total_with_depth:<10} {total_target:<10} {total_matched:<10} {(total_matched/total_target*100):<10.1f} {total_with_depth-total_matched:<10}")
    
#     # ==================== 可视化模块 ====================
    
#     def visualize_results(self):
#         """可视化匹配结果"""
#         print("\n📊 生成可视化...")
        
#         if not self.global_centers_3d:
#             print("没有全局点可可视化")
#             return
        
#         fig = plt.figure(figsize=(15, 12))
        
#         ax1 = fig.add_subplot(221, projection='3d')
        
#         points_3d = []
#         point_colors = []
#         point_labels = []
        
#         for global_id, info in self.global_centers_3d.items():
#             point = info['point']
#             points_3d.append(point)
#             color_rgb = np.array(self.global_id_colors.get(global_id, (128,128,128))) / 255.0
#             point_colors.append(color_rgb)
#             point_labels.append(f"ID:{global_id}")
        
#         points_3d = np.array(points_3d)
        
#         if len(points_3d) > 0:
#             scatter = ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
#                                  c=point_colors, s=50)
            
#             # 标注固定类别的点
#             for i, (global_id, label) in enumerate(zip(self.global_centers_3d.keys(), point_labels)):
#                 if global_id in self.fixed_class_ids.values():
#                     ax1.text(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 
#                             f" {label}", fontsize=8, color='red')
            
#             if self.first_frame in self.instance_centers_3d:
#                 for local_id, point in self.instance_centers_3d[self.first_frame].items():
#                     if local_id <= self.frame_mask_counts.get(self.first_frame, 0):
#                         key = (self.first_frame, local_id)
#                         if key in self.global_instance_map:
#                             global_id = self.global_instance_map[key]
#                             ax1.scatter(point[0], point[1], point[2], 
#                                       c='red', s=100, marker='*')
        
#         ax1.set_xlabel('X (m)')
#         ax1.set_ylabel('Y (m)')
#         ax1.set_zlabel('Z (m)')
#         ax1.set_title(f'3D中心点分布 (红*为第一帧 {self.first_frame:04d})')
        
#         ax2 = fig.add_subplot(222)
#         confidences = [info.get('confidence', 0.5) for gid, info in self.global_centers_3d.items() 
#                       if gid not in self.fixed_class_ids.values()]
#         if confidences:
#             ax2.hist(confidences, bins=20, alpha=0.7, color='green')
#             ax2.axvline(self.confidence_threshold, color='r', linestyle='--', 
#                        label=f'Threshold: {self.confidence_threshold}')
#         ax2.set_xlabel('Confidence')
#         ax2.set_ylabel('Frequency')
#         ax2.set_title('目标类实例置信度分布')
#         ax2.legend()
        
#         ax3 = fig.add_subplot(223)
#         views_counts = [info['num_views'] for gid, info in self.global_centers_3d.items() 
#                        if gid not in self.fixed_class_ids.values()]
#         if views_counts:
#             ax3.hist(views_counts, bins=20, alpha=0.7, color='blue')
#         ax3.set_xlabel('Number of views')
#         ax3.set_ylabel('Frequency')
#         ax3.set_title('目标类视角数分布')
        
#         ax4 = fig.add_subplot(224)
#         stats_names = ['目标类实例', '固定类(3个)', '未匹配', '协同匹配', '运动匹配', '双向匹配', '形状匹配']
#         stats_values = [
#             len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()]),
#             3,
#             self.matching_stats['unmatched_instances'],
#             self.matching_stats['cooperative_matches'],
#             self.matching_stats['motion_matches'],
#             self.matching_stats['bidirectional_matches'],
#             self.matching_stats['shape_matches']
#         ]
#         colors = ['green', 'gray', 'red', 'purple', 'orange', 'cyan', 'yellow']
#         ax4.bar(stats_names, stats_values, color=colors)
#         ax4.set_ylabel('Count')
#         ax4.set_title('匹配统计')
#         ax4.tick_params(axis='x', rotation=45)
        
#         plt.tight_layout()
        
#         viz_path = self.output_dir / "visualization.png"
#         plt.savefig(viz_path, dpi=150, bbox_inches='tight')
#         plt.show()
        
#         print(f"✅ 可视化已保存: {viz_path}")


# # ==================== 主函数 ====================

# def main():
#     """主函数"""
#     config = {
#         # 深度图目录
#         'depth_dir': "/datashare/dir_liusha/xibeinonglin/第一批/B73-1_frames/output_B73-1/train1/ours_30000/depth",
        
#         # 相机参数文件
#         'camera_json': "/datashare/dir_liusha/xibeinonglin/第一批/B73-1_frames/output_B73-1/cameras.json",
        
#         # 分割掩码目录 (包含mask_xxx.png和class_info_xxx.json)
#         'mask_dir': "/datashare/dir_liusha/xibeinonglin/第一批/B73-1_frames/masks_results/integer_masks",
        
#         # 输出目录
#         'output_dir': "/datashare/dir_liusha/corn_yolov12_1215_2/yolo自带匹配/数据驱动匹配_B73-1_fixed_classes",

#         # 目标类别和固定ID
#         'target_class': 3,  # 目标类别，进行匹配
#         'fixed_class_ids': {
#             0: 220,  # 类别0固定为220
#             1: 221,  # 类别1固定为221
#             2: 222   # 类别2固定为222
#         },

#         # 候选初始帧（从0001.jpg到0005.jpg）
#         'first_frame_candidates': [1, 2, 3, 4, 5],

#         # 参数配置
#         'depth_scale': 1000.0,
#         'depth_format': '16bit',
#         'max_depth': 10.0,
#         'spatial_threshold': 0.3,        # 空间距离阈值
#         'motion_threshold': 0.5,          # 运动预测阈值
#         'motion_weight': 0.4,              # 运动连续性权重
#         'bidirectional_weight': 0.4,       # 双向匹配权重
#         'shape_weight': 0.2,               # 形状相似性权重
#         'coherence_threshold': 0.2,        # 协同阈值
#         'confidence_threshold': 0.5,
#         'min_instance_area': 5,
        
#         # 新增：只处理前10帧
#         'max_frames': 9,
        
#         # 运动相关配置
#         'use_motion_prior': True,
        
#         # 前K个最小深度的配置
#         'use_top_k_depths': True,
#         'top_k_min_depths': 2000,
        
#         # 颜色映射方式
#         'colormap': 'tab20',
        
#         # 像素深度保存配置
#         'pixel_depth_save': {
#             'enabled': True,
#             'frames_to_save': [3, 4],
#             'depth_decimals': 3,
#             'min_depth': 0.0,
#             'max_depth': 10.0,
#             'include_coordinates': True,
#             'separate_files': True,
#             'save_visualization': True,
#         },
        
#         # 文字绘制参数
#         'id_font_scale': 0.4,
#         'depth_font_scale': 0.3,
#         'text_color': (255, 255, 255),
#         'unmatched_color': (128, 128, 128),
        
#         # 手动指定第一帧（会被候选帧覆盖，除非手动设置）
#         'first_frame': None
#     }
    
#     # 创建匹配器并运行
#     matcher = CenterPoint3DMatcher(config)
    
#     # 运行匹配
#     success = matcher.run_matching()
    
#     if success:
#         # 可视化结果
#         matcher.visualize_results()
        
#         print("\n" + "="*60)
#         print("✅ 所有处理完成！")
#         print("="*60)
#         print(f"结果保存在: {config['output_dir']}")
#         print(f"📊 初始帧: {matcher.first_frame:04d} (从候选帧 {config['first_frame_candidates']} 中选择)")
#         print(f"📊 固定类别ID: 220(类别0), 221(类别1), 222(类别2)")
#         print(f"📊 目标类别: {config['target_class']}")
#         print(f"📊 深度计算方法: {'前' + str(config['top_k_min_depths']) + '个最小深度' if config['use_top_k_depths'] else '全部深度'}")
#         print(f"📊 处理帧数: 前{config['max_frames']}帧")
#     else:
#         print("\n❌ 处理失败")


# if __name__ == "__main__":
#     main()













# 处理多个文件夹
# 单个文件跑
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from datetime import datetime
from scipy.ndimage import binary_erosion, binary_dilation

class CenterPoint3DMatcher:
    """
    以第一帧为初始帧的3D空间实例匹配器
    核心功能：数据驱动的多证据融合匹配
    - 运动连续性预测
    - 双向空间匹配
    - 掩码形状相似性
    """
    
    def __init__(self, config):
        """
        config: {
            'depth_dir': 深度图目录,
            'camera_json': 相机参数文件,
            'mask_dir': 分割掩码目录,
            'output_dir': 输出目录,
            'target_class': 目标类别 (默认3),
            'fixed_class_ids': 固定类别ID映射 {class_id: global_id},
            'depth_scale': 深度缩放因子 (默认1000),
            'depth_format': 深度图格式 ('16bit' 或 'npy'),
            'max_depth': 最大有效深度 (默认10.0米),
            'spatial_threshold': 空间距离阈值 (默认0.3米),
            'motion_threshold': 运动预测阈值 (默认0.5米),
            'motion_weight': 运动连续性权重 (默认0.4),
            'bidirectional_weight': 双向匹配权重 (默认0.4),
            'shape_weight': 形状相似性权重 (默认0.2),
            'coherence_threshold': 协同阈值 (默认0.2),
            'confidence_threshold': 置信度阈值 (默认0.5),
            'first_frame': 指定第一帧 (可选),
            'colormap': 颜色映射方式 ('random', 'tab20', 'hsv') 默认'tab20',
            
            # 运动相关配置
            'use_motion_prior': True,         # 是否使用运动先验
            
            # 前K个最小深度的配置
            'top_k_min_depths': 5000,
            'use_top_k_depths': True,
            
            # 像素深度保存配置
            'pixel_depth_save': {...},
            
            'id_font_scale': 0.4,
            'depth_font_scale': 0.3,
            'text_color': (255, 255, 255),
            'unmatched_color': (128, 128, 128),
            'min_instance_area': 5,
            
            # 新增：只处理前N帧
            'max_frames': 10,  # 只处理前10帧
            'first_frame_candidates': [1,2,3,4,5],  # 候选初始帧
        }
        """
        self.config = config
        self.depth_dir = Path(config['depth_dir'])
        self.camera_json = config['camera_json']
        self.mask_dir = Path(config['mask_dir'])
        self.output_dir = Path(config['output_dir'])
        
        # 新增：目标类别和固定类别ID
        self.target_class = config.get('target_class', 3)
        self.fixed_class_ids = config.get('fixed_class_ids', {
            0: 220,  # 类别0固定为220
            1: 221,  # 类别1固定为221
            2: 222   # 类别2固定为222
        })
        
        # 新增：候选初始帧
        self.first_frame_candidates = config.get('first_frame_candidates', [1, 2, 3, 4, 5])
        
        # 参数配置
        self.depth_scale = config.get('depth_scale', 1000.0)
        self.depth_format = config.get('depth_format', '16bit')
        self.max_depth = config.get('max_depth', 10.0)
        self.spatial_threshold = config.get('spatial_threshold', 0.3)
        self.motion_threshold = config.get('motion_threshold', 0.5)
        self.motion_weight = config.get('motion_weight', 0.4)        # 运动连续性权重
        self.bidirectional_weight = config.get('bidirectional_weight', 0.4)  # 双向匹配权重
        self.shape_weight = config.get('shape_weight', 0.2)          # 形状相似性权重
        self.coherence_threshold = config.get('coherence_threshold', 0.2)
        self.confidence_threshold = config.get('confidence_threshold', 0.5)
        self.colormap = config.get('colormap', 'tab20')
        self.min_instance_area = config.get('min_instance_area', 5)
        
        # 新增：只处理前N帧
        self.max_frames = config.get('max_frames', 10)
        
        # 运动相关配置
        self.use_motion_prior = config.get('use_motion_prior', True)
        
        # ========== 前K个最小深度的配置 ==========
        self.top_k_min_depths = config.get('top_k_min_depths', 2000)
        self.use_top_k_depths = config.get('use_top_k_depths', True)
        
        # 像素深度保存配置
        self.pixel_depth_config = config.get('pixel_depth_save', {})
        self.pixel_depth_enabled = self.pixel_depth_config.get('enabled', False)
        self.frames_to_save = self.pixel_depth_config.get('frames_to_save', [])
        self.depth_decimals = self.pixel_depth_config.get('depth_decimals', 3)
        self.pixel_min_depth = self.pixel_depth_config.get('min_depth', 0.0)
        self.pixel_max_depth = self.pixel_depth_config.get('max_depth', 10.0)
        self.include_coordinates = self.pixel_depth_config.get('include_coordinates', True)
        self.separate_files = self.pixel_depth_config.get('separate_files', True)
        self.save_visualization = self.pixel_depth_config.get('save_visualization', True)
        
        # 文字绘制参数
        self.id_font_scale = config.get('id_font_scale', 0.4)
        self.depth_font_scale = config.get('depth_font_scale', 0.3)
        self.text_color = config.get('text_color', (255, 255, 255))
        self.unmatched_color = config.get('unmatched_color', (128, 128, 128))
        
        # 可以手动指定第一帧，但会被覆盖如果设置了自动选择
        self.first_frame = config.get('first_frame', None)
        
        # 数据结构
        self.cameras = {}           # 相机参数 {frame_num: {...}}
        self.depth_maps = {}         # 深度图 {frame_num: array}
        self.depth_maps_meters = {}  # 深度图（米为单位）{frame_num: array}
        self.masks = {}              # 分割掩码 {frame_num: array} - 只包含目标类别的实例
        self.mask_contours = {}      # 掩码轮廓 {frame_num: {instance_id: contour}}
        self.mask_moments = {}       # 掩码矩 {frame_num: {instance_id: moments}}
        
        # 类别信息
        self.frame_class_info = defaultdict(dict)  # {frame_num: {local_id: class_id}}
        
        # 实例中心点信息
        self.instance_centers_2d = defaultdict(dict)  # {frame_num: {instance_id: (x, y)}}
        self.instance_depths = defaultdict(dict)      # {frame_num: {instance_id: depth_median}}
        self.instance_stats = defaultdict(dict)       # {frame_num: {instance_id: stats}}
        
        # 保存每个实例的所有有效深度值
        self.instance_all_depths = defaultdict(dict)  # {frame_num: {instance_id: depths_list}}
        
        # 3D中心点
        self.instance_centers_3d = defaultdict(dict)  # {frame_num: {instance_id: [x, y, z]}}
        
        # 全局ID管理
        self.global_instance_counter = 223  # 从223开始，因为220-222已被固定类别占用
        self.global_instance_map = {}  # (frame_num, local_id) -> global_id
        self.global_centers_3d = {}    # global_id -> 3D中心点
        self.global_stats = {}          # global_id -> 统计信息
        self.global_masks = {}           # global_id -> 历史掩码 {frame_num: mask}
        
        # ========== 运动跟踪相关 ==========
        self.motion_trajectories = defaultdict(list)  # global_id -> [(frame_num, position), ...]
        
        # 颜色映射
        self.global_id_colors = {}      # global_id -> (B, G, R)
        
        # 匹配结果统计
        self.matching_stats = {
            'total_matches': 0,
            'cooperative_matches': 0,    # 协同匹配数
            'motion_matches': 0,          # 纯运动匹配数
            'bidirectional_matches': 0,   # 纯双向匹配数
            'shape_matches': 0,           # 纯形状匹配数
            'high_confidence': 0,
            'low_confidence': 0,
            'unmatched_instances': 0,
            'rejected_matches': 0
        }
        
        # 帧掩码数量统计
        self.frame_mask_counts = {}  # frame_num -> mask_count (目标类实例数)
        self.frame_original_counts = {}  # 原始实例总数
        self.frame_matched_counts = {}   # 匹配成功的实例数
        self.frame_instances_with_depth = {}  # 有有效深度的实例数
        
        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 彩色掩码目录
        self.color_masks_dir = self.output_dir / "color_masks"
        self.color_masks_dir.mkdir(exist_ok=True)
        
        # 带信息的彩色掩码目录
        self.color_masks_with_info_dir = self.output_dir / "color_masks_with_info"
        self.color_masks_with_info_dir.mkdir(exist_ok=True)
        
        # 像素深度数据目录
        self.pixel_depth_data_dir = self.output_dir / "pixel_depth_data"
        self.pixel_depth_data_dir.mkdir(exist_ok=True)
        
        # 可视化图像目录
        self.saved_frames_viz_dir = self.output_dir / "saved_frames_visualization"
        self.saved_frames_viz_dir.mkdir(exist_ok=True)
        
        # ========== 深度统计目录 ==========
        self.depth_stats_dir = self.output_dir / "depth_statistics"
        self.depth_stats_dir.mkdir(exist_ok=True)
        
        # ========== 中心点3D坐标保存目录 ==========
        self.centers_3d_dir = self.output_dir / "centers_3d_info"
        self.centers_3d_dir.mkdir(exist_ok=True)
        
        # ========== 匹配详细信息目录 ==========
        self.matching_info_dir = self.output_dir / "matching_info"
        self.matching_info_dir.mkdir(exist_ok=True)
        
        print("="*60)
        print("🚀 初始化数据驱动3D匹配器")
        print("="*60)
        print(f"📊 配置参数:")
        print(f"   目标类别: {self.target_class} (进行匹配的类别)")
        print(f"   固定类别ID映射: {self.fixed_class_ids}")
        print(f"   候选初始帧: {self.first_frame_candidates}")
        print(f"   最小实例面积: {self.min_instance_area} 像素")
        print(f"   空间阈值: {self.spatial_threshold} 米")
        print(f"   运动阈值: {self.motion_threshold} 米")
        print(f"   运动权重: {self.motion_weight}")
        print(f"   双向权重: {self.bidirectional_weight}")
        print(f"   形状权重: {self.shape_weight}")
        print(f"   协同阈值: {self.coherence_threshold}")
        print(f"   最大深度: {self.max_depth} 米")
        print(f"   处理帧数限制: 前 {self.max_frames} 帧")
        
        if self.use_top_k_depths:
            print(f"   深度计算方法: 使用前 {self.top_k_min_depths} 个最小深度值的中位数")
        else:
            print(f"   深度计算方法: 使用所有有效深度的中位数")
        
        if self.pixel_depth_enabled:
            print("\n📝 像素深度保存配置:")
            print(f"   启用: {self.pixel_depth_enabled}")
            print(f"   要保存的帧: {self.frames_to_save}")
        print("="*60)
        
        # 加载数据
        self.load_camera_parameters()
        self.load_depth_maps()
        self.load_masks_with_class_info()
        
        # 提取掩码轮廓和矩
        self.extract_mask_features()
        
        # 计算所有实例的中心点和深度中位数
        self.compute_instance_centers()
        
        # 将2D中心点转换为3D
        self.convert_to_3d_centers()
        
        # 初始化固定类别的全局点
        self.initialize_fixed_class_points()
        
        # 自动选择最佳初始帧
        self.select_best_first_frame()
    
    # ==================== 选择最佳初始帧 ====================
    
    def select_best_first_frame(self):
        """从候选帧中选择实例数量最大的作为初始帧"""
        print("\n🔍 选择最佳初始帧...")
        print(f"   候选帧: {self.first_frame_candidates}")
        
        candidate_stats = []
        
        for frame_num in self.first_frame_candidates:
            if frame_num in self.instance_centers_3d:
                # 统计目标类实例数量
                target_instances = [inst_id for inst_id in self.instance_centers_3d[frame_num].keys()
                                   if inst_id <= self.frame_mask_counts.get(frame_num, 0)]
                instance_count = len(target_instances)
                
                if instance_count > 0:
                    candidate_stats.append({
                        'frame': frame_num,
                        'instance_count': instance_count,
                        'has_depth': frame_num in self.instance_depths,
                        'has_camera': frame_num in self.cameras
                    })
                    print(f"     帧 {frame_num:04d}: {instance_count} 个目标类实例")
                else:
                    print(f"     帧 {frame_num:04d}: 没有目标类实例")
            else:
                print(f"     帧 {frame_num:04d}: 没有3D中心点数据")
        
        if not candidate_stats:
            print("   ⚠️ 候选帧中都没有目标类实例，将使用第一个有实例的帧")
            # 找第一个有实例的帧
            for frame_num in sorted(self.instance_centers_3d.keys()):
                if frame_num in self.instance_centers_3d:
                    target_instances = [inst_id for inst_id in self.instance_centers_3d[frame_num].keys()
                                       if inst_id <= self.frame_mask_counts.get(frame_num, 0)]
                    if target_instances:
                        self.first_frame = frame_num
                        print(f"   选择帧 {frame_num:04d} 作为初始帧 ({len(target_instances)} 个实例)")
                        return
        
        # 按实例数量排序，选择最大的
        candidate_stats.sort(key=lambda x: -x['instance_count'])
        best_frame = candidate_stats[0]['frame']
        self.first_frame = best_frame
        
        print(f"\n✅ 选择帧 {best_frame:04d} 作为初始帧 (实例数: {candidate_stats[0]['instance_count']})")
        
        # 如果还有并列第二的，也显示出来
        if len(candidate_stats) > 1 and candidate_stats[1]['instance_count'] == candidate_stats[0]['instance_count']:
            print(f"   ⚠️ 注意: 帧 {candidate_stats[1]['frame']:04d} 也有相同的实例数")
    
    # ==================== 颜色生成模块 ====================
    
    def generate_colormap(self, num_colors):
        """为全局ID生成颜色映射"""
        colors = {}
        
        if self.colormap == 'tab20':
            import matplotlib.cm as cm
            cmap = cm.get_cmap('tab20')
            for i in range(1, num_colors + 1):
                rgba = cmap((i-1) % 20 / 20)
                colors[i] = (int(rgba[2]*255), int(rgba[1]*255), int(rgba[0]*255))
        
        elif self.colormap == 'hsv':
            for i in range(1, num_colors + 1):
                hue = (i-1) / max(num_colors, 1)
                rgb = plt.cm.hsv(hue)[:3]
                colors[i] = (int(rgb[2]*255), int(rgb[1]*255), int(rgb[0]*255))
        
        else:  # 'random'
            for i in range(1, num_colors + 1):
                random.seed(i)
                colors[i] = (random.randint(0, 255), 
                           random.randint(0, 255), 
                           random.randint(0, 255))
        
        return colors
    
    def get_instance_color(self, global_id):
        """获取全局ID对应的颜色"""
        if global_id not in self.global_id_colors:
            max_current = max(self.global_id_colors.keys()) if self.global_id_colors else 0
            if global_id > max_current:
                new_colors = self.generate_colormap(global_id)
                self.global_id_colors.update(new_colors)
        
        return self.global_id_colors.get(global_id, (128, 128, 128))
    
    # ==================== 数据加载模块 ====================
    
    def load_camera_parameters(self):
        """加载相机参数"""
        print("\n📷 加载相机参数...")
        
        with open(self.camera_json, 'r') as f:
            camera_data = json.load(f)
        
        for cam_info in camera_data:
            img_name = cam_info.get('img_name', '')
            frame_str = Path(img_name).stem
            
            match = re.search(r'(\d+)', frame_str)
            if match:
                frame_num = int(match.group(1))
                
                # 新增：只保留前max_frames帧的相机参数
                if frame_num <= self.max_frames:
                    fx = cam_info['fx']
                    fy = cam_info['fy']
                    width = cam_info['width']
                    height = cam_info['height']
                    cx = width / 2
                    cy = height / 2
                    
                    intrinsics = np.array([
                        [fx, 0, cx],
                        [0, fy, cy],
                        [0, 0, 1]
                    ])
                    
                    rotation = np.array(cam_info['rotation'])
                    position = np.array(cam_info['position'])
                    
                    self.cameras[frame_num] = {
                        'intrinsics': intrinsics,
                        'rotation': rotation,
                        'position': position,
                        'width': width,
                        'height': height,
                        'fx': fx,
                        'fy': fy,
                        'cx': cx,
                        'cy': cy,
                        'frame': frame_num
                    }
        
        print(f"✅ 已加载 {len(self.cameras)} 帧相机参数 (前{self.max_frames}帧)")

    def load_depth_maps(self):
        """加载深度图并转换为米"""
        print("\n📊 加载深度图...")
        
        depth_files = []
        if self.depth_format == '16bit':
            depth_files.extend(sorted(Path(self.depth_dir).glob("*.png")))
            depth_files.extend(sorted(Path(self.depth_dir).glob("*.tif")))
        elif self.depth_format == 'npy':
            depth_files.extend(sorted(Path(self.depth_dir).glob("*.npy")))
        
        print(f"找到 {len(depth_files)} 个深度文件")
        
        for depth_file in tqdm(depth_files, desc="加载深度图"):
            frame_str = depth_file.stem
            if frame_str.startswith('depth_'):
                frame_str = frame_str.replace('depth_', '')
            
            match = re.search(r'(\d+)', frame_str)
            if match:
                frame_num = int(match.group(1))
                
                # 新增：只保留前max_frames帧的深度图
                if frame_num <= self.max_frames:
                    try:
                        if self.depth_format == 'npy':
                            depth_data = np.load(str(depth_file))
                            self.depth_maps[frame_num] = depth_data
                        else:
                            depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)
                            if depth_img is not None:
                                if len(depth_img.shape) == 3:
                                    depth_img = cv2.cvtColor(depth_img, cv2.COLOR_BGR2GRAY)
                                self.depth_maps[frame_num] = depth_img
                    except Exception as e:
                        print(f"警告: 加载深度文件 {depth_file} 时出错: {e}")
        
        # 将深度图转换为米
        print("\n🔄 转换深度图到米...")
        for frame_num, depth_map in tqdm(self.depth_maps.items(), desc="转换深度"):
            if self.depth_format == '16bit':
                depth_m = depth_map.astype(np.float32) / self.depth_scale
            else:
                depth_m = depth_map.astype(np.float32)
            
            # 标记无效深度
            depth_m[(depth_map <= 0) | (depth_m >= self.max_depth)] = np.nan
            
            self.depth_maps_meters[frame_num] = depth_m
        
        print(f"✅ 成功加载 {len(self.depth_maps)} 帧深度图 (前{self.max_frames}帧)")
    
    def extract_frame_number(self, filename):
        """从文件名中提取帧号"""
        match = re.search(r'mask_(\d+)', filename)
        if match:
            return int(match.group(1))
        
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        
        match = re.search(r'(\d+)', filename)
        if match:
            return int(match.group(1))
        
        return None

    def load_masks_with_class_info(self):
        """加载分割掩码，同时读取类别信息，只保留目标类别的实例"""
        print("\n🎭 加载分割掩码和类别信息...")
        print(f"   目标类别: {self.target_class}")
        print(f"   固定类别: {list(self.fixed_class_ids.keys())} (ID映射: {self.fixed_class_ids})")
        
        mask_files = list(self.mask_dir.glob("*.png"))
        mask_files.extend(self.mask_dir.glob("*.tif"))
        
        print(f"找到 {len(mask_files)} 个掩码文件")
        
        # 首先找到所有对应的class_info文件
        class_info_files = list(self.mask_dir.glob("class_info_*.json"))
        class_info_map = {}
        for info_file in class_info_files:
            # 从class_info_xxx.json中提取基础名称
            base_name = info_file.stem.replace('class_info_', '')
            class_info_map[base_name] = info_file
        
        print(f"找到 {len(class_info_map)} 个类别信息文件")
        
        frame_file_map = {}
        for mask_file in mask_files:
            frame_num = self.extract_frame_number(mask_file.stem)
            if frame_num is not None:
                # 新增：只保留前max_frames帧的掩码
                if frame_num <= self.max_frames:
                    frame_file_map[frame_num] = mask_file
            else:
                print(f"警告: 无法从文件名提取帧号: {mask_file.name}")
        
        print(f"成功解析 {len(frame_file_map)} 个帧号 (前{self.max_frames}帧)")
        
        sorted_frames = sorted(frame_file_map.keys())
        if sorted_frames:
            print(f"排序后的帧号范围: {sorted_frames[0]} - {sorted_frames[-1]}")
        
        # 不再在这里设置第一帧，将在select_best_first_frame中设置
        
        total_instances = 0
        target_instances = 0
        fixed_instances = {0: 0, 1: 0, 2: 0}
        
        for frame_num in tqdm(sorted_frames, desc="加载掩码"):
            mask_file = frame_file_map[frame_num]
            
            # 查找对应的类别信息文件
            base_name = mask_file.stem.replace('mask_', '')
            if base_name in class_info_map:
                with open(class_info_map[base_name], 'r') as f:
                    class_data = json.load(f)
                
                # 创建类别映射：instance_id -> class_id
                instance_to_class = {}
                for inst in class_data['instances']:
                    instance_to_class[inst['instance_id']] = inst['class_id']
            else:
                print(f"警告: 帧 {frame_num} 没有对应的类别信息文件")
                continue
            
            mask = cv2.imread(str(mask_file), cv2.IMREAD_UNCHANGED)
            if mask is None:
                print(f"警告: 无法读取 {mask_file}")
                continue
                
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            original_unique = np.unique(mask)
            original_unique = original_unique[original_unique != 0]
            original_count = len(original_unique)
            
            self.frame_original_counts[frame_num] = original_count
            
            # 根据类别信息分离实例
            all_instance_ids = np.unique(mask)
            all_instance_ids = all_instance_ids[all_instance_ids != 0]
            
            # 分类实例
            target_class_instances = []  # 目标类别的实例ID
            fixed_class_instances = defaultdict(list)  # 固定类别的实例ID
            
            for inst_id in all_instance_ids:
                if inst_id in instance_to_class:
                    class_id = instance_to_class[inst_id]
                    if class_id == self.target_class:
                        target_class_instances.append(inst_id)
                    elif class_id in self.fixed_class_ids:
                        fixed_class_instances[class_id].append(inst_id)
                else:
                    print(f"  警告: 实例 {inst_id} 在帧 {frame_num} 中没有类别信息")
            
            # 创建新掩码，只包含目标类别的实例
            new_mask = np.zeros_like(mask)
            
            # 首先添加固定类别的实例（保留原始ID）
            for class_id, inst_list in fixed_class_instances.items():
                fixed_global_id = self.fixed_class_ids[class_id]
                for inst_id in inst_list:
                    new_mask[mask == inst_id] = fixed_global_id
                    # 保存类别信息
                    self.frame_class_info[frame_num][fixed_global_id] = class_id
                    fixed_instances[class_id] += 1
                    total_instances += 1
            
            # 然后添加目标类别的实例（重新编号从1开始）
            target_local_id = 1
            for inst_id in target_class_instances:
                new_mask[mask == inst_id] = target_local_id
                # 保存类别信息
                self.frame_class_info[frame_num][target_local_id] = self.target_class
                target_local_id += 1
                target_instances += 1
                total_instances += 1
            
            self.masks[frame_num] = new_mask
            self.frame_mask_counts[frame_num] = target_local_id - 1  # 只统计目标类别的实例数
            
            if len(self.masks) <= 5 or frame_num in self.first_frame_candidates:
                print(f"  帧 {frame_num:04d}: 目标类实例 {target_local_id-1}个, "
                      f"固定类实例 C0:{len(fixed_class_instances[0])} "
                      f"C1:{len(fixed_class_instances[1])} "
                      f"C2:{len(fixed_class_instances[2])} "
                      f"(原始总数: {original_count})")
        
        print(f"\n✅ 成功加载 {len(self.masks)} 帧掩码 (前{self.max_frames}帧)")
        print(f"   总实例数: {total_instances}")
        print(f"   目标类别 {self.target_class} 实例数: {target_instances}")
        print(f"   固定类别0实例数: {fixed_instances[0]}")
        print(f"   固定类别1实例数: {fixed_instances[1]}")
        print(f"   固定类别2实例数: {fixed_instances[2]}")
    
    def initialize_fixed_class_points(self):
        """初始化固定类别的全局点"""
        print("\n🌟 初始化固定类别的全局点...")
        
        for class_id, global_id in self.fixed_class_ids.items():
            # 为每个固定类别创建一个虚拟的全局点
            self.global_centers_3d[global_id] = {
                'point': [0, 0, 0],  # 虚拟点
                'frames': [],
                'instances': [],
                'distances': [],
                'num_views': 0,
                'confidence': 1.0,
                'is_fixed': True,
                'class_id': class_id
            }
            
            self.global_stats[global_id] = {
                'avg_depth': 0,
                'total_area': 0,
                'num_views': 0,
                'class_id': class_id
            }
            
            print(f"   初始化固定ID {global_id} (类别{class_id})")
        
        # 更新颜色映射
        self.global_id_colors.update(self.generate_colormap(max(self.fixed_class_ids.values())))
        print(f"✅ 固定类别全局点初始化完成")
    
    def extract_mask_features(self):
        """提取掩码的轮廓和矩"""
        print("\n🔍 提取掩码特征...")
        
        for frame_num, mask in tqdm(self.masks.items(), desc="提取特征"):
            self.mask_contours[frame_num] = {}
            self.mask_moments[frame_num] = {}
            
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            
            for instance_id in instance_ids:
                mask_binary = (mask == instance_id).astype(np.uint8) * 255
                
                # 提取轮廓
                contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    self.mask_contours[frame_num][instance_id] = contours[0]
                
                # 计算矩
                moments = cv2.moments(mask_binary)
                self.mask_moments[frame_num][instance_id] = moments
        
        print("✅ 掩码特征提取完成")
    
    # ==================== 形状相似性计算 ====================
    
    def compute_shape_similarity(self, mask1, mask2):
        """
        计算两个掩码的形状相似度
        返回 0-1 之间的分数
        """
        # 方法1：Hu矩（旋转不变性）
        moments1 = cv2.moments(mask1.astype(np.uint8) * 255)
        moments2 = cv2.moments(mask2.astype(np.uint8) * 255)
        
        if moments1['m00'] == 0 or moments2['m00'] == 0:
            return 0.0
        
        hu1 = cv2.HuMoments(moments1).flatten()
        hu2 = cv2.HuMoments(moments2).flatten()
        
        # 计算Hu矩的相似度
        hu_diff = np.sum(np.abs(np.log(np.abs(hu1) + 1e-10) - np.log(np.abs(hu2) + 1e-10)))
        hu_similarity = 1.0 / (1.0 + hu_diff)
        
        # 方法2：面积相似度
        area1 = np.sum(mask1 > 0)
        area2 = np.sum(mask2 > 0)
        area_ratio = min(area1, area2) / max(area1, area2) if max(area1, area2) > 0 else 0
        
        # 方法3：轮廓相似度（如果轮廓存在）
        contour_similarity = 0.5
        try:
            # 计算轮廓的圆形度、伸长度等特征
            pass
        except:
            pass
        
        # 加权融合
        shape_score = 0.5 * hu_similarity + 0.5 * area_ratio
        
        return shape_score
    
    # ==================== 前K个最小深度计算方法 ====================
    
    def compute_top_k_depth_median(self, depths, k=None):
        """
        计算前K个最小深度的中位数
        """
        if k is None:
            k = self.top_k_min_depths
        
        if len(depths) == 0:
            return None, 0, None, None
        
        sorted_depths = np.sort(depths)
        k_actual = min(k, len(sorted_depths))
        top_k_depths = sorted_depths[:k_actual]
        
        median_depth = np.median(top_k_depths)
        min_depth = np.min(top_k_depths)
        max_of_top_k = np.max(top_k_depths)
        
        return median_depth, k_actual, min_depth, max_of_top_k
    
    # ==================== 核心计算模块 ====================
    
    def compute_instance_centers(self):
        """
        计算每个实例的中心点和深度中位数
        只计算目标类别的实例
        """
        print("\n🎯 计算实例中心点和深度中位数...")
        print(f"   只计算目标类别 {self.target_class} 的实例")
        print(f"   最小实例面积: {self.min_instance_area} 像素")
        
        if self.use_top_k_depths:
            print(f"   深度计算方法: 使用前 {self.top_k_min_depths} 个最小深度值的中位数")
        else:
            print(f"   深度计算方法: 使用所有有效深度的中位数")
        
        total_instances = 0
        frames_with_depth = 0
        instances_skipped_small_area = 0
        instances_skipped_no_depth = 0
        
        # 统计信息
        depth_stats = {
            'total_instances': 0,
            'instances_with_top_k': 0,
            'instances_using_all': 0,
            'avg_k_used': 0,
            'avg_pixel_count': 0,
            'avg_min_depth': 0,
            'avg_max_of_top_k': 0
        }
        
        depth_stats_file = self.depth_stats_dir / "depth_calculation_stats.txt"
        
        for frame_num, mask in tqdm(self.masks.items(), desc="处理帧"):
            self.frame_instances_with_depth[frame_num] = 0
            
            if frame_num not in self.depth_maps_meters:
                print(f"  警告: 帧 {frame_num} 没有对应的深度图")
                continue
            
            if frame_num not in self.cameras:
                print(f"  警告: 帧 {frame_num} 没有对应的相机参数")
                continue
            
            depth_map = self.depth_maps_meters[frame_num]
            
            if depth_map.shape != mask.shape:
                print(f"  警告: 帧 {frame_num} 深度图尺寸 {depth_map.shape} 与掩码尺寸 {mask.shape} 不匹配")
                continue
            
            frames_with_depth += 1
            
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            
            for instance_id in instance_ids:
                # 只处理目标类别的实例（即ID小于等于frame_mask_counts[frame_num]的实例）
                if instance_id <= self.frame_mask_counts.get(frame_num, 0):
                    # 这是目标类别的实例
                    pass
                else:
                    # 这是固定类别的实例，跳过计算
                    continue
                
                mask_binary = (mask == instance_id)
                
                y_indices, x_indices = np.where(mask_binary)
                
                if len(y_indices) < self.min_instance_area:
                    instances_skipped_small_area += 1
                    continue
                
                center_x = int(np.mean(x_indices))
                center_y = int(np.mean(y_indices))
                
                depths_in_mask = depth_map[mask_binary]
                valid_depths = depths_in_mask[~np.isnan(depths_in_mask)]
                
                has_valid_depth = False
                
                if len(valid_depths) == 0:
                    # 尝试使用中心点深度
                    if 0 <= center_y < depth_map.shape[0] and 0 <= center_x < depth_map.shape[1]:
                        center_depth = depth_map[center_y, center_x]
                        if not np.isnan(center_depth):
                            valid_depths = np.array([center_depth])
                            has_valid_depth = True
                else:
                    has_valid_depth = True
                
                if not has_valid_depth:
                    self.instance_centers_2d[frame_num][int(instance_id)] = (center_x, center_y)
                    self.instance_stats[frame_num][int(instance_id)] = {
                        'area': len(y_indices),
                        'has_depth': False,
                        'num_valid_depths': 0,
                        'class_id': self.target_class
                    }
                    instances_skipped_no_depth += 1
                    continue
                
                # 保存所有有效深度值
                self.instance_all_depths[frame_num][int(instance_id)] = valid_depths.tolist()
                
                # 计算深度中位数
                if self.use_top_k_depths and len(valid_depths) > 0:
                    depth_median, k_used, min_depth, max_of_top_k = self.compute_top_k_depth_median(
                        valid_depths, self.top_k_min_depths
                    )
                    
                    if depth_median is not None:
                        depth_stats['instances_with_top_k'] += 1
                        depth_stats['avg_k_used'] += k_used
                        depth_std = np.std(valid_depths[:k_used] if k_used < len(valid_depths) else valid_depths)
                    else:
                        depth_median = float(np.median(valid_depths))
                        depth_std = float(np.std(valid_depths))
                        depth_stats['instances_using_all'] += 1
                        k_used = len(valid_depths)
                        min_depth = float(np.min(valid_depths))
                        max_of_top_k = float(np.max(valid_depths[:min(self.top_k_min_depths, len(valid_depths))]))
                else:
                    depth_median = float(np.median(valid_depths))
                    depth_std = float(np.std(valid_depths))
                    k_used = len(valid_depths)
                    min_depth = float(np.min(valid_depths))
                    max_of_top_k = float(np.max(valid_depths))
                    depth_stats['instances_using_all'] += 1
                
                # 更新统计
                depth_stats['total_instances'] += 1
                depth_stats['avg_pixel_count'] += len(valid_depths)
                depth_stats['avg_min_depth'] += min_depth
                depth_stats['avg_max_of_top_k'] += max_of_top_k
                
                self.instance_centers_2d[frame_num][int(instance_id)] = (center_x, center_y)
                self.instance_depths[frame_num][int(instance_id)] = depth_median
                self.instance_stats[frame_num][int(instance_id)] = {
                    'area': len(y_indices),
                    'depth_median': depth_median,
                    'depth_std': depth_std,
                    'num_valid_depths': len(valid_depths),
                    'has_depth': True,
                    'depth_calculation_method': 'top_k' if (self.use_top_k_depths and k_used <= self.top_k_min_depths) else 'all',
                    'top_k_used': k_used if self.use_top_k_depths else len(valid_depths),
                    'top_k_ratio': (k_used / len(valid_depths)) if len(valid_depths) > 0 else 0,
                    'min_depth': min_depth,
                    'max_of_top_k': max_of_top_k,
                    'center_point': (center_x, center_y),
                    'class_id': self.target_class
                }
                
                self.frame_instances_with_depth[frame_num] += 1
                total_instances += 1
        
        # 计算平均值
        if depth_stats['total_instances'] > 0:
            depth_stats['avg_k_used'] /= depth_stats['total_instances']
            depth_stats['avg_pixel_count'] /= depth_stats['total_instances']
            depth_stats['avg_min_depth'] /= depth_stats['total_instances']
            depth_stats['avg_max_of_top_k'] /= depth_stats['total_instances']
        
        print(f"\n✅ 计算完成:")
        print(f"   处理帧数: {frames_with_depth}/{len(self.masks)}")
        print(f"   目标类实例数 (有有效深度): {total_instances}")
        print(f"   因面积过小跳过的实例: {instances_skipped_small_area}")
        print(f"   因无有效深度跳过的实例: {instances_skipped_no_depth}")
        
        print(f"\n📊 深度计算统计:")
        print(f"   使用前K个最小深度的实例: {depth_stats['instances_with_top_k']}")
        print(f"   使用全部深度的实例: {depth_stats['instances_using_all']}")
        print(f"   平均使用的像素数: {depth_stats['avg_k_used']:.1f}")
        print(f"   实例平均总像素数: {depth_stats['avg_pixel_count']:.1f}")
        print(f"   平均最小深度: {depth_stats['avg_min_depth']:.3f}m")
        print(f"   平均前K个最大深度: {depth_stats['avg_max_of_top_k']:.3f}m")
        
        # 保存深度统计信息到文件
        with open(depth_stats_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("深度计算统计报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"配置参数:\n")
            f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
            if self.use_top_k_depths:
                f.write(f"  K值: {self.top_k_min_depths}\n")
            f.write(f"  最小实例面积: {self.min_instance_area} 像素\n\n")
            
            f.write(f"总体统计:\n")
            f.write(f"  总实例数: {total_instances}\n")
            f.write(f"  使用前K个最小深度的实例: {depth_stats['instances_with_top_k']}\n")
            f.write(f"  使用全部深度的实例: {depth_stats['instances_using_all']}\n")
            f.write(f"  平均使用的像素数: {depth_stats['avg_k_used']:.1f}\n")
            f.write(f"  实例平均总像素数: {depth_stats['avg_pixel_count']:.1f}\n")
            f.write(f"  平均最小深度: {depth_stats['avg_min_depth']:.3f}m\n")
            f.write(f"  平均前K个最大深度: {depth_stats['avg_max_of_top_k']:.3f}m\n")
        
        print(f"\n✅ 深度统计已保存: {depth_stats_file}")
    
    def convert_to_3d_centers(self):
        """将2D中心点转换为3D坐标"""
        print("\n🔄 转换2D中心点到3D坐标...")
        
        total_converted = 0
        total_skipped = 0
        
        for frame_num in tqdm(self.instance_centers_2d.keys(), desc="转换"):
            if frame_num not in self.cameras:
                total_skipped += len(self.instance_centers_2d[frame_num])
                continue
            
            cam = self.cameras[frame_num]
            K = cam['intrinsics']
            R = cam['rotation']
            C = cam['position']
            
            K_inv = np.linalg.inv(K)
            
            for instance_id, (x, y) in self.instance_centers_2d[frame_num].items():
                if instance_id not in self.instance_depths[frame_num]:
                    total_skipped += 1
                    continue
                
                depth = self.instance_depths[frame_num][instance_id]
                
                point_2d_homo = np.array([x, y, 1.0])
                
                ray_camera = K_inv @ point_2d_homo
                ray_camera = ray_camera / np.linalg.norm(ray_camera)
                
                point_camera = ray_camera * depth
                
                point_world = R @ point_camera + C
                
                self.instance_centers_3d[frame_num][instance_id] = point_world.tolist()
                total_converted += 1
        
        print(f"✅ 转换完成:")
        print(f"   成功转换: {total_converted} 个实例")
        print(f"   跳过 (无深度或相机参数): {total_skipped} 个实例")
    
    # ==================== 初始化第一帧 ====================
    
    def initialize_with_first_frame(self):
        """用第一帧初始化全局ID（只处理目标类别的实例）"""
        if self.first_frame is None:
            print("❌ 没有找到第一帧")
            return None
        
        print(f"\n🌟 用第一帧初始化全局ID: {self.first_frame:04d}")
        
        if self.first_frame not in self.instance_centers_3d:
            print(f"❌ 第一帧 {self.first_frame} 没有3D中心点")
            return None
        
        instance_count = 0
        for instance_id in self.instance_centers_3d[self.first_frame].keys():
            # 只处理目标类别的实例
            if instance_id > self.frame_mask_counts.get(self.first_frame, 0):
                continue
                
            key = (self.first_frame, instance_id)
            if key not in self.global_instance_map:
                global_id = self.global_instance_counter
                self.global_instance_counter += 1
                
                self.global_instance_map[key] = global_id
                
                point = self.instance_centers_3d[self.first_frame][instance_id]
                stats = self.instance_stats[self.first_frame][instance_id]
                
                self.global_centers_3d[global_id] = {
                    'point': point,
                    'frames': [self.first_frame],
                    'instances': [(self.first_frame, instance_id)],
                    'distances': [],
                    'num_views': 1,
                    'confidence': 1.0,
                    'is_fixed': False,
                    'class_id': self.target_class
                }
                
                # 保存掩码
                if self.first_frame in self.masks and instance_id in np.unique(self.masks[self.first_frame]):
                    mask = (self.masks[self.first_frame] == instance_id)
                    self.global_masks[global_id] = {self.first_frame: mask}
                
                self.global_stats[global_id] = {
                    'avg_depth': stats['depth_median'],
                    'total_area': stats['area'],
                    'num_views': 1,
                    'class_id': self.target_class
                }
                
                # 初始化运动轨迹
                self.motion_trajectories[global_id].append((self.first_frame, point))
                
                instance_count += 1
                print(f"   初始ID {global_id:3d}: 帧{self.first_frame:04d}:{instance_id} "
                      f"(深度: {stats['depth_median']:.3f}m)")
        
        self.frame_matched_counts[self.first_frame] = instance_count
        
        # 更新颜色映射（包括固定类别）
        max_id = max(max(self.fixed_class_ids.values()), self.global_instance_counter - 1)
        self.global_id_colors.update(self.generate_colormap(max_id))
        
        print(f"✅ 初始化完成: 分配了 {instance_count} 个全局ID (目标类实例)")
        
        return self.first_frame
    
    # ==================== 数据驱动的运动预测 ====================
    
    def predict_position_by_motion(self, global_id, target_frame):
        """
        数据驱动的运动预测
        使用加权移动平均，不做物理假设
        """
        if not self.use_motion_prior:
            return None, 0
        
        # 跳过固定类别的点
        if global_id in self.fixed_class_ids.values():
            return None, 0
        
        if global_id not in self.motion_trajectories:
            return None, 0
        
        trajectory = self.motion_trajectories[global_id]
        if len(trajectory) < 2:
            return None, 0
        
        # 获取最近几帧
        last_frame, last_pos = trajectory[-1]
        
        # 方法1：线性预测（基于最近两帧）
        if len(trajectory) >= 2:
            prev_frame, prev_pos = trajectory[-2]
            velocity = np.array(last_pos) - np.array(prev_pos)
            linear_pred = np.array(last_pos) + velocity
        
        # 方法2：加权移动平均（考虑更多历史）
        if len(trajectory) >= 3:
            weights = [0.5, 0.3, 0.2]  # 越近权重越大
            weighted_sum = np.zeros(3)
            weight_total = 0
            
            for i, (_, pos) in enumerate(reversed(trajectory[-3:])):
                weighted_sum += weights[i] * np.array(pos)
                weight_total += weights[i]
            
            wma_pred = weighted_sum / weight_total
            
            # 融合两种预测
            if len(trajectory) >= 3:
                predicted_pos = 0.7 * linear_pred + 0.3 * wma_pred
            else:
                predicted_pos = linear_pred
        else:
            predicted_pos = linear_pred
        
        # 置信度：基于轨迹长度和运动平滑度
        length_conf = min(0.9, len(trajectory) * 0.1)
        
        # 运动平滑度（连续两帧位移的一致性）
        if len(trajectory) >= 3:
            v1 = np.array(trajectory[-1][1]) - np.array(trajectory[-2][1])
            v2 = np.array(trajectory[-2][1]) - np.array(trajectory[-3][1])
            
            # 计算方向一致性（余弦相似度）
            if np.linalg.norm(v1) > 0 and np.linalg.norm(v2) > 0:
                cos_sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
                smooth_conf = max(0.5, (cos_sim + 1) / 2)
            else:
                smooth_conf = 0.7
        else:
            smooth_conf = 0.7
        
        confidence = length_conf * smooth_conf
        
        return predicted_pos.tolist(), confidence
    
    def get_motion_predictions(self, frame_num, global_ids):
        """
        获取所有全局点的运动预测（跳过固定类别）
        """
        predictions = []
        
        for global_id in global_ids:
            # 跳过固定类别的点
            if global_id in self.fixed_class_ids.values():
                continue
                
            predicted_pos, confidence = self.predict_position_by_motion(global_id, frame_num)
            if predicted_pos is not None:
                predictions.append({
                    'global_id': global_id,
                    'predicted_pos': predicted_pos,
                    'confidence': confidence
                })
        
        # 按置信度排序
        predictions.sort(key=lambda x: -x['confidence'])
        
        return predictions
    
    # ==================== 协同匹配模块（核心） ====================
    
    def match_with_coherence(self, frame_num):
        """
        协同匹配：多证据融合
        只匹配目标类别的实例
        """
        if frame_num not in self.instance_centers_3d:
            return {}
        
        # 只获取目标类别的实例
        current_instances = {}
        for inst_id, point in self.instance_centers_3d[frame_num].items():
            if inst_id <= self.frame_mask_counts.get(frame_num, 0):
                current_instances[inst_id] = point
        
        if not current_instances:
            return {}
        
        # 获取全局点（跳过固定类别）
        global_points = []
        global_ids = []
        for gid, info in self.global_centers_3d.items():
            if gid in self.fixed_class_ids.values():
                continue
            global_points.append(info['point'])
            global_ids.append(gid)
        
        if not global_points:
            return {}
        
        global_points = np.array(global_points)
        frame_points = np.array(list(current_instances.values()))
        frame_ids = list(current_instances.keys())
        
        # 计算距离矩阵
        distances = cdist(frame_points, global_points)
        
        print(f"\n  🔍 协同匹配: 帧 {frame_num:04d} ({len(frame_ids)}目标类实例) <-> 全局点 ({len(global_ids)}个)")
        
        # ========== 第1步：收集所有证据 ==========
        match_evidence = defaultdict(list)  # (local_id, global_id) -> [证据列表]
        
        # 证据1：运动连续性预测
        motion_predictions = self.get_motion_predictions(frame_num, global_ids)
        for pred in motion_predictions:
            global_id = pred['global_id']
            predicted_pos = pred['predicted_pos']
            confidence = pred['confidence']
            
            for i, local_id in enumerate(frame_ids):
                dist = np.linalg.norm(np.array(predicted_pos) - frame_points[i])
                if dist < self.motion_threshold * 2:
                    # 运动得分：置信度 * 距离衰减
                    motion_score = confidence * (1.0 / (1.0 + dist * 5))
                    match_evidence[(local_id, global_id)].append({
                        'type': 'motion',
                        'confidence': confidence,
                        'distance': dist,
                        'score': motion_score,
                        'details': f'运动预测(置信度:{confidence:.2f}, 距离:{dist:.3f}m)'
                    })
        
        # 证据2：双向空间匹配
        for i, local_id in enumerate(frame_ids):
            for j, global_id in enumerate(global_ids):
                dist = distances[i, j]
                if dist < self.spatial_threshold * 1.5:
                    is_forward_nn = (np.argmin(distances[i]) == j)
                    is_backward_nn = (np.argmin(distances[:, j]) == i)
                    
                    if is_forward_nn and is_backward_nn:
                        bid_score = 1.0 / (1.0 + dist * 5) * 1.5
                        evidence_type = 'perfect_bidirectional'
                        details = f'完美双向(距离:{dist:.3f}m)'
                    elif is_forward_nn:
                        bid_score = 1.0 / (1.0 + dist * 5) * 1.0
                        evidence_type = 'forward_nn'
                        details = f'正向最近邻(距离:{dist:.3f}m)'
                    elif is_backward_nn:
                        bid_score = 1.0 / (1.0 + dist * 5) * 1.0
                        evidence_type = 'backward_nn'
                        details = f'反向最近邻(距离:{dist:.3f}m)'
                    else:
                        bid_score = 1.0 / (1.0 + dist * 5) * 0.6
                        evidence_type = 'distance'
                        details = f'距离证据(距离:{dist:.3f}m)'
                    
                    match_evidence[(local_id, global_id)].append({
                        'type': 'bidirectional',
                        'subtype': evidence_type,
                        'distance': dist,
                        'score': bid_score,
                        'details': details
                    })
        
        # 证据3：形状相似性
        for i, local_id in enumerate(frame_ids):
            # 当前帧的掩码
            current_mask = (self.masks[frame_num] == local_id)
            
            for j, global_id in enumerate(global_ids):
                # 获取这个全局点最近一帧的掩码
                last_frame, last_local_id = self.get_last_occurrence(global_id)
                if last_frame in self.masks and last_local_id in np.unique(self.masks[last_frame]):
                    last_mask = (self.masks[last_frame] == last_local_id)
                    
                    # 计算形状相似度
                    shape_score = self.compute_shape_similarity(current_mask, last_mask)
                    
                    if shape_score > 0.5:  # 阈值
                        match_evidence[(local_id, global_id)].append({
                            'type': 'shape',
                            'score': shape_score,
                            'details': f'形状相似(得分:{shape_score:.2f})'
                        })
        
        # 证据4：历史匹配
        for (local_id, global_id) in list(match_evidence.keys()):
            key = (frame_num, local_id)
            if key in self.global_instance_map and self.global_instance_map[key] == global_id:
                match_evidence[(local_id, global_id)].append({
                    'type': 'history',
                    'score': 0.8,
                    'details': '历史匹配'
                })
        
        # ========== 第2步：计算综合得分 ==========
        match_candidates = []
        
        for (local_id, global_id), evidences in match_evidence.items():
            total_score = 0
            motion_count = 0
            bidirectional_count = 0
            shape_count = 0
            evidence_details = []
            
            for ev in evidences:
                if ev['type'] == 'motion':
                    total_score += ev['score'] * self.motion_weight
                    motion_count += 1
                elif ev['type'] == 'bidirectional':
                    total_score += ev['score'] * self.bidirectional_weight
                    bidirectional_count += 1
                elif ev['type'] == 'shape':
                    total_score += ev['score'] * self.shape_weight
                    shape_count += 1
                elif ev['type'] == 'history':
                    total_score += ev['score'] * 0.1  # 历史权重低
                
                evidence_details.append(ev['details'])
            
            # 协同效应：多种证据加分
            evidence_types = (motion_count > 0) + (bidirectional_count > 0) + (shape_count > 0)
            if evidence_types >= 2:
                total_score *= 1.2
                match_type = '协同匹配'
            elif motion_count > 0:
                match_type = '运动匹配'
            elif bidirectional_count > 0:
                match_type = '双向匹配'
            elif shape_count > 0:
                match_type = '形状匹配'
            else:
                match_type = '历史匹配'
            
            match_candidates.append({
                'local_id': local_id,
                'global_id': global_id,
                'score': total_score,
                'match_type': match_type,
                'motion_count': motion_count,
                'bidirectional_count': bidirectional_count,
                'shape_count': shape_count,
                'evidence_count': len(evidences),
                'evidence_details': evidence_details
            })
        
        # 按得分排序
        match_candidates.sort(key=lambda x: -x['score'])
        
        # ========== 第3步：解决冲突 ==========
        final_matches = {}
        used_globals = set()
        used_locals = set()
        matching_details = []
        
        # 先处理协同匹配
        for cand in match_candidates:
            if cand['match_type'] == '协同匹配':
                if cand['local_id'] not in used_locals and cand['global_id'] not in used_globals:
                    final_matches[cand['local_id']] = {
                        'global_id': cand['global_id'],
                        'confidence': cand['score'],
                        'match_type': cand['match_type'],
                        'evidence_count': cand['evidence_count']
                    }
                    used_locals.add(cand['local_id'])
                    used_globals.add(cand['global_id'])
                    self.matching_stats['cooperative_matches'] += 1
                    
                    print(f"    🤝 协同匹配: 本地{cand['local_id']} <-> 全局{cand['global_id']} "
                          f"(得分:{cand['score']:.3f})")
        
        # 再处理单一证据的
        for cand in match_candidates:
            if cand['local_id'] not in used_locals and cand['global_id'] not in used_globals:
                # 检查本地冲突
                conflict = False
                for selected_local in final_matches:
                    i1 = frame_ids.index(selected_local)
                    i2 = frame_ids.index(cand['local_id'])
                    local_dist = np.linalg.norm(frame_points[i1] - frame_points[i2])
                    
                    if local_dist < self.coherence_threshold:
                        conflict = True
                        break
                
                if not conflict:
                    final_matches[cand['local_id']] = {
                        'global_id': cand['global_id'],
                        'confidence': cand['score'],
                        'match_type': cand['match_type'],
                        'evidence_count': cand['evidence_count']
                    }
                    used_locals.add(cand['local_id'])
                    used_globals.add(cand['global_id'])
                    
                    if cand['match_type'] == '运动匹配':
                        self.matching_stats['motion_matches'] += 1
                    elif cand['match_type'] == '双向匹配':
                        self.matching_stats['bidirectional_matches'] += 1
                    elif cand['match_type'] == '形状匹配':
                        self.matching_stats['shape_matches'] += 1
                    
                    print(f"    { '🌀' if cand['match_type']=='运动匹配' else '🔍' } {cand['match_type']}: 本地{cand['local_id']} <-> 全局{cand['global_id']} "
                          f"(得分:{cand['score']:.3f})")
        
        # 处理未匹配的实例
        for local_id in frame_ids:
            if local_id not in used_locals:
                stats = self.instance_stats[frame_num].get(local_id, {})
                
                i = frame_ids.index(local_id)
                min_dist = np.min(distances[i]) if len(distances[i]) > 0 else float('inf')
                min_dist_global = global_ids[np.argmin(distances[i])] if min_dist < float('inf') else None
                
                # 检查是否有证据
                has_evidence = False
                for (lid, gid), evidences in match_evidence.items():
                    if lid == local_id:
                        has_evidence = True
                        break
                
                if min_dist < self.spatial_threshold:
                    if has_evidence:
                        reason = f"有证据但被竞争 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
                    else:
                        reason = f"距离近但证据不足 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
                else:
                    reason = f"距离太远 (最近全局:{min_dist_global}, 距离:{min_dist:.3f}m)"
                
                matching_details.append({
                    'frame': frame_num,
                    'local_id': local_id,
                    'global_id': None,
                    'match_status': 'UNMATCHED',
                    'reason': reason,
                    'depth_method': stats.get('depth_calculation_method', 'unknown'),
                    'top_k_used': stats.get('top_k_used', 0),
                    'total_pixels': stats.get('num_valid_depths', 0)
                })
                
                print(f"    ⚠️ 未匹配: 本地{local_id} ({reason})")
        
        self.save_matching_details(frame_num, matching_details)
        
        print(f"    最终接受: {len(final_matches)}/{len(frame_ids)} 个匹配")
        print(f"    协同匹配: {self.matching_stats['cooperative_matches']} | 运动匹配: {self.matching_stats['motion_matches']} | 双向匹配: {self.matching_stats['bidirectional_matches']} | 形状匹配: {self.matching_stats['shape_matches']}")
        
        self.frame_matched_counts[frame_num] = len(final_matches)
        
        return final_matches
    
    def get_last_occurrence(self, global_id):
        """获取全局点最近一次出现的帧和本地ID"""
        if global_id in self.global_centers_3d:
            instances = self.global_centers_3d[global_id]['instances']
            if instances:
                return instances[-1]  # (frame_num, local_id)
        return None, None
    
    def update_global_point(self, global_id, new_point, frame_num, local_id, distance):
        """更新全局点的3D位置和轨迹"""
        # 跳过固定类别的点
        if global_id in self.fixed_class_ids.values():
            return new_point
        
        info = self.global_centers_3d[global_id]
        current_point = np.array(info['point'])
        current_views = info['num_views']
        
        new_avg_point = (current_point * current_views + new_point) / (current_views + 1)
        
        old_confidence = info.get('confidence', 0.5)
        new_confidence = 1.0 / (1.0 + np.mean(info['distances'] + [distance]) if info['distances'] else distance)
        info['confidence'] = (old_confidence * current_views + new_confidence) / (current_views + 1)
        
        info['point'] = new_avg_point.tolist()
        info['frames'].append(frame_num)
        info['instances'].append((frame_num, local_id))
        info['distances'].append(distance)
        info['num_views'] = current_views + 1
        
        # 保存掩码
        if frame_num in self.masks and local_id in np.unique(self.masks[frame_num]):
            mask = (self.masks[frame_num] == local_id)
            if global_id not in self.global_masks:
                self.global_masks[global_id] = {}
            self.global_masks[global_id][frame_num] = mask
        
        # 更新运动轨迹
        self.motion_trajectories[global_id].append((frame_num, new_point.tolist()))
        if len(self.motion_trajectories[global_id]) > 20:
            self.motion_trajectories[global_id].pop(0)
        
        stats = self.instance_stats[frame_num][local_id]
        self.global_stats[global_id]['num_views'] += 1
        self.global_stats[global_id]['total_area'] += stats['area']
        self.global_stats[global_id]['avg_depth'] = (
            (self.global_stats[global_id]['avg_depth'] * current_views + stats['depth_median']) / 
            (current_views + 1)
        )
        
        return new_avg_point
    
    # ==================== 像素深度保存模块 ====================
    
    def save_pixel_depths_to_txt(self):
        """保存指定帧的像素深度数据"""
        if not self.pixel_depth_enabled:
            return
        
        print("\n📝 保存像素深度数据到txt文件...")
        print(f"   要保存的帧: {self.frames_to_save}")
        
        valid_frames = []
        for frame_num in self.frames_to_save:
            # 新增：只保存前max_frames帧内的像素深度
            if frame_num <= self.max_frames and frame_num in self.masks and frame_num in self.depth_maps_meters:
                valid_frames.append(frame_num)
            else:
                print(f"   ⚠️ 帧 {frame_num} 超出处理范围或不存在，跳过")
        
        if not valid_frames:
            print("   ⚠️ 没有有效的帧可保存")
            return
        
        total_instances_saved = 0
        total_pixels_saved = 0
        
        for frame_num in tqdm(valid_frames, desc="保存像素深度"):
            mask = self.masks[frame_num]
            depth_map = self.depth_maps_meters[frame_num]
            
            instance_ids = np.unique(mask)
            instance_ids = instance_ids[instance_ids != 0]
            
            frame_pixel_count = 0
            frame_instance_count = 0
            
            if self.separate_files:
                for instance_id in instance_ids:
                    mask_binary = (mask == instance_id)
                    y_indices, x_indices = np.where(mask_binary)
                    
                    valid_pixels = []
                    valid_depths = []
                    for y, x in zip(y_indices, x_indices):
                        depth_val = depth_map[y, x]
                        if not np.isnan(depth_val) and self.pixel_min_depth <= depth_val <= self.pixel_max_depth:
                            valid_pixels.append((y, x, depth_val))
                            valid_depths.append(depth_val)
                    
                    if not valid_pixels:
                        continue
                    
                    valid_pixels.sort(key=lambda p: p[2])
                    
                    if self.use_top_k_depths and len(valid_depths) > 0:
                        depth_median_top_k, k_used, depth_min, depth_max_top_k = self.compute_top_k_depth_median(
                            valid_depths, self.top_k_min_depths
                        )
                        depth_median = depth_median_top_k
                    else:
                        depth_median = np.median(valid_depths)
                        depth_min = np.min(valid_depths)
                        depth_max_top_k = np.max(valid_depths)
                        k_used = len(valid_depths)
                    
                    depth_mean = np.mean(valid_depths)
                    depth_std = np.std(valid_depths)
                    
                    key = (frame_num, int(instance_id))
                    global_id = self.global_instance_map.get(key, instance_id if instance_id in self.fixed_class_ids.values() else None)
                    
                    stats = self.instance_stats[frame_num].get(int(instance_id), {})
                    depth_method = stats.get('depth_calculation_method', 'unknown')
                    
                    class_id = self.frame_class_info[frame_num].get(int(instance_id), 'unknown')
                    
                    filename = f"frame{frame_num:04d}_instance{instance_id}_global{global_id if global_id else 'unmatched'}_class{class_id}.txt"
                    filepath = self.pixel_depth_data_dir / filename
                    
                    with open(filepath, 'w') as f:
                        f.write(f"# ========== 实例深度数据 ==========\n")
                        f.write(f"# Frame: {frame_num}\n")
                        f.write(f"# Instance ID (local): {instance_id}\n")
                        f.write(f"# Global ID: {global_id if global_id else 'unmatched'}\n")
                        f.write(f"# Class ID: {class_id}\n")
                        f.write(f"# Pixel count: {len(valid_pixels)}\n")
                        f.write(f"# Depth method: {depth_method}\n")
                        f.write(f"# Top K used: {k_used}/{len(valid_pixels)}\n")
                        f.write(f"# Depth median: {depth_median:.{self.depth_decimals}f} m\n")
                        f.write(f"# Depth mean: {depth_mean:.{self.depth_decimals}f} m\n")
                        f.write(f"# Depth std: {depth_std:.{self.depth_decimals}f} m\n")
                        f.write(f"# Depth range: {depth_min:.{self.depth_decimals}f} - {depth_max_top_k:.{self.depth_decimals}f} m\n")
                        f.write("#" + "-"*50 + "\n")
                        
                        for y, x, depth_val in valid_pixels:
                            if self.include_coordinates:
                                f.write(f"{y:4d} {x:4d} {depth_val:.{self.depth_decimals}f}\n")
                            else:
                                f.write(f"{depth_val:.{self.depth_decimals}f}\n")
                    
                    frame_pixel_count += len(valid_pixels)
                    frame_instance_count += 1
                    total_pixels_saved += len(valid_pixels)
            
            if self.save_visualization:
                self.save_frame_visualization(frame_num)
            
            print(f"   帧 {frame_num:04d}: 保存了 {frame_instance_count} 个实例, {frame_pixel_count} 个像素")
        
        print(f"✅ 像素深度数据保存完成: {total_instances_saved} 实例, {total_pixels_saved} 像素")
    
    def save_frame_visualization(self, frame_num):
        """保存指定帧的可视化图像"""
        if frame_num not in self.masks:
            return
        
        mask = self.masks[frame_num]
        h, w = mask.shape
        
        color_mask = np.zeros((h, w, 3), dtype=np.uint8)
        
        for local_id in np.unique(mask):
            if local_id == 0:
                continue
            
            key = (frame_num, int(local_id))
            if key in self.global_instance_map:
                global_id = self.global_instance_map[key]
                color = self.get_instance_color(global_id)
                color_mask[mask == local_id] = color
            elif local_id in self.fixed_class_ids.values():
                # 固定类别的实例
                color = self.get_instance_color(local_id)
                color_mask[mask == local_id] = color
            else:
                color_mask[mask == local_id] = self.unmatched_color
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        for local_id in np.unique(mask):
            if local_id == 0:
                continue
            
            if frame_num in self.instance_centers_2d and local_id in self.instance_centers_2d[frame_num]:
                center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                
                depth_value = None
                if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
                    depth_value = self.instance_depths[frame_num][local_id]
                
                key = (frame_num, int(local_id))
                global_id = self.global_instance_map.get(key)
                
                class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
                
                texts = []
                if global_id is not None:
                    texts.append(f"ID:{global_id}")
                elif local_id in self.fixed_class_ids.values():
                    texts.append(f"ID:{local_id}(C{class_id})")
                if depth_value is not None:
                    texts.append(f"{depth_value:.2f}m")
                
                if not texts:
                    continue
                
                full_text = " ".join(texts)
                
                font_scale = self.id_font_scale if (global_id or local_id in self.fixed_class_ids.values()) else self.depth_font_scale
                thickness = 1
                
                (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                
                text_x = int(center_x - text_width / 2)
                text_y = int(center_y + text_height / 2)
                
                if 0 <= text_x < w and 0 <= text_y < h:
                    padding = 2
                    cv2.rectangle(color_mask, 
                                (text_x - padding, text_y - text_height - padding),
                                (text_x + text_width + padding, text_y + padding),
                                (0, 0, 0), -1)
                    
                    cv2.putText(color_mask, full_text, (text_x, text_y),
                              font, font_scale, self.text_color, thickness)
        
        info_text = f"Frame {frame_num:04d}"
        if self.use_top_k_depths:
            info_text += f" (Top-{self.top_k_min_depths} Depth)"
        cv2.putText(color_mask, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        output_path = self.saved_frames_viz_dir / f"frame_{frame_num:04d}_visualization.png"
        cv2.imwrite(str(output_path), color_mask)
    
    # ==================== 保存中心点3D坐标信息 ====================
    
    def save_all_centers_3d_info(self):
        """保存所有掩码中心点的3D坐标信息"""
        print("\n📌 保存所有掩码中心点的3D坐标信息...")
        
        for frame_num in tqdm(sorted(self.instance_centers_3d.keys()), desc="保存中心点3D坐标"):
            if frame_num not in self.instance_centers_3d or frame_num not in self.instance_centers_2d:
                continue
            
            centers_3d = self.instance_centers_3d[frame_num]
            centers_2d = self.instance_centers_2d[frame_num]
            depths = self.instance_depths.get(frame_num, {})
            stats = self.instance_stats.get(frame_num, {})
            all_depths = self.instance_all_depths.get(frame_num, {})
            
            if not centers_3d:
                continue
            
            filename = self.centers_3d_dir / f"frame_{frame_num:04d}_centers_3d.txt"
            
            with open(filename, 'w') as f:
                f.write("#" + "="*100 + "\n")
                f.write(f"# 帧 {frame_num:04d} 的所有掩码中心点3D坐标信息\n")
                f.write("#" + "="*100 + "\n\n")
                
                f.write("# local_id,global_id,class_id,center_2d_x,center_2d_y,depth_median,point_3d_x,point_3d_y,point_3d_z,depth_method,top_k_used,total_pixels\n")
                f.write("-"*100 + "\n")
                
                for local_id in sorted(centers_3d.keys()):
                    if local_id not in centers_2d:
                        continue
                    
                    center_x, center_y = centers_2d[local_id]
                    depth_median = depths.get(local_id, -1)
                    point_3d = centers_3d[local_id]
                    
                    key = (frame_num, local_id)
                    global_id = self.global_instance_map.get(key, local_id if local_id in self.fixed_class_ids.values() else -1)
                    
                    class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
                    
                    stat = stats.get(local_id, {})
                    depth_method = stat.get('depth_calculation_method', 'unknown')
                    top_k_used = stat.get('top_k_used', 0)
                    total_pixels = stat.get('num_valid_depths', 0)
                    
                    f.write(f"{local_id},{global_id},{class_id},{center_x:.2f},{center_y:.2f},{depth_median:.6f},"
                           f"{point_3d[0]:.6f},{point_3d[1]:.6f},{point_3d[2]:.6f},"
                           f"{depth_method},{top_k_used},{total_pixels}\n")
        
        summary_file = self.centers_3d_dir / "all_frames_centers_3d_summary.csv"
        with open(summary_file, 'w') as f:
            f.write("frame_num,local_id,global_id,class_id,center_2d_x,center_2d_y,depth_median,point_3d_x,point_3d_y,point_3d_z,depth_method,top_k_used,total_pixels\n")
            
            for frame_num in sorted(self.instance_centers_3d.keys()):
                if frame_num not in self.instance_centers_2d:
                    continue
                
                centers_3d = self.instance_centers_3d[frame_num]
                centers_2d = self.instance_centers_2d[frame_num]
                depths = self.instance_depths.get(frame_num, {})
                stats = self.instance_stats.get(frame_num, {})
                
                for local_id in sorted(centers_3d.keys()):
                    if local_id not in centers_2d:
                        continue
                    
                    center_x, center_y = centers_2d[local_id]
                    depth_median = depths.get(local_id, -1)
                    point_3d = centers_3d[local_id]
                    key = (frame_num, local_id)
                    global_id = self.global_instance_map.get(key, local_id if local_id in self.fixed_class_ids.values() else -1)
                    class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
                    stat = stats.get(local_id, {})
                    depth_method = stat.get('depth_calculation_method', 'unknown')
                    top_k_used = stat.get('top_k_used', 0)
                    total_pixels = stat.get('num_valid_depths', 0)
                    
                    f.write(f"{frame_num},{local_id},{global_id},{class_id},{center_x:.2f},{center_y:.2f},{depth_median:.6f},"
                           f"{point_3d[0]:.6f},{point_3d[1]:.6f},{point_3d[2]:.6f},"
                           f"{depth_method},{top_k_used},{total_pixels}\n")
        
        print(f"✅ 所有帧的中心点3D坐标已保存到: {self.centers_3d_dir}")
    
    # ==================== 保存匹配详细信息 ====================
    
    def save_matching_details(self, frame_num, matching_details):
        """保存当前帧的匹配详细信息"""
        filename = self.matching_info_dir / f"frame_{frame_num:04d}_matching_info.txt"
        
        with open(filename, 'w') as f:
            f.write("="*120 + "\n")
            f.write(f"帧 {frame_num:04d} 匹配详细信息\n")
            f.write("="*120 + "\n\n")
            
            f.write(f"空间阈值: {self.spatial_threshold} 米\n")
            f.write(f"运动阈值: {self.motion_threshold} 米\n")
            f.write(f"协同阈值: {self.coherence_threshold} 米\n\n")
            
            total_instances = len(matching_details)
            unmatched_count = sum(1 for d in matching_details if d['match_status'] == 'UNMATCHED')
            
            f.write(f"实例总数: {total_instances}\n")
            f.write(f"未匹配  : {unmatched_count}\n\n")
            
            f.write("-"*120 + "\n")
            f.write("详细匹配信息:\n")
            f.write("-"*120 + "\n\n")
            
            for detail in matching_details:
                if detail['match_status'] == 'UNMATCHED':
                    f.write(f"⚠️ 未匹配:\n")
                    f.write(f"   本地ID: {detail['local_id']}\n")
                    f.write(f"   原因: {detail['reason']}\n")
                    f.write(f"   深度方法: {detail['depth_method']}\n")
                    f.write(f"   使用像素: {detail.get('top_k_used', 0)}/{detail.get('total_pixels', 0)}\n")
                    f.write("\n")
            
            f.write("="*120 + "\n")
        
        print(f"   ✅ 匹配详细信息已保存: {filename}")
    
    # ==================== 主运行流程 ====================
    
    def run_matching(self):
        """运行完整的匹配流程"""
        print("\n" + "="*60)
        print("🚀 开始数据驱动协同匹配")
        print("="*60)
        
        first_frame = self.initialize_with_first_frame()
        if first_frame is None:
            print("❌ 初始化失败")
            return False
        
        all_frames = sorted(list(self.instance_centers_3d.keys()))
        max_frame = max(all_frames)
        
        # 按照时间顺序重组帧序列
        # 从第一帧开始，先处理后面的帧直到最后一帧
        forward_frames = [f for f in all_frames if f > first_frame]
        # 然后从第一帧开始，处理前面的帧（循环到开头）
        backward_frames = [f for f in all_frames if f < first_frame]
        
        # 按照时间顺序重组：初始帧 → 后面的帧 → 前面的帧（循环）
        ordered_frames = [first_frame] + forward_frames + backward_frames
        
        print(f"\n📊 处理顺序 (时间循环):")
        print(f"   第一帧 (初始): {first_frame:04d}")
        if forward_frames:
            print(f"   后续帧 (递增): {forward_frames}")
        if backward_frames:
            print(f"   循环帧 (从头开始): {backward_frames}")
        print(f"   完整顺序: {ordered_frames}")
        print(f"   总处理帧数: {len(all_frames)}/{self.max_frames}")
        
        processed_frames = {first_frame}
        self.total_matches = 0
        self.unmatched_total = 0
        
        # 按照时间顺序处理所有帧
        for current_frame in tqdm(ordered_frames[1:], desc="按时间顺序处理帧"):
            print(f"\n{'='*50}")
            print(f"📌 处理帧: {current_frame:04d} (时间顺序中的第 {ordered_frames.index(current_frame)+1} 帧)")
            print(f"{'='*50}")
            
            matches = self.match_with_coherence(current_frame)
            
            frame_matches = len(matches)
            frame_unmatched = len([k for k in self.instance_centers_3d[current_frame].keys() 
                                if k <= self.frame_mask_counts.get(current_frame, 0)]) - frame_matches
            
            if matches:
                for local_id, match_info in matches.items():
                    global_id = match_info['global_id']
                    distance = match_info.get('distance', 0.3)
                    point = np.array(self.instance_centers_3d[current_frame][local_id])
                    
                    key = (current_frame, local_id)
                    self.global_instance_map[key] = global_id
                    
                    self.update_global_point(global_id, point, current_frame, local_id, distance)
                    
                    self.total_matches += 1
                    if match_info['confidence'] > self.confidence_threshold:
                        self.matching_stats['high_confidence'] += 1
                    else:
                        self.matching_stats['low_confidence'] += 1
                
                print(f"\n  ✅ 匹配成功: {frame_matches} 个实例")
                if frame_unmatched > 0:
                    print(f"  ⚠️ 未匹配: {frame_unmatched} 个实例")
                    self.unmatched_total += frame_unmatched
            else:
                print(f"\n  ⚠️ 没有找到任何匹配")
                target_count = len([k for k in self.instance_centers_3d[current_frame].keys() 
                                if k <= self.frame_mask_counts.get(current_frame, 0)])
                self.unmatched_total += target_count
            
            processed_frames.add(current_frame)
            print(f"     已处理 {len(processed_frames)}/{len(all_frames)} 帧")
            print(f"     当前全局实例数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}")
        
        self.matching_stats['unmatched_instances'] = self.unmatched_total
        self.matching_stats['total_matches'] = self.total_matches
        
        print("\n" + "="*60)
        print("✅ 匹配完成！")
        print("="*60)
        print(f"第一帧: {first_frame:04d} (从候选帧 {self.first_frame_candidates} 中选择)")
        print(f"处理顺序: {ordered_frames}")
        print(f"   → 从初始帧向后处理到最后一帧")
        print(f"   → 然后从头开始处理到初始帧之前的帧")
        print(f"固定类别ID: 220(类别0), 221(类别1), 222(类别2)")
        print(f"目标类实例数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}")
        print(f"总匹配次数: {self.total_matches}")
        print(f"协同匹配: {self.matching_stats['cooperative_matches']}")
        print(f"运动匹配: {self.matching_stats['motion_matches']}")
        print(f"双向匹配: {self.matching_stats['bidirectional_matches']}")
        print(f"形状匹配: {self.matching_stats['shape_matches']}")
        print(f"高质量匹配: {self.matching_stats['high_confidence']}")
        print(f"低质量匹配: {self.matching_stats['low_confidence']}")
        print(f"未匹配实例总数: {self.unmatched_total}")
        
        self.save_results()
        self.save_all_centers_3d_info()
        self.verify_matching_results()
        
        return True
    
    # ==================== 结果保存模块 ====================
    
    # def save_results(self):
    #     """保存所有结果"""
    #     print("\n💾 保存结果...")
        
    #     mapping_file = self.output_dir / "id_mapping.json"
        
    #     mapping_data = {
    #         'first_frame': self.first_frame,
    #         'first_frame_candidates': self.first_frame_candidates,
    #         'fixed_class_ids': self.fixed_class_ids,
    #         'target_class': self.target_class,
    #         'global_instances': {},
    #         'frame_mappings': defaultdict(dict),
    #         'stats': self.matching_stats,
    #         'color_map': {str(k): v for k, v in self.global_id_colors.items()},
    #         'depth_config': {
    #             'use_top_k_depths': self.use_top_k_depths,
    #             'top_k_min_depths': self.top_k_min_depths
    #         },
    #         'processing_info': {
    #             'max_frames': self.max_frames,
    #             'processed_frames': len(self.masks)
    #         }
    #     }
        
    #     for (frame, local_id), global_id in self.global_instance_map.items():
    #         mapping_data['frame_mappings'][str(frame)][str(local_id)] = global_id
        
    #     for global_id, info in self.global_centers_3d.items():
    #         class_id = info.get('class_id', 'unknown')
    #         mapping_data['global_instances'][str(global_id)] = {
    #             'point_3d': info['point'],
    #             'frames': info['frames'],
    #             'instances': [(f, i) for f, i in info['instances']],
    #             'num_views': info['num_views'],
    #             'confidence': info.get('confidence', 0.5),
    #             'avg_distance': np.mean(info['distances']) if info['distances'] else 0,
    #             'color': self.global_id_colors.get(global_id, [128,128,128]),
    #             'class_id': class_id,
    #             'is_fixed': info.get('is_fixed', False)
    #         }
        
    #     mapping_data['summary'] = {
    #         'total_frames': len(self.masks),
    #         'total_instances': len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0),
    #         'global_ids': len(self.global_centers_3d),
    #         'fixed_global_ids': list(self.fixed_class_ids.values()),
    #         'target_global_ids': [gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()],
    #         'unmatched_instances': self.matching_stats['unmatched_instances'],
    #         'rejected_matches': self.matching_stats['rejected_matches'],
    #         'spatial_threshold': self.spatial_threshold,
    #         'motion_threshold': self.motion_threshold,
    #         'coherence_threshold': self.coherence_threshold,
    #         'min_instance_area': self.min_instance_area,
    #         'depth_calculation': 'top_k' if self.use_top_k_depths else 'all',
    #         'top_k_value': self.top_k_min_depths if self.use_top_k_depths else None,
    #         'max_frames': self.max_frames,
    #         'processed_frames': len(self.masks)
    #     }
        
    #     with open(mapping_file, 'w') as f:
    #         json.dump(mapping_data, f, indent=2)
        
    #     print(f"✅ ID映射已保存: {mapping_file}")
        
    #     # 生成灰度掩码
    #     unified_mask_dir = self.output_dir / "unified_masks"
    #     unified_mask_dir.mkdir(exist_ok=True)
        
    #     print("\n🎨 生成灰度统一ID掩码...")
        
    #     for frame_num, mask in tqdm(self.masks.items(), desc="生成灰度掩码"):
    #         new_mask = np.zeros_like(mask)
            
    #         for local_id in np.unique(mask):
    #             if local_id == 0:
    #                 continue
                
    #             if local_id in self.fixed_class_ids.values():
    #                 # 固定类别的实例
    #                 new_mask[mask == local_id] = local_id
    #             else:
    #                 key = (frame_num, int(local_id))
    #                 if key in self.global_instance_map:
    #                     global_id = self.global_instance_map[key]
    #                     new_mask[mask == local_id] = global_id
            
    #         output_path = unified_mask_dir / f"unified_mask_{frame_num:04d}.png"
    #         cv2.imwrite(str(output_path), new_mask)
        
    #     print(f"✅ 灰度掩码已保存到: {unified_mask_dir}")
        
    #     # 生成彩色掩码
    #     print("\n🎨 生成彩色统一ID掩码...")
        
    #     for frame_num, mask in tqdm(self.masks.items(), desc="生成彩色掩码"):
    #         h, w = mask.shape
    #         color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
    #         for local_id in np.unique(mask):
    #             if local_id == 0:
    #                 continue
                
    #             if local_id in self.fixed_class_ids.values():
    #                 # 固定类别的实例
    #                 color = self.get_instance_color(local_id)
    #                 color_mask[mask == local_id] = color
    #             else:
    #                 key = (frame_num, int(local_id))
    #                 if key in self.global_instance_map:
    #                     global_id = self.global_instance_map[key]
    #                     color = self.get_instance_color(global_id)
    #                     color_mask[mask == local_id] = color
    #                 else:
    #                     color_mask[mask == local_id] = self.unmatched_color
            
    #         output_path = self.color_masks_dir / f"mask_{frame_num:04d}.png"
    #         cv2.imwrite(str(output_path), color_mask)
        
    #     print(f"✅ 彩色掩码已保存到: {self.color_masks_dir}")
        
    #     # 生成带信息的彩色掩码
    #     print("\n🎨 生成带ID和深度信息的彩色掩码...")
        
    #     for frame_num, mask in tqdm(self.masks.items(), desc="生成带信息掩码"):
    #         h, w = mask.shape
    #         color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
    #         for local_id in np.unique(mask):
    #             if local_id == 0:
    #                 continue
                
    #             if local_id in self.fixed_class_ids.values():
    #                 color = self.get_instance_color(local_id)
    #                 color_mask[mask == local_id] = color
    #             else:
    #                 key = (frame_num, int(local_id))
    #                 if key in self.global_instance_map:
    #                     global_id = self.global_instance_map[key]
    #                     color = self.get_instance_color(global_id)
    #                     color_mask[mask == local_id] = color
    #                 else:
    #                     color_mask[mask == local_id] = self.unmatched_color
            
    #         for local_id in np.unique(mask):
    #             if local_id == 0:
    #                 continue
                
    #             if frame_num in self.instance_centers_2d and local_id in self.instance_centers_2d[frame_num]:
    #                 center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                    
    #                 depth_value = None
    #                 if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
    #                     depth_value = self.instance_depths[frame_num][local_id]
                    
    #                 if local_id in self.fixed_class_ids.values():
    #                     global_id = local_id
    #                     class_id = self.frame_class_info[frame_num].get(int(local_id), '?')
    #                     texts = [f"ID:{global_id}(C{class_id})"]
    #                 else:
    #                     key = (frame_num, int(local_id))
    #                     global_id = self.global_instance_map.get(key)
    #                     if global_id is None:
    #                         continue
    #                     texts = [f"ID:{global_id}"]
                    
    #                 if depth_value is not None:
    #                     texts.append(f"{depth_value:.2f}m")
                    
    #                 if not texts:
    #                     continue
                    
    #                 full_text = " ".join(texts)
                    
    #                 font = cv2.FONT_HERSHEY_SIMPLEX
    #                 font_scale = self.id_font_scale
    #                 thickness = 1
                    
    #                 (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                    
    #                 text_x = int(center_x - text_width / 2)
    #                 text_y = int(center_y + text_height / 2)
                    
    #                 if 0 <= text_x < w and 0 <= text_y < h:
    #                     padding = 2
    #                     cv2.rectangle(color_mask, 
    #                                 (text_x - padding, text_y - text_height - padding),
    #                                 (text_x + text_width + padding, text_y + padding),
    #                                 (0, 0, 0), -1)
                        
    #                     cv2.putText(color_mask, full_text, (text_x, text_y),
    #                               font, font_scale, self.text_color, thickness)
            
    #         output_path = self.color_masks_with_info_dir / f"mask_with_info_{frame_num:04d}.png"
    #         cv2.imwrite(str(output_path), color_mask)
        
    #     print(f"✅ 带信息彩色掩码已保存到: {self.color_masks_with_info_dir}")
        
    #     self.save_pixel_depths_to_txt()
        
    #     legend_file = self.output_dir / "color_legend.png"
    #     self.save_color_legend(legend_file)
        
    #     centers_3d_file = self.output_dir / "centers_3d.txt"
    #     with open(centers_3d_file, 'w') as f:
    #         f.write("# global_id x y z num_views avg_depth confidence first_frame class_id is_fixed color_rgb depth_method top_k_used\n")
    #         for global_id, info in self.global_centers_3d.items():
    #             point = info['point']
    #             stats = self.global_stats.get(global_id, {})
    #             avg_depth = stats.get('avg_depth', 0)
    #             confidence = info.get('confidence', 0.5)
    #             first_frame = info['frames'][0] if info['frames'] else 0
    #             color = self.global_id_colors.get(global_id, (128,128,128))
    #             class_id = info.get('class_id', 'unknown')
    #             is_fixed = info.get('is_fixed', False)
                
    #             if not is_fixed and info['frames']:
    #                 first_inst = info['instances'][0]
    #                 frame_num, local_id = first_inst
    #                 depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown')
    #                 top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0)
    #             else:
    #                 depth_method = 'fixed'
    #                 top_k_used = 0
                
    #             f.write(f"{global_id} {point[0]:.3f} {point[1]:.3f} {point[2]:.3f} "
    #                    f"{info['num_views']} {avg_depth:.3f} {confidence:.2f} {first_frame:04d} "
    #                    f"{class_id} {is_fixed} ({color[0]},{color[1]},{color[2]}) "
    #                    f"{depth_method} {top_k_used}\n")
        
    #     print(f"✅ 3D中心点已保存: {centers_3d_file}")
        
    #     report_file = self.output_dir / "matching_report.txt"
    #     with open(report_file, 'w') as f:
    #         f.write("="*60 + "\n")
    #         f.write("数据驱动3D匹配报告\n")
    #         f.write("="*60 + "\n\n")
            
    #         f.write(f"配置参数:\n")
    #         f.write(f"  目标类别: {self.target_class}\n")
    #         f.write(f"  固定类别ID: 220(类别0), 221(类别1), 222(类别2)\n")
    #         f.write(f"  候选初始帧: {self.first_frame_candidates}\n")
    #         f.write(f"  选中的初始帧: {self.first_frame}\n")
    #         f.write(f"  空间阈值: {self.spatial_threshold}米\n")
    #         f.write(f"  运动阈值: {self.motion_threshold}米\n")
    #         f.write(f"  协同阈值: {self.coherence_threshold}米\n")
    #         f.write(f"  运动权重: {self.motion_weight}\n")
    #         f.write(f"  双向权重: {self.bidirectional_weight}\n")
    #         f.write(f"  形状权重: {self.shape_weight}\n")
    #         f.write(f"  置信度阈值: {self.confidence_threshold}\n")
    #         f.write(f"  最小实例面积: {self.min_instance_area}像素\n")
    #         f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
    #         if self.use_top_k_depths:
    #             f.write(f"  K值: {self.top_k_min_depths}\n")
    #         f.write(f"  处理帧数限制: 前 {self.max_frames} 帧\n\n")
            
    #         f.write(f"数据统计:\n")
    #         f.write(f"  总帧数: {len(self.masks)}\n")
    #         f.write(f"  第一帧: {self.first_frame:04d}\n")
    #         f.write(f"  固定类全局ID: 220, 221, 222\n")
    #         f.write(f"  目标类全局ID数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}\n")
    #         f.write(f"  总实例数: {len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0)}\n")
    #         f.write(f"  未匹配实例数: {self.matching_stats['unmatched_instances']}\n\n")
            
    #         f.write(f"匹配统计:\n")
    #         f.write(f"  总匹配次数: {self.matching_stats['total_matches']}\n")
    #         f.write(f"  协同匹配: {self.matching_stats['cooperative_matches']}\n")
    #         f.write(f"  运动匹配: {self.matching_stats['motion_matches']}\n")
    #         f.write(f"  双向匹配: {self.matching_stats['bidirectional_matches']}\n")
    #         f.write(f"  形状匹配: {self.matching_stats['shape_matches']}\n")
    #         f.write(f"  高质量匹配: {self.matching_stats['high_confidence']}\n")
    #         f.write(f"  低质量匹配: {self.matching_stats['low_confidence']}\n")
        
    #     print(f"✅ 报告已保存: {report_file}")


    def save_results(self):
        """保存所有结果"""
        print("\n💾 保存结果...")
        
        mapping_file = self.output_dir / "id_mapping.json"
        
        mapping_data = {
            'first_frame': self.first_frame,
            'first_frame_candidates': self.first_frame_candidates,
            'fixed_class_ids': self.fixed_class_ids,
            'target_class': self.target_class,
            'global_instances': {},
            'frame_mappings': defaultdict(dict),
            'stats': self.matching_stats,
            'color_map': {str(k): v for k, v in self.global_id_colors.items()},
            'depth_config': {
                'use_top_k_depths': self.use_top_k_depths,
                'top_k_min_depths': self.top_k_min_depths
            },
            'processing_info': {
                'max_frames': self.max_frames,
                'processed_frames': len(self.masks)
            }
        }
        
        for (frame, local_id), global_id in self.global_instance_map.items():
            mapping_data['frame_mappings'][str(frame)][str(local_id)] = global_id
        
        for global_id, info in self.global_centers_3d.items():
            class_id = info.get('class_id', 'unknown')
            mapping_data['global_instances'][str(global_id)] = {
                'point_3d': info['point'],
                'frames': info['frames'],
                'instances': [(f, i) for f, i in info['instances']],
                'num_views': info['num_views'],
                'confidence': info.get('confidence', 0.5),
                'avg_distance': np.mean(info['distances']) if info['distances'] else 0,
                'color': self.global_id_colors.get(global_id, [128,128,128]),
                'class_id': class_id,
                'is_fixed': info.get('is_fixed', False)
            }
        
        mapping_data['summary'] = {
            'total_frames': len(self.masks),
            'total_instances': len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0),
            'global_ids': len(self.global_centers_3d),
            'fixed_global_ids': list(self.fixed_class_ids.values()),
            'target_global_ids': [gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()],
            'unmatched_instances': self.matching_stats['unmatched_instances'],
            'rejected_matches': self.matching_stats['rejected_matches'],
            'spatial_threshold': self.spatial_threshold,
            'motion_threshold': self.motion_threshold,
            'coherence_threshold': self.coherence_threshold,
            'min_instance_area': self.min_instance_area,
            'depth_calculation': 'top_k' if self.use_top_k_depths else 'all',
            'top_k_value': self.top_k_min_depths if self.use_top_k_depths else None,
            'max_frames': self.max_frames,
            'processed_frames': len(self.masks)
        }
        
        with open(mapping_file, 'w') as f:
            json.dump(mapping_data, f, indent=2)
        
        print(f"✅ ID映射已保存: {mapping_file}")
        
        # 生成灰度掩码
        unified_mask_dir = self.output_dir / "unified_masks"
        unified_mask_dir.mkdir(exist_ok=True)
        
        print("\n🎨 生成灰度统一ID掩码...")
        
        for frame_num, mask in tqdm(self.masks.items(), desc="生成灰度掩码"):
            new_mask = np.zeros_like(mask)
            
            for local_id in np.unique(mask):
                if local_id == 0:
                    continue
                
                if local_id in self.fixed_class_ids.values():
                    # 固定类别的实例
                    new_mask[mask == local_id] = local_id
                else:
                    key = (frame_num, int(local_id))
                    if key in self.global_instance_map:
                        global_id = self.global_instance_map[key]
                        new_mask[mask == local_id] = global_id
            
            output_path = unified_mask_dir / f"unified_mask_{frame_num:04d}.png"
            cv2.imwrite(str(output_path), new_mask)
        
        print(f"✅ 灰度掩码已保存到: {unified_mask_dir}")
        
        # 生成彩色掩码
        print("\n🎨 生成彩色统一ID掩码...")
        
        for frame_num, mask in tqdm(self.masks.items(), desc="生成彩色掩码"):
            h, w = mask.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            for local_id in np.unique(mask):
                if local_id == 0:
                    continue
                
                if local_id in self.fixed_class_ids.values():
                    # 固定类别的实例
                    color = self.get_instance_color(local_id)
                    color_mask[mask == local_id] = color
                else:
                    key = (frame_num, int(local_id))
                    if key in self.global_instance_map:
                        global_id = self.global_instance_map[key]
                        color = self.get_instance_color(global_id)
                        color_mask[mask == local_id] = color
                    else:
                        color_mask[mask == local_id] = self.unmatched_color
            
            output_path = self.color_masks_dir / f"mask_{frame_num:04d}.png"
            cv2.imwrite(str(output_path), color_mask)
        
        print(f"✅ 彩色掩码已保存到: {self.color_masks_dir}")
        
        # ========== 修改开始：生成带信息的彩色掩码（固定类别也显示ID信息） ==========
        print("\n🎨 生成带ID和深度信息的彩色掩码（所有类别都显示ID）...")
        
        for frame_num, mask in tqdm(self.masks.items(), desc="生成带信息掩码"):
            h, w = mask.shape
            color_mask = np.zeros((h, w, 3), dtype=np.uint8)
            
            # 第一步：为所有实例上色
            for local_id in np.unique(mask):
                if local_id == 0:
                    continue
                
                if local_id in self.fixed_class_ids.values():
                    # 固定类别的实例 - 上色
                    color = self.get_instance_color(local_id)
                    color_mask[mask == local_id] = color
                else:
                    key = (frame_num, int(local_id))
                    if key in self.global_instance_map:
                        global_id = self.global_instance_map[key]
                        color = self.get_instance_color(global_id)
                        color_mask[mask == local_id] = color
                    else:
                        color_mask[mask == local_id] = self.unmatched_color
            
            # 第二步：为所有实例添加文字信息（包括固定类别）
            for local_id in np.unique(mask):
                if local_id == 0:
                    continue
                
                # 获取中心点坐标（固定类别需要临时计算）
                if local_id in self.fixed_class_ids.values():
                    # 固定类别 - 临时计算中心点
                    y_indices, x_indices = np.where(mask == local_id)
                    if len(y_indices) == 0:
                        continue
                    center_x = int(np.mean(x_indices))
                    center_y = int(np.mean(y_indices))
                    
                    # 固定类别没有深度信息
                    depth_value = None
                    
                    # 固定类别的文字
                    class_map = {220: '0', 221: '1', 222: '2'}
                    texts = [f"ID:{local_id}(C{class_map[local_id]})"]
                    font_scale = self.id_font_scale
                    
                else:
                    # 目标类别 - 使用已有的中心点
                    if frame_num not in self.instance_centers_2d or local_id not in self.instance_centers_2d[frame_num]:
                        continue
                        
                    center_x, center_y = self.instance_centers_2d[frame_num][local_id]
                    
                    depth_value = None
                    if frame_num in self.instance_depths and local_id in self.instance_depths[frame_num]:
                        depth_value = self.instance_depths[frame_num][local_id]
                    
                    key = (frame_num, int(local_id))
                    global_id = self.global_instance_map.get(key)
                    if global_id is None:
                        continue
                    
                    texts = [f"ID:{global_id}"]
                    font_scale = self.id_font_scale
                
                # 添加深度信息（如果有）
                if depth_value is not None:
                    texts.append(f"{depth_value:.2f}m")
                
                if not texts:
                    continue
                
                full_text = " ".join(texts)
                
                # 绘制文字
                font = cv2.FONT_HERSHEY_SIMPLEX
                thickness = 1
                
                (text_width, text_height), baseline = cv2.getTextSize(full_text, font, font_scale, thickness)
                
                text_x = int(center_x - text_width / 2)
                text_y = int(center_y + text_height / 2)
                
                if 0 <= text_x < w and 0 <= text_y < h:
                    padding = 2
                    cv2.rectangle(color_mask, 
                                (text_x - padding, text_y - text_height - padding),
                                (text_x + text_width + padding, text_y + padding),
                                (0, 0, 0), -1)
                    
                    cv2.putText(color_mask, full_text, (text_x, text_y),
                            font, font_scale, self.text_color, thickness)
            
            output_path = self.color_masks_with_info_dir / f"mask_with_info_{frame_num:04d}.png"
            cv2.imwrite(str(output_path), color_mask)
        
        print(f"✅ 带信息彩色掩码已保存到: {self.color_masks_with_info_dir}（所有类别都显示ID）")
        # ========== 修改结束 ==========
        
        self.save_pixel_depths_to_txt()
        
        legend_file = self.output_dir / "color_legend.png"
        self.save_color_legend(legend_file)
        
        centers_3d_file = self.output_dir / "centers_3d.txt"
        with open(centers_3d_file, 'w') as f:
            f.write("# global_id x y z num_views avg_depth confidence first_frame class_id is_fixed color_rgb depth_method top_k_used\n")
            for global_id, info in self.global_centers_3d.items():
                point = info['point']
                stats = self.global_stats.get(global_id, {})
                avg_depth = stats.get('avg_depth', 0)
                confidence = info.get('confidence', 0.5)
                first_frame = info['frames'][0] if info['frames'] else 0
                color = self.global_id_colors.get(global_id, (128,128,128))
                class_id = info.get('class_id', 'unknown')
                is_fixed = info.get('is_fixed', False)
                
                if not is_fixed and info['frames']:
                    first_inst = info['instances'][0]
                    frame_num, local_id = first_inst
                    depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown')
                    top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0)
                else:
                    depth_method = 'fixed'
                    top_k_used = 0
                
                f.write(f"{global_id} {point[0]:.3f} {point[1]:.3f} {point[2]:.3f} "
                    f"{info['num_views']} {avg_depth:.3f} {confidence:.2f} {first_frame:04d} "
                    f"{class_id} {is_fixed} ({color[0]},{color[1]},{color[2]}) "
                    f"{depth_method} {top_k_used}\n")
        
        print(f"✅ 3D中心点已保存: {centers_3d_file}")
        
        report_file = self.output_dir / "matching_report.txt"
        with open(report_file, 'w') as f:
            f.write("="*60 + "\n")
            f.write("数据驱动3D匹配报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"配置参数:\n")
            f.write(f"  目标类别: {self.target_class}\n")
            f.write(f"  固定类别ID: 220(类别0), 221(类别1), 222(类别2)\n")
            f.write(f"  候选初始帧: {self.first_frame_candidates}\n")
            f.write(f"  选中的初始帧: {self.first_frame}\n")
            f.write(f"  空间阈值: {self.spatial_threshold}米\n")
            f.write(f"  运动阈值: {self.motion_threshold}米\n")
            f.write(f"  协同阈值: {self.coherence_threshold}米\n")
            f.write(f"  运动权重: {self.motion_weight}\n")
            f.write(f"  双向权重: {self.bidirectional_weight}\n")
            f.write(f"  形状权重: {self.shape_weight}\n")
            f.write(f"  置信度阈值: {self.confidence_threshold}\n")
            f.write(f"  最小实例面积: {self.min_instance_area}像素\n")
            f.write(f"  深度计算方法: {'前K个最小深度' if self.use_top_k_depths else '全部深度'}\n")
            if self.use_top_k_depths:
                f.write(f"  K值: {self.top_k_min_depths}\n")
            f.write(f"  处理帧数限制: 前 {self.max_frames} 帧\n\n")
            
            f.write(f"数据统计:\n")
            f.write(f"  总帧数: {len(self.masks)}\n")
            f.write(f"  第一帧: {self.first_frame:04d}\n")
            f.write(f"  固定类全局ID: 220, 221, 222\n")
            f.write(f"  目标类全局ID数: {len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])}\n")
            f.write(f"  总实例数: {len(self.global_instance_map) + sum(1 for gid in self.fixed_class_ids.values() if self.global_centers_3d[gid]['num_views'] > 0)}\n")
            f.write(f"  未匹配实例数: {self.matching_stats['unmatched_instances']}\n\n")
            
            f.write(f"匹配统计:\n")
            f.write(f"  总匹配次数: {self.matching_stats['total_matches']}\n")
            f.write(f"  协同匹配: {self.matching_stats['cooperative_matches']}\n")
            f.write(f"  运动匹配: {self.matching_stats['motion_matches']}\n")
            f.write(f"  双向匹配: {self.matching_stats['bidirectional_matches']}\n")
            f.write(f"  形状匹配: {self.matching_stats['shape_matches']}\n")
            f.write(f"  高质量匹配: {self.matching_stats['high_confidence']}\n")
            f.write(f"  低质量匹配: {self.matching_stats['low_confidence']}\n")
        
        print(f"✅ 报告已保存: {report_file}")




    
    def save_color_legend(self, output_path):
        """保存颜色图例"""
        if not self.global_centers_3d:
            return
        
        num_ids = len(self.global_centers_3d)
        legend_height = max(30 * num_ids, 100)
        legend_width = 600
        
        legend = np.ones((legend_height, legend_width, 3), dtype=np.uint8) * 255
        
        # 先显示固定类别
        fixed_ids = sorted([gid for gid in self.global_centers_3d.keys() if gid in self.fixed_class_ids.values()])
        dynamic_ids = sorted([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()])
        
        i = 0
        for global_id in fixed_ids + dynamic_ids:
            y_start = i * 30 + 10
            y_end = y_start + 20
            
            color = self.global_id_colors.get(global_id, (128,128,128))
            
            cv2.rectangle(legend, (20, y_start), (60, y_end), color, -1)
            cv2.rectangle(legend, (20, y_start), (60, y_end), (0,0,0), 1)
            
            info = self.global_centers_3d[global_id]
            class_id = info.get('class_id', '?')
            is_fixed = info.get('is_fixed', False)
            
            if is_fixed:
                text = f"ID {global_id:3d} (C{class_id}): FIXED CLASS"
            else:
                first_inst = info['instances'][0] if info['instances'] else (0,0)
                frame_num, local_id = first_inst
                depth_method = self.instance_stats[frame_num].get(local_id, {}).get('depth_calculation_method', 'unknown') if frame_num in self.instance_stats else 'unknown'
                top_k_used = self.instance_stats[frame_num].get(local_id, {}).get('top_k_used', 0) if frame_num in self.instance_stats else 0
                total_pixels = self.instance_stats[frame_num].get(local_id, {}).get('num_valid_depths', 0) if frame_num in self.instance_stats else 0
                
                method_text = f" ({depth_method}:{top_k_used}/{total_pixels})" if self.use_top_k_depths else ""
                text = f"ID {global_id:3d} (C{class_id}): {info['num_views']}帧, 置信度:{info.get('confidence',0.5):.2f}{method_text}"
            
            cv2.putText(legend, text, (70, y_end-2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            i += 1
        
        title = f"初始帧: {self.first_frame:04d} | 固定ID: 220(C0), 221(C1), 222(C2) | 目标类: {self.target_class} | 深度计算方法: {'前K个最小深度 (K=' + str(self.top_k_min_depths) + ')' if self.use_top_k_depths else '全部深度'}"
        cv2.putText(legend, title, (20, legend_height-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        
        cv2.imwrite(str(output_path), legend)
        print(f"✅ 颜色图例已保存: {output_path}")
    
    # ==================== 验证模块 ====================
    
    def verify_matching_results(self):
        """验证匹配结果"""
        print("\n" + "="*60)
        print("🔍 验证匹配结果")
        print("="*60)
        
        all_frames = sorted(self.frame_matched_counts.keys())
        
        print("\n每帧匹配情况:")
        print("-"*100)
        print(f"{'帧号':<8} {'原始掩码':<10} {'有深度':<10} {'目标类':<10} {'匹配数':<10} {'匹配率(%)':<10} {'未匹配':<10}")
        print("-"*100)
        
        total_original = 0
        total_with_depth = 0
        total_target = 0
        total_matched = 0
        
        for frame_num in all_frames:
            original = self.frame_original_counts.get(frame_num, 0)
            with_depth = self.frame_instances_with_depth.get(frame_num, 0)
            target_count = self.frame_mask_counts.get(frame_num, 0)
            matched = self.frame_matched_counts.get(frame_num, 0)
            unmatched = with_depth - matched
            match_rate = (matched / target_count * 100) if target_count > 0 else 0
            
            total_original += original
            total_with_depth += with_depth
            total_target += target_count
            total_matched += matched
            
            marker = " ⚠️" if match_rate < 50 and target_count > 0 else ""
            
            print(f"{frame_num:04d}    {original:<10} {with_depth:<10} {target_count:<10} {matched:<10} {match_rate:<10.1f}{marker} {unmatched:<10}")
        
        print("-"*100)
        print(f"{'总计':<8} {total_original:<10} {total_with_depth:<10} {total_target:<10} {total_matched:<10} {(total_matched/total_target*100):<10.1f} {total_with_depth-total_matched:<10}")
    
    # ==================== 可视化模块 ====================
    
    def visualize_results(self):
        """可视化匹配结果"""
        print("\n📊 生成可视化...")
        
        if not self.global_centers_3d:
            print("没有全局点可可视化")
            return
        
        fig = plt.figure(figsize=(15, 12))
        
        ax1 = fig.add_subplot(221, projection='3d')
        
        points_3d = []
        point_colors = []
        point_labels = []
        
        for global_id, info in self.global_centers_3d.items():
            point = info['point']
            points_3d.append(point)
            color_rgb = np.array(self.global_id_colors.get(global_id, (128,128,128))) / 255.0
            point_colors.append(color_rgb)
            point_labels.append(f"ID:{global_id}")
        
        points_3d = np.array(points_3d)
        
        if len(points_3d) > 0:
            scatter = ax1.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2],
                                 c=point_colors, s=50)
            
            # 标注固定类别的点
            for i, (global_id, label) in enumerate(zip(self.global_centers_3d.keys(), point_labels)):
                if global_id in self.fixed_class_ids.values():
                    ax1.text(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], 
                            f" {label}", fontsize=8, color='red')
            
            if self.first_frame in self.instance_centers_3d:
                for local_id, point in self.instance_centers_3d[self.first_frame].items():
                    if local_id <= self.frame_mask_counts.get(self.first_frame, 0):
                        key = (self.first_frame, local_id)
                        if key in self.global_instance_map:
                            global_id = self.global_instance_map[key]
                            ax1.scatter(point[0], point[1], point[2], 
                                      c='red', s=100, marker='*')
        
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'3D中心点分布 (红*为第一帧 {self.first_frame:04d})')
        
        ax2 = fig.add_subplot(222)
        confidences = [info.get('confidence', 0.5) for gid, info in self.global_centers_3d.items() 
                      if gid not in self.fixed_class_ids.values()]
        if confidences:
            ax2.hist(confidences, bins=20, alpha=0.7, color='green')
            ax2.axvline(self.confidence_threshold, color='r', linestyle='--', 
                       label=f'Threshold: {self.confidence_threshold}')
        ax2.set_xlabel('Confidence')
        ax2.set_ylabel('Frequency')
        ax2.set_title('目标类实例置信度分布')
        ax2.legend()
        
        ax3 = fig.add_subplot(223)
        views_counts = [info['num_views'] for gid, info in self.global_centers_3d.items() 
                       if gid not in self.fixed_class_ids.values()]
        if views_counts:
            ax3.hist(views_counts, bins=20, alpha=0.7, color='blue')
        ax3.set_xlabel('Number of views')
        ax3.set_ylabel('Frequency')
        ax3.set_title('目标类视角数分布')
        
        ax4 = fig.add_subplot(224)
        stats_names = ['目标类实例', '固定类(3个)', '未匹配', '协同匹配', '运动匹配', '双向匹配', '形状匹配']
        stats_values = [
            len([gid for gid in self.global_centers_3d.keys() if gid not in self.fixed_class_ids.values()]),
            3,
            self.matching_stats['unmatched_instances'],
            self.matching_stats['cooperative_matches'],
            self.matching_stats['motion_matches'],
            self.matching_stats['bidirectional_matches'],
            self.matching_stats['shape_matches']
        ]
        colors = ['green', 'gray', 'red', 'purple', 'orange', 'cyan', 'yellow']
        ax4.bar(stats_names, stats_values, color=colors)
        ax4.set_ylabel('Count')
        ax4.set_title('匹配统计')
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        viz_path = self.output_dir / "visualization.png"
        plt.savefig(viz_path, dpi=150, bbox_inches='tight')
        plt.show()
        
        print(f"✅ 可视化已保存: {viz_path}")


# ==================== 主函数 ====================

# ==================== 主函数 ====================

def process_single_folder(folder_path, output_base_dir=None):
    """
    处理单个文件夹
    Args:
        folder_path: 输入文件夹路径，如 "/datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames"
        output_base_dir: 输出基础目录，如果为None则自动在输入文件夹下创建
    """
    folder_path = Path(folder_path)
    folder_name = folder_path.name
    
    print(f"\n{'='*60}")
    print(f"📁 处理文件夹: {folder_name}")
    print(f"{'='*60}")
    
    # 自动查找所需的子文件夹
    # 1. 查找 depth_dir: output_xxx/train/ours_30000/depth
    output_dirs = list(folder_path.glob("output_*"))
    depth_dir = None
    camera_json = None
    
    for output_dir in output_dirs:
        # 查找 depth 目录
        potential_depth = output_dir / "train" / "ours_30000" / "depth"
        if potential_depth.exists():
            depth_dir = str(potential_depth)
            print(f"✅ 找到深度图目录: {depth_dir}")
        
        # 查找 cameras.json
        potential_camera = output_dir / "cameras.json"
        if potential_camera.exists():
            camera_json = str(potential_camera)
            print(f"✅ 找到相机参数文件: {camera_json}")
    
    if not depth_dir:
        # 尝试其他可能的路径
        for output_dir in output_dirs:
            # 尝试直接找 depth 文件夹
            potential_depth = output_dir / "depth"
            if potential_depth.exists():
                depth_dir = str(potential_depth)
                print(f"✅ 找到深度图目录(备选): {depth_dir}")
                break
    
    if not camera_json:
        # 尝试在根目录查找
        potential_camera = folder_path / "cameras.json"
        if potential_camera.exists():
            camera_json = str(potential_camera)
            print(f"✅ 找到相机参数文件(根目录): {camera_json}")
    
    # 2. 查找 mask_dir: masks_results/integer_masks
    mask_dir = folder_path / "masks_results" / "integer_masks"
    if mask_dir.exists():
        mask_dir = str(mask_dir)
        print(f"✅ 找到掩码目录: {mask_dir}")
    else:
        # 尝试其他可能的掩码目录
        mask_candidates = list(folder_path.glob("**/integer_masks"))
        if mask_candidates:
            mask_dir = str(mask_candidates[0])
            print(f"✅ 找到掩码目录(备选): {mask_dir}")
    
    # 3. 确定输出目录
    if output_base_dir is None:
        output_dir = folder_path / "数据驱动匹配"
    else:
        output_dir = Path(output_base_dir) / folder_name
    
    output_dir = str(output_dir)
    print(f"📁 输出目录: {output_dir}")
    
    # 检查是否所有必要路径都找到了
    missing_paths = []
    if not depth_dir:
        missing_paths.append("深度图目录")
    if not camera_json:
        missing_paths.append("相机参数文件")
    if not mask_dir:
        missing_paths.append("掩码目录")
    
    if missing_paths:
        print(f"\n❌ 错误: 找不到以下路径: {', '.join(missing_paths)}")
        print("请确保文件夹结构正确:")
        print("  - output_xxx/train/ours_30000/depth 或 output_xxx/depth")
        print("  - output_xxx/cameras.json 或 根目录下的 cameras.json")
        print("  - masks_results/integer_masks")
        return None
    
    # 创建配置
    config = {
        'depth_dir': depth_dir,
        'camera_json': camera_json,
        'mask_dir': mask_dir,
        'output_dir': output_dir,
        
        # 目标类别和固定ID
        'target_class': 3,
        'fixed_class_ids': {
            0: 220,
            1: 221,
            2: 222
        },
        
        # 候选初始帧
        'first_frame_candidates': [1, 2, 3, 4, 5],
        
        # 参数配置
        'depth_scale': 1000.0,
        'depth_format': '16bit',
        'max_depth': 10.0,
        'spatial_threshold': 0.3,
        'motion_threshold': 0.5,
        'motion_weight': 0.4,
        'bidirectional_weight': 0.4,
        'shape_weight': 0.2,
        'coherence_threshold': 0.2,
        'confidence_threshold': 0.5,
        'min_instance_area': 5,
        
        # 处理帧数限制
        'max_frames': 9,
        
        # 运动相关配置
        'use_motion_prior': True,
        
        # 前K个最小深度的配置
        'use_top_k_depths': True,
        'top_k_min_depths': 2000,
        
        # 颜色映射方式
        'colormap': 'tab20',
        
        # 像素深度保存配置
        'pixel_depth_save': {
            'enabled': True,
            'frames_to_save': [3, 4],
            'depth_decimals': 3,
            'min_depth': 0.0,
            'max_depth': 10.0,
            'include_coordinates': True,
            'separate_files': True,
            'save_visualization': True,
        },
        
        # 文字绘制参数
        'id_font_scale': 0.4,
        'depth_font_scale': 0.3,
        'text_color': (255, 255, 255),
        'unmatched_color': (128, 128, 128),
        
        'first_frame': None
    }
    
    return config


def process_folders(folder_paths, output_base_dir=None):
    """
    处理一个或多个文件夹
    Args:
        folder_paths: 单个文件夹路径字符串，或文件夹路径列表
        output_base_dir: 输出基础目录（可选）
    """
    # 转换为列表
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]
    
    print("\n" + "="*80)
    print("🚀 开始处理多个文件夹")
    print("="*80)
    
    results = []
    
    for folder_path in folder_paths:
        try:
            # 为每个文件夹生成配置
            config = process_single_folder(folder_path, output_base_dir)
            
            if config:
                print(f"\n⚙️  开始处理 {Path(folder_path).name}...")
                
                # 创建匹配器并运行
                matcher = CenterPoint3DMatcher(config)
                success = matcher.run_matching()
                
                if success:
                    matcher.visualize_results()
                    results.append({
                        'folder': folder_path,
                        'success': True,
                        'output_dir': config['output_dir']
                    })
                    print(f"\n✅ 成功处理: {Path(folder_path).name}")
                else:
                    results.append({
                        'folder': folder_path,
                        'success': False,
                        'error': '匹配失败'
                    })
                    print(f"\n❌ 处理失败: {Path(folder_path).name}")
            else:
                results.append({
                    'folder': folder_path,
                    'success': False,
                    'error': '找不到必要的文件夹'
                })
                
        except Exception as e:
            print(f"\n❌ 处理 {folder_path} 时出错: {e}")
            import traceback
            traceback.print_exc()
            results.append({
                'folder': folder_path,
                'success': False,
                'error': str(e)
            })
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("📊 处理结果汇总")
    print("="*80)
    
    for result in results:
        folder_name = Path(result['folder']).name
        if result['success']:
            print(f"✅ {folder_name}: 成功 (输出: {result['output_dir']})")
        else:
            print(f"❌ {folder_name}: 失败 - {result.get('error', '未知错误')}")
    
    return results


def main():
    """主函数 - 支持单个或多个文件夹"""
    import argparse
    
    parser = argparse.ArgumentParser(description='数据驱动3D匹配器')
    parser.add_argument('folders', nargs='+', 
                       help='要处理的文件夹路径，可以指定一个或多个')
    parser.add_argument('--output', '-o', 
                       help='输出基础目录（可选）')
    
    args = parser.parse_args()
    
    # 处理文件夹
    results = process_folders(args.folders, args.output)
    
    # 统计结果
    success_count = sum(1 for r in results if r['success'])
    total_count = len(results)
    
    print("\n" + "="*80)
    print(f"🎉 处理完成！成功: {success_count}/{total_count}")
    print("="*80)


if __name__ == "__main__":
    main()


# 单个文件夹：python /datashare/dir_liusha/xibeinonglin/1_15_提取表型/0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测+双向匹配_0311_不跑全部帧.py /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames
# 多个文件夹：python /datashare/dir_liusha/xibeinonglin/1_15_提取表型/0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测+双向匹配_0311_不跑全部帧.py /datashare/dir_liusha/xibeinonglin/样本数据/91227-1_frames /datashare/dir_liusha/xibeinonglin/样本数据/B73-1_frames