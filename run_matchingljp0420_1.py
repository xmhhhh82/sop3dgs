#!/usr/bin/env python3
"""
run_matchingljp0326.py  (全帧匹配 + Tracking 证据)
====================================================
核心功能：
  1. auto_detect_max_depth()   —— 扫描实际深度图，自动设定合理的 max_depth
  2. 全帧匹配                    —— 不再使用关键帧，所有帧都参与匹配
  3. Tracking 证据               —— 读取 YOLO Tracking 的 track_id 作为匹配强信号
  4. 三轮渐进式匹配             —— 继承自 matching_fix_patch 的修复逻辑
  5. 更合理的匹配参数默认值     —— 针对玉米植株场景调优

用法：
  python run_matchingljp0326.py /path/to/scene
  python run_matchingljp0326.py /path/f1 /path/f2
  python run_matchingljp0326.py /path/to/scene --max_depth 80.0
"""

import sys

ORIGINAL_SCRIPT_DIR = "/datashare/dir_liusha/xibeinonglin/1_15_提取表型"
sys.path.insert(0, ORIGINAL_SCRIPT_DIR)

import os
import re
import glob
import json
import argparse
import numpy as np
import cv2
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── 导入原始 CenterPoint3DMatcher ─────────────────────────────
try:
    from importlib.util import spec_from_file_location, module_from_spec
    spec = spec_from_file_location(
        "matcher_module",
        os.path.join(ORIGINAL_SCRIPT_DIR,
                     "0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测"
                     "+双向匹配_0311_不跑全部帧.py")
    )
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    _OrigMatcher = mod.CenterPoint3DMatcher
    print("✅ 成功从原始脚本导入 CenterPoint3DMatcher")
except Exception as e:
    print(f"❌ 无法导入原始脚本: {e}")
    sys.exit(1)

from scipy.spatial.distance import cdist


# ══════════════════════════════════════════════════════════════
#  第一步：自动探测合理的 max_depth
# ══════════════════════════════════════════════════════════════

def auto_detect_max_depth(depth_dir: str, depth_scale: float = 1000.0,
                          sample_n: int = 10,
                          percentile: float = 95.0,
                          scene_hard_cap: float = 30.0) -> float:
    """
    扫描 depth_dir 中的深度图（采样 sample_n 帧），
    使用两阶段过滤：先 IQR 去除离群值，再取分位数。

    改进要点（修复深度噪声问题）:
      1. 第一阶段用 IQR 方法剔除极端离群深度（如 690m、951m）
      2. scene_hard_cap 为场景级硬性上限（玉米 ≤ 30m 足够）
      3. 最终 clip 范围从 [5, 500] 缩小到 [3, scene_hard_cap]

    Args:
        depth_dir      : 深度图目录
        depth_scale    : 原始像素值 → 米 的除数（16bit时通常1000）
        sample_n       : 最多采样帧数（避免全扫耗时）
        percentile     : 取多大百分位数（95 表示过滤掉最远5%的噪声点）
        scene_hard_cap : 场景硬性上限（米），玉米植株场景建议 30m

    Returns:
        建议的 max_depth（米），至少 3.0m，最多 scene_hard_cap
    """
    depth_dir = Path(depth_dir)
    files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.tif")))

    if not files:
        print(f"  ⚠️  未找到深度图，使用默认 max_depth=10.0m")
        return 10.0

    # 均匀采样
    step   = max(1, len(files) // sample_n)
    sample = files[::step][:sample_n]

    all_depths = []
    for f in sample:
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        if img is None:
            continue
        if len(img.shape) == 3:
            img = img[:, :, 0]
        d = img.astype(np.float32) / depth_scale
        # 阶段0: 预过滤明显无效值 & 应用硬性上限
        valid = d[(d > 0.1) & (d < scene_hard_cap)]
        if len(valid) > 0:
            all_depths.append(valid)

    if not all_depths:
        print(f"  ⚠️  深度图无有效像素，使用默认 max_depth=10.0m")
        return 10.0

    combined = np.concatenate(all_depths)

    # 阶段1: IQR 离群值剔除 —— 防止少量极端值污染 P95
    q25, q75 = np.percentile(combined, [25, 75])
    iqr = q75 - q25
    iqr_upper = q75 + 3.0 * iqr   # 3倍 IQR，相当宽松，只去极端值
    combined_clean = combined[combined <= iqr_upper]

    if len(combined_clean) < 100:
        # IQR 过滤太激进时回退到原始数据
        combined_clean = combined

    # 阶段2: 在干净数据上取分位数
    p_val     = float(np.percentile(combined_clean, percentile))
    max_depth = float(np.clip(p_val * 1.2, 3.0, scene_hard_cap))

    print(f"\n🔭 自动检测 max_depth (鲁棒去噪版):")
    print(f"   采样帧数     : {len(sample)}")
    print(f"   场景硬性上限 : {scene_hard_cap:.1f}m")
    print(f"   原始像素数   : {len(combined):,}")
    print(f"   IQR上限      : {iqr_upper:.2f}m  (Q25={q25:.2f}, Q75={q75:.2f})")
    print(f"   去噪后像素数 : {len(combined_clean):,}  "
          f"(去除 {len(combined)-len(combined_clean):,} 个离群点)")
    print(f"   深度中位数   : {np.median(combined_clean):.2f}m")
    print(f"   P{percentile:.0f} 深度     : {p_val:.2f}m")
    print(f"   → 设定 max_depth = {max_depth:.2f}m")
    return max_depth


# ══════════════════════════════════════════════════════════════
#  第二步：智能关键帧选择
# ══════════════════════════════════════════════════════════════

def _frame_quality_score(depth_map_path: str,
                         mask_dir: str,
                         depth_scale: float,
                         max_depth: float) -> dict:
    """
    计算单帧质量分，返回 dict:
      valid_ratio   : 掩码区域内有效深度像素比例（0~1）
      instance_count: 该帧检测到的实例数
      depth_stability: 1 / (深度标准差 / 深度均值)，越大越稳定
      quality_score  : 综合得分
    """
    # 解析帧号
    stem      = Path(depth_map_path).stem
    match     = re.search(r'(\d+)', stem.replace('depth_', ''))
    if not match:
        return None
    frame_num = int(match.group(1))

    # 读深度图
    img = cv2.imread(str(depth_map_path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    if len(img.shape) == 3:
        img = img[:, :, 0]
    depth_m = img.astype(np.float32) / depth_scale
    depth_m[(img <= 0) | (depth_m >= max_depth)] = np.nan

    # 读掩码（找对应帧的 mask_xxx.png）
    mask_path = Path(mask_dir) / f"mask_{frame_num:04d}.png"
    n_instances = 0
    valid_ratio = 0.0
    depth_stability = 0.0

    if mask_path.exists():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if mask is not None:
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            ids = np.unique(mask)
            ids = ids[ids != 0]
            n_instances = len(ids)

            # 掩码区域内有效深度比例
            mask_bool     = mask > 0
            depths_in_mask = depth_m[mask_bool]
            valid_pixels  = depths_in_mask[~np.isnan(depths_in_mask)]
            total_pixels  = np.sum(mask_bool)
            valid_ratio   = len(valid_pixels) / max(total_pixels, 1)

            # 深度稳定性
            if len(valid_pixels) > 10:
                mu  = float(np.mean(valid_pixels))
                std = float(np.std(valid_pixels))
                depth_stability = mu / (std + 1e-6)
    else:
        # 无掩码时用全图估算
        valid = depth_m[~np.isnan(depth_m)]
        valid_ratio = len(valid) / max(depth_m.size, 1)
        if len(valid) > 10:
            depth_stability = float(np.mean(valid)) / (float(np.std(valid)) + 1e-6)

    # 综合得分（可调权重）
    quality = (0.40 * valid_ratio +
               0.35 * min(n_instances / 10.0, 1.0) +   # 归一化，最多10个实例算满分
               0.25 * min(depth_stability / 5.0, 1.0))  # 归一化

    return {
        'frame_num':       frame_num,
        'valid_ratio':     valid_ratio,
        'instance_count':  n_instances,
        'depth_stability': depth_stability,
        'quality_score':   quality
    }


def select_keyframes(depth_dir: str,
                     mask_dir: str,
                     depth_scale: float,
                     max_depth: float,
                     n_keyframes: int = 15,
                     min_frames: int = 5) -> list:
    """
    从所有深度图帧中智能选取 n_keyframes 个关键帧。

    策略：
      1. 对所有帧计算质量分
      2. 将序列均匀切成 n_keyframes 段
      3. 每段取质量最高的帧
      4. 保证首段和尾段都有代表帧（首尾覆盖）

    Returns:
        排好序的帧号列表
    """
    depth_files = sorted(
        list(Path(depth_dir).glob("*.png")) +
        list(Path(depth_dir).glob("*.tif"))
    )

    if not depth_files:
        print(f"  ⚠️  depth_dir 为空，无法选关键帧")
        return []

    print(f"\n🎯 关键帧选择 (总帧数={len(depth_files)}, 目标={n_keyframes}帧)")

    # 计算所有帧质量（带进度条）
    scores = []
    for f in tqdm(depth_files, desc="评估帧质量"):
        s = _frame_quality_score(str(f), mask_dir, depth_scale, max_depth)
        if s is not None:
            scores.append(s)

    if not scores:
        print("  ⚠️  无有效质量分，回退到均匀采样")
        all_frames = []
        for f in depth_files:
            m = re.search(r'(\d+)', Path(f).stem.replace('depth_', ''))
            if m:
                all_frames.append(int(m.group(1)))
        step = max(1, len(all_frames) // n_keyframes)
        return sorted(all_frames[::step][:n_keyframes])

    scores.sort(key=lambda x: x['frame_num'])
    all_frame_nums = [s['frame_num'] for s in scores]
    n_total        = len(scores)
    n_select       = min(n_keyframes, n_total)

    # 均匀分段，每段取最高分帧
    segment_size = n_total / n_select
    selected     = []

    for seg_i in range(n_select):
        seg_start = int(seg_i * segment_size)
        seg_end   = int((seg_i + 1) * segment_size)
        seg_end   = min(seg_end, n_total)
        seg       = scores[seg_start:seg_end]
        if not seg:
            continue
        best = max(seg, key=lambda x: x['quality_score'])
        selected.append(best)

    selected_frames = sorted([s['frame_num'] for s in selected])

    # 打印质量统计
    print(f"\n📊 关键帧质量报告:")
    print(f"  {'帧号':>6}  {'有效深度%':>9}  {'实例数':>6}  {'深度稳定性':>10}  {'综合分':>6}")
    print(f"  {'-'*50}")
    for s in sorted(selected, key=lambda x: x['frame_num']):
        fn = s['frame_num']
        marker = " ◀ " if fn in selected_frames else ""
        print(f"  {fn:6d}  {s['valid_ratio']*100:8.1f}%  "
              f"{s['instance_count']:6d}  {s['depth_stability']:10.2f}  "
              f"{s['quality_score']:6.3f}{marker}")

    # 打印被排除的低质帧
    excluded = [s for s in scores
                if s['frame_num'] not in selected_frames]
    if excluded:
        worst = sorted(excluded, key=lambda x: x['quality_score'])[:5]
        print(f"\n  ℹ️  排除的低质帧（最差5个）: "
              f"{[s['frame_num'] for s in worst]}")

    print(f"\n✅ 选取 {len(selected_frames)} 个关键帧: {selected_frames}")
    return selected_frames


# ══════════════════════════════════════════════════════════════
#  修改版匹配器（三轮渐进式，继承自原版）
# ══════════════════════════════════════════════════════════════

class OptimizedCenterPoint3DMatcher(_OrigMatcher):
    """
    在原版基础上叠加以下优化：
      ① 自适应 spatial_threshold（按点云尺度）
      ② 三轮渐进式匹配（严格 → 宽松 → 最宽松）
      ③ 仍未匹配的实例分配新全局 ID
      ④ run_matching 兼容新建 ID 逻辑
      ⑤ 读取 YOLO Tracking 的 track_id，作为匹配强证据
    """

    def __init__(self, config):
        self._loose_multiplier     = config.get('loose_multiplier',     3.0)
        self._loosest_multiplier   = config.get('loosest_multiplier',   8.0)
        self._allow_new_global_ids = config.get('allow_new_global_ids', True)
        self._adaptive_threshold   = config.get('adaptive_threshold',   True)

        # track_id 相关数据结构
        self.frame_track_ids = {}    # {frame_num: {local_id: track_id}}
        self.global_track_ids = {}   # {global_id: track_id}

        super().__init__(config)
        self._load_track_ids()
        self._adapt_threshold()

    # ── 加载 track_id ──────────────────────────────────────────
    def _load_track_ids(self):
        """从 class_info JSON 中读取 track_id（如果存在）。"""
        mask_dir = Path(self.config['mask_dir'])
        info_files = sorted(mask_dir.glob("class_info_*.json"))
        n_with_track = 0

        for info_file in info_files:
            base_name = info_file.stem.replace('class_info_', '')
            m = re.search(r'(\d+)', base_name)
            if not m:
                continue
            frame_num = int(m.group(1))

            with open(info_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            tid_map = {}
            for inst in data.get('instances', []):
                if 'track_id' in inst:
                    # instance_id 在 mask 加载时会被重新编号
                    # 但 track_id 是基于原始 instance_id 的
                    tid_map[inst['instance_id']] = inst['track_id']

            if tid_map:
                self.frame_track_ids[frame_num] = tid_map
                n_with_track += 1

        if n_with_track > 0:
            print(f"\n🔗 Tracking 信息: {n_with_track}/{len(info_files)} 帧包含 track_id")
        else:
            print(f"\n⚠️  未检测到 track_id（class_info 中无 track_id 字段）")
            print(f"   3D 匹配将仅使用空间+运动+形状证据")

    def _get_remapped_track_id(self, frame_num, local_id):
        """
        获取重新编号后的 local_id 对应的 track_id。
        基类在加载 mask 时将 target_class 实例从 1 开始重新编号。
        我们需要建立 remapped_local_id -> track_id 的映射。
        """
        if frame_num not in self.frame_track_ids:
            return None

        # frame_track_ids 使用原始 instance_id（mask 文件中的像素值）
        # 基类在 load_masks_with_class_info 中将 target_class 实例
        # 按原始顺序重新编号为 1, 2, 3, ...
        # 所以 remapped local_id=1 对应第1个 target_class 实例的原始 instance_id
        # 我们需要通过 frame_class_info 来反查

        # 简化处理：如果 track_id map 有 local_id 作为 key，直接用
        tid_map = self.frame_track_ids[frame_num]
        if local_id in tid_map:
            return tid_map[local_id]

        # 否则尝试按顺序映射（remapped id N 对应原始第 N 个 target_class 实例）
        # 获取原始 mask 中按排序的 target_class instance_ids
        target_ids_sorted = sorted(
            iid for iid in tid_map.keys()
            if self.frame_class_info.get(frame_num, {}).get(local_id) == self.target_class
        )
        if 0 < local_id <= len(target_ids_sorted):
            orig_id = target_ids_sorted[local_id - 1]
            return tid_map.get(orig_id)

    # ── 自适应阈值 ─────────────────────────────────────────────
    def _adapt_threshold(self):
        if not self._adaptive_threshold:
            return
        all_pts = []
        for frame_num, frame_pts in self.instance_centers_3d.items():
            max_id = self.frame_mask_counts.get(frame_num, 0)
            for inst_id, pt in frame_pts.items():
                if inst_id <= max_id:
                    all_pts.append(pt)
        if len(all_pts) < 2:
            return
        pts_arr = np.array(all_pts)

        # ── 离群3D点剔除（per-axis IQR）──────────────────────
        # 防止少量被噪声深度反投影出的极远点撑大 bbox
        mask = np.ones(len(pts_arr), dtype=bool)
        for axis in range(pts_arr.shape[1]):
            vals = pts_arr[:, axis]
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            mask &= (vals >= q25 - 3.0 * iqr) & (vals <= q75 + 3.0 * iqr)
        pts_clean = pts_arr[mask]

        outlier_count = len(pts_arr) - len(pts_clean)
        if len(pts_clean) < 2:
            pts_clean = pts_arr  # 回退

        bbox_diag_raw   = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))
        bbox_diag_clean = np.linalg.norm(pts_clean.max(axis=0) - pts_clean.min(axis=0))

        # 硬性上限：玉米植株场景 bbox 不应超过 10m（株高+行间距）
        BBOX_DIAG_CAP = 10.0
        bbox_diag = min(bbox_diag_clean, BBOX_DIAG_CAP)

        # 取对角线的 3%（玉米叶间距比一般场景小，用更小比例）
        adaptive_base = max(self.spatial_threshold, bbox_diag * 0.03)

        # 同时设一个自适应阈值上限，防止 threshold 飞到不合理的值
        THRESHOLD_CAP = 1.0  # 玉米叶间距一般 < 0.5m，1m 已很宽松
        adaptive_base = min(adaptive_base, THRESHOLD_CAP)

        if adaptive_base > self.spatial_threshold * 1.1:
            print(f"\n🔧 自适应阈值: {self.spatial_threshold:.3f}m "
                  f"→ {adaptive_base:.3f}m")
            print(f"   bbox(原始)={bbox_diag_raw:.3f}m  "
                  f"bbox(去噪)={bbox_diag_clean:.3f}m  "
                  f"bbox(截断)={bbox_diag:.3f}m  "
                  f"去除3D离群点: {outlier_count}")
            self.spatial_threshold = adaptive_base
            self.motion_threshold  = max(self.motion_threshold,
                                         adaptive_base * 1.5)

    # ── 新建全局 ID ────────────────────────────────────────────
    def _assign_new_global_ids(self, frame_num, unmatched_local_ids):
        new_matches = {}
        for lid in unmatched_local_ids:
            if lid not in self.instance_centers_3d.get(frame_num, {}):
                continue
            stats = self.instance_stats.get(frame_num, {}).get(lid)
            if stats is None:
                continue
            point     = self.instance_centers_3d[frame_num][lid]
            global_id = self.global_instance_counter
            self.global_instance_counter += 1
            key = (frame_num, lid)
            self.global_instance_map[key] = global_id
            self.global_centers_3d[global_id] = {
                'point':     list(point) if not isinstance(point, list) else point,
                'frames':    [frame_num],
                'instances': [(frame_num, lid)],
                'distances': [],
                'num_views': 1,
                'confidence': 1.0,
                'is_fixed':  False,
                'class_id':  self.target_class
            }
            if frame_num in self.masks and lid in np.unique(self.masks[frame_num]):
                self.global_masks[global_id] = {
                    frame_num: (self.masks[frame_num] == lid)}
            self.global_stats[global_id] = {
                'avg_depth':  stats.get('depth_median', 0),
                'total_area': stats.get('area', 0),
                'num_views':  1,
                'class_id':   self.target_class
            }
            self.motion_trajectories[global_id].append(
                (frame_num, list(point) if not isinstance(point, list) else point))
            max_id = max(self.global_id_colors.keys()) if self.global_id_colors else 0
            if global_id > max_id:
                self.global_id_colors.update(self.generate_colormap(global_id))
            new_matches[lid] = {
                'global_id':    global_id,
                'confidence':   1.0,
                'match_type':   '新建',
                'evidence_count': 0
            }
            print(f"    🆕 本地{lid} → 新全局ID {global_id}")
        return new_matches

    # ── 三轮渐进匹配（核心） ────────────────────────────────────
    def match_with_coherence(self, frame_num):
        if frame_num not in self.instance_centers_3d:
            return {}

        current_instances = {
            iid: pt
            for iid, pt in self.instance_centers_3d[frame_num].items()
            if iid <= self.frame_mask_counts.get(frame_num, 0)
        }
        if not current_instances:
            return {}

        global_ids, global_points = [], []
        for gid, info in self.global_centers_3d.items():
            if gid in self.fixed_class_ids.values():
                continue
            global_ids.append(gid)
            global_points.append(info['point'])

        if not global_points:
            if self._allow_new_global_ids:
                nm = self._assign_new_global_ids(
                    frame_num, list(current_instances.keys()))
                self.frame_matched_counts[frame_num] = len(nm)
                return nm
            return {}

        gp_arr   = np.array(global_points)
        frame_ids = list(current_instances.keys())
        fp_arr   = np.array(list(current_instances.values()))
        distances = cdist(fp_arr, gp_arr)

        print(f"\n  🔍 帧{frame_num:04d}: "
              f"{len(frame_ids)}实例 <-> {len(global_ids)}全局点")

        # ── 单轮匹配函数 ────────────────────────────────────────
        def _one_round(pending, used_g, sp_thr, mo_thr, label):
            if not pending:
                return {}, []
            lidx = {lid: frame_ids.index(lid) for lid in pending}
            ev   = defaultdict(list)

            # tracking 证据（最强信号：同一 track_id 的实例优先匹配）
            if self.frame_track_ids:
                for lid in pending:
                    tid = self._get_remapped_track_id(frame_num, lid)
                    if tid is None:
                        continue
                    for gid in global_ids:
                        if gid in used_g:
                            continue
                        g_tid = self.global_track_ids.get(gid)
                        if g_tid is not None and g_tid == tid:
                            ev[(lid, gid)].append({
                                'type': 'tracking',
                                'score': 2.0})  # 高分，强证据

            # 运动证据
            for pred in self.get_motion_predictions(frame_num, global_ids):
                gid, ppos, conf = (pred['global_id'],
                                   pred['predicted_pos'], pred['confidence'])
                for lid, fi in lidx.items():
                    d = np.linalg.norm(np.array(ppos) - fp_arr[fi])
                    if d < mo_thr * 2:
                        ev[(lid, gid)].append({
                            'type': 'motion',
                            'score': conf / (1.0 + d * 5), 'distance': d})

            # 双向空间证据
            for lid, fi in lidx.items():
                for ji, gid in enumerate(global_ids):
                    if gid in used_g:
                        continue
                    d = distances[fi, ji]
                    if d >= sp_thr * 1.5:
                        continue
                    fwd = (np.argmin(distances[fi]) == ji)
                    bwd = (np.argmin(distances[:, ji]) == fi)
                    mult = 1.5 if (fwd and bwd) else (1.0 if (fwd or bwd) else 0.6)
                    ev[(lid, gid)].append({
                        'type': 'bidirectional',
                        'score': mult / (1.0 + d * 5), 'distance': d})

            # 形状证据
            for lid in pending:
                cm = (self.masks[frame_num] == lid)
                for gid in global_ids:
                    if gid in used_g:
                        continue
                    lf, ll = self.get_last_occurrence(gid)
                    if (lf in self.masks and
                            ll in np.unique(self.masks[lf])):
                        sh = self.compute_shape_similarity(
                            cm, (self.masks[lf] == ll))
                        if sh > 0.4:
                            ev[(lid, gid)].append({'type': 'shape', 'score': sh})

            # 综合得分
            cands = []
            for (lid, gid), evs in ev.items():
                total = nm = nb = ns = nt = 0
                for e in evs:
                    t = e['type']
                    if   t == 'tracking':     total += e['score'] * 1.0;                      nt += 1
                    elif t == 'motion':       total += e['score']*self.motion_weight;          nm += 1
                    elif t == 'bidirectional':total += e['score']*self.bidirectional_weight;    nb += 1
                    elif t == 'shape':        total += e['score']*self.shape_weight;            ns += 1
                n_t = (nt > 0) + (nm > 0) + (nb > 0) + (ns > 0)
                if n_t >= 2: total *= 1.2; mt = '协同'
                elif nt: mt = 'tracking'
                elif nm: mt = '运动'
                elif nb: mt = '双向'
                elif ns: mt = '形状'
                else:    mt = '其他'
                cands.append({'lid': lid, 'gid': gid, 'score': total, 'mt': mt})

            cands.sort(key=lambda x: -x['score'])
            rm = {}
            ul = set()
            for c in cands:
                lid, gid = c['lid'], c['gid']
                if lid in ul or gid in used_g:
                    continue
                conflict = any(
                    np.linalg.norm(fp_arr[frame_ids.index(sl)] -
                                   fp_arr[lidx[lid]])
                    < self.coherence_threshold
                    for sl in rm
                )
                if conflict:
                    continue
                rm[lid] = {'global_id': gid, 'confidence': c['score'],
                           'match_type': c['mt'], 'evidence_count': 1}
                ul.add(lid); used_g.add(gid)
                print(f"    [{label}] ✅ 本地{lid}<->全局{gid} "
                      f"({c['mt']},{c['score']:.3f})")
            still = [lid for lid in pending if lid not in ul]
            return rm, still

        # ── 执行三轮 ────────────────────────────────────────────
        final   = {}
        used_g  = set()
        details = []

        thr1 = self.spatial_threshold
        thr2 = thr1 * self._loose_multiplier
        thr3 = thr1 * self._loosest_multiplier

        r1, pending = _one_round(frame_ids, used_g, thr1, self.motion_threshold, "R1")
        final.update(r1)
        for m in r1.values():
            used_g.add(m['global_id'])
            self._stat(m['match_type'])

        if pending:
            print(f"\n  🔄 R2({len(pending)}个, thr={thr2:.3f}m)")
            r2, pending = _one_round(
                pending, used_g, thr2, self.motion_threshold * self._loose_multiplier, "R2")
            final.update(r2)
            for m in r2.values():
                used_g.add(m['global_id']); self._stat(m['match_type'])

        if pending:
            print(f"\n  🔄 R3 最近邻({len(pending)}个, thr={thr3:.3f}m)")
            still3 = []
            for lid in pending:
                fi  = frame_ids.index(lid)
                row = distances[fi].copy()
                for ji, gid in enumerate(global_ids):
                    if gid in used_g:
                        row[ji] = np.inf
                bj, bd = int(np.argmin(row)), float(np.min(row))
                bgid = global_ids[bj]
                if bd < thr3:
                    final[lid] = {'global_id': bgid,
                                  'confidence': 1.0/(1.0+bd),
                                  'match_type': '最近邻', 'evidence_count': 1}
                    used_g.add(bgid)
                    print(f"    [R3] ✅ 本地{lid}<->全局{bgid} (d={bd:.3f}m)")
                    self.matching_stats['bidirectional_matches'] += 1
                else:
                    still3.append(lid)
                    print(f"    [R3] ⚠️  本地{lid} d={bd:.3f}m > {thr3:.3f}m → 新ID")
            pending = still3

        if pending and self._allow_new_global_ids:
            nm = self._assign_new_global_ids(frame_num, pending)
            final.update(nm)
            pending = [lid for lid in pending if lid not in nm]

        for lid in pending:
            fi = frame_ids.index(lid)
            md = float(np.min(distances[fi]))
            mg = global_ids[int(np.argmin(distances[fi]))]
            details.append({
                'frame': frame_num, 'local_id': lid,
                'global_id': None, 'match_status': 'UNMATCHED',
                'reason': f'全策略失败(最近{mg},d={md:.3f}m)',
                'depth_method': self.instance_stats[frame_num].get(lid, {}).get(
                    'depth_calculation_method', 'unknown'),
                'top_k_used': 0, 'total_pixels': 0
            })

        self.save_matching_details(frame_num, details)
        self.frame_matched_counts[frame_num] = len(final)
        n_un = len(frame_ids) - len(final)
        print(f"\n  📊 {frame_num:04d}: {len(final)}/{len(frame_ids)} 匹配"
              f"{'  🎉' if n_un == 0 else f'  ({n_un}未匹配)'}")
        return final

    def _stat(self, mt):
        if 'tracking' in mt: self.matching_stats.setdefault('tracking_matches', 0); self.matching_stats['tracking_matches'] += 1
        elif '协同' in mt:  self.matching_stats['cooperative_matches']   += 1
        elif '运动' in mt: self.matching_stats['motion_matches']        += 1
        elif '双向' in mt: self.matching_stats['bidirectional_matches'] += 1
        elif '形状' in mt: self.matching_stats['shape_matches']         += 1

    # ── run_matching（兼容新建 ID） ─────────────────────────────
    def run_matching(self):
        print("\n" + "="*60)
        print("🚀 优化版三轮渐进匹配")
        print(f"   R1 阈值: {self.spatial_threshold:.3f}m")
        print(f"   R2 阈值: {self.spatial_threshold*self._loose_multiplier:.3f}m")
        print(f"   R3 阈值: {self.spatial_threshold*self._loosest_multiplier:.3f}m")
        print("="*60)

        first_frame = self.initialize_with_first_frame()
        if first_frame is None:
            print("❌ 初始化失败")
            return False

        # 为初始帧的全局实例记录 track_id
        for (fn, lid), gid in self.global_instance_map.items():
            if fn == first_frame:
                tid = self._get_remapped_track_id(fn, lid)
                if tid is not None:
                    self.global_track_ids[gid] = tid

        all_frames      = sorted(self.instance_centers_3d.keys())
        forward_frames  = [f for f in all_frames if f > first_frame]
        backward_frames = [f for f in all_frames if f < first_frame]
        ordered_frames  = [first_frame] + forward_frames + backward_frames

        print(f"\n处理顺序: {ordered_frames}")

        processed = {first_frame}
        self.total_matches = self.unmatched_total = 0

        for cf in tqdm(ordered_frames[1:], desc="处理帧"):
            print(f"\n{'='*50}\n📌 帧 {cf:04d}\n{'='*50}")
            matches      = self.match_with_coherence(cf)
            target_count = len([k for k in self.instance_centers_3d[cf]
                                 if k <= self.frame_mask_counts.get(cf, 0)])

            for lid, mi in matches.items():
                gid   = mi['global_id']
                point = np.array(self.instance_centers_3d[cf][lid])
                key   = (cf, lid)
                if key not in self.global_instance_map:
                    self.global_instance_map[key] = gid
                if mi.get('match_type') != '新建':
                    self.update_global_point(
                        gid, point, cf, lid, mi.get('distance', 0.3))
                self.total_matches += 1
                if mi['confidence'] > self.confidence_threshold:
                    self.matching_stats['high_confidence'] += 1
                else:
                    self.matching_stats['low_confidence'] += 1

                # 记录 track_id → global_id 映射
                tid = self._get_remapped_track_id(cf, lid)
                if tid is not None and gid not in self.global_track_ids:
                    self.global_track_ids[gid] = tid

            self.unmatched_total += max(0, target_count - len(matches))
            processed.add(cf)
            dyn = len([g for g in self.global_centers_3d
                        if g not in self.fixed_class_ids.values()])
            print(f"   已处理 {len(processed)}/{len(all_frames)} 帧 | 全局实例 {dyn}")

        self.matching_stats['unmatched_instances'] = self.unmatched_total
        self.matching_stats['total_matches']       = self.total_matches

        print("\n" + "="*60)
        print("✅ 匹配完成！")
        print(f"   总匹配: {self.total_matches}  "
              f"tracking:{self.matching_stats.get('tracking_matches', 0)}  "
              f"协同:{self.matching_stats['cooperative_matches']}  "
              f"运动:{self.matching_stats['motion_matches']}  "
              f"双向:{self.matching_stats['bidirectional_matches']}  "
              f"形状:{self.matching_stats['shape_matches']}")
        print(f"   仍未匹配: {self.unmatched_total}")
        print("="*60)

        self.save_results()
        self.save_all_centers_3d_info()
        self.verify_matching_results()
        return True


# ══════════════════════════════════════════════════════════════
#  路径解析 & 批量处理
# ══════════════════════════════════════════════════════════════

def build_config(folder_path, output_base_dir, max_depth, scene_hard_cap=30.0):
    """构建 config 字典，所有核心参数集中在此。全帧匹配模式。"""
    folder_path = Path(folder_path)
    folder_name = folder_path.name

    output_dirs = list(folder_path.glob("output_*"))
    depth_dir = camera_json = None

    for od in output_dirs:
        pd = od / "train" / "ours_30000" / "depth"
        if pd.exists():
            depth_dir = str(pd)
            print(f"✅ depth_dir   : {depth_dir}")
        pc = od / "cameras.json"
        if pc.exists():
            camera_json = str(pc)
            print(f"✅ camera_json : {camera_json}")

    if not depth_dir:
        for od in output_dirs:
            pd = od / "depth"
            if pd.exists():
                depth_dir = str(pd); break
    if not camera_json:
        pc = folder_path / "cameras.json"
        if pc.exists():
            camera_json = str(pc)

    mask_dir = folder_path / "masks_results" / "integer_masks"
    if mask_dir.exists():
        mask_dir = str(mask_dir)
    else:
        cands = list(folder_path.glob("**/integer_masks"))
        mask_dir = str(cands[0]) if cands else None

    output_dir = str(folder_path / "数据驱动匹配") \
        if output_base_dir is None \
        else str(Path(output_base_dir) / folder_name)
    print(f"📁 output_dir : {output_dir}")

    missing = (["depth_dir"] if not depth_dir else []) + \
              (["camera_json"] if not camera_json else []) + \
              (["mask_dir"] if not mask_dir else [])
    if missing:
        print(f"❌ 缺少路径: {missing}")
        return None

    # ── 自动检测 max_depth ──────────────────────────────────────
    if max_depth is None:
        max_depth = auto_detect_max_depth(
            depth_dir, depth_scale=1000.0, sample_n=10, percentile=95.0,
            scene_hard_cap=scene_hard_cap)
    else:
        # 即使手动指定，也用硬性上限保护
        if max_depth > scene_hard_cap:
            print(f"  ⚠️  手动 max_depth={max_depth:.1f}m 超过场景上限 "
                  f"{scene_hard_cap:.1f}m，截断为 {scene_hard_cap:.1f}m")
            max_depth = scene_hard_cap

    # ── 全帧匹配：扫描实际帧数 ──────────────────────────────────
    all_mask_files = sorted(Path(mask_dir).glob("mask_*.png"))
    all_frame_nums = []
    for mf in all_mask_files:
        m = re.search(r'(\d+)', mf.stem.replace('mask_', ''))
        if m:
            all_frame_nums.append(int(m.group(1)))

    if all_frame_nums:
        max_frames_val = max(all_frame_nums)
        print(f"   全帧匹配: 共 {len(all_frame_nums)} 帧 (1~{max_frames_val})")
    else:
        max_frames_val = 20
        print(f"   ⚠️  未找到 mask 文件，使用默认 max_frames={max_frames_val}")

    # 候选初始帧：取前5帧
    first_frame_candidates = sorted(all_frame_nums[:5]) if len(all_frame_nums) >= 5 \
        else (all_frame_nums if all_frame_nums else [1, 2, 3, 4, 5])

    print(f"\n⚙️  最终配置:")
    print(f"   max_depth              = {max_depth:.2f}m")
    print(f"   max_frames             = {max_frames_val}")
    print(f"   first_frame_candidates = {first_frame_candidates}")

    # ── top_k_min_depths 自适应 ──────────────────────────────────
    # 玉米叶片面积小，2000像素通常已超出单叶掩码大小，改为300
    # 同时允许用户按场景调整
    top_k = 300

    config = {
        'depth_dir':    depth_dir,
        'camera_json':  camera_json,
        'mask_dir':     mask_dir,
        'output_dir':   output_dir,

        'target_class':    3,
        'fixed_class_ids': {0: 220, 1: 221, 2: 222},
        'first_frame_candidates': first_frame_candidates,

        'depth_scale':         1000.0,
        'depth_format':        '16bit',
        'max_depth':           max_depth,          # ← 自动检测

        # ── 匹配阈值（自适应会进一步调整）────────────────────────
        'spatial_threshold':   0.15,   # 玉米叶间距约10~30cm，基础阈值设小
        'motion_threshold':    0.30,
        'loose_multiplier':    4.0,    # R2 = 0.15*4 = 0.6m
        'loosest_multiplier':  12.0,   # R3 = 0.15*12 = 1.8m
        'allow_new_global_ids': True,
        'adaptive_threshold':  True,

        'motion_weight':        0.4,
        'bidirectional_weight': 0.4,
        'shape_weight':         0.2,
        'coherence_threshold':  0.05,  # 玉米叶密集，降低冲突半径
        'confidence_threshold': 0.3,   # 降低置信度门槛，减少误分"低质量"
        'min_instance_area':    10,

        'max_frames':       max_frames_val,
        'use_motion_prior': True,

        # ── 深度计算：取最近 top_k 像素的中位数 ─────────────────
        'use_top_k_depths':  True,
        'top_k_min_depths':  top_k,    # 300像素，比原版2000小得多

        'colormap': 'tab20',

        'pixel_depth_save': {
            'enabled':             False,
            'frames_to_save':      [],
            'depth_decimals':      3,
            'min_depth':           0.0,
            'max_depth':           max_depth,
            'include_coordinates': True,
            'separate_files':      True,
            'save_visualization':  True,
        },

        'id_font_scale':    0.4,
        'depth_font_scale': 0.3,
        'text_color':       (255, 255, 255),
        'unmatched_color':  (128, 128, 128),
        'first_frame':      None,
    }
    return config


def process_folders(folder_paths, output_base_dir=None,
                    max_depth_override=None, scene_hard_cap=30.0):
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    print("\n" + "="*80)
    print(f"🚀 全帧匹配 + Tracking 证据  ({len(folder_paths)} 个文件夹)")
    print("="*80)

    results = []
    for fp in folder_paths:
        fp = str(fp)
        print(f"\n{'='*60}\n📁 {Path(fp).name}\n{'='*60}")
        try:
            config = build_config(
                fp, output_base_dir,
                max_depth=max_depth_override,
                scene_hard_cap=scene_hard_cap,
            )
            if config:
                matcher = OptimizedCenterPoint3DMatcher(config)
                success = matcher.run_matching()
                if success:
                    matcher.visualize_results()
                results.append({'folder': fp, 'success': success,
                                 'output_dir': config['output_dir']})
            else:
                results.append({'folder': fp, 'success': False,
                                 'error': '路径不完整'})
        except Exception as e:
            import traceback; traceback.print_exc()
            results.append({'folder': fp, 'success': False, 'error': str(e)})

    print("\n" + "="*80)
    print("📊 汇总")
    print("="*80)
    for r in results:
        name = Path(r['folder']).name
        if r['success']:
            print(f"✅ {name} → {r['output_dir']}")
        else:
            print(f"❌ {name}: {r.get('error','')}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='3D匹配（全帧匹配 + Tracking 证据 + 自动深度检测）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 全帧匹配
  python run_matchingljp0326.py /path/to/scene

  # 多场景批量处理
  python run_matchingljp0326.py /path/f1 /path/f2

  # 手动指定 max_depth（不自动检测）
  python run_matchingljp0326.py /path/to/scene --max_depth 80.0
        """
    )
    parser.add_argument('folders',       nargs='+')
    parser.add_argument('--max_depth',   type=float, default=None,
                        help='手动指定 max_depth（米），不指定则自动检测')
    parser.add_argument('--scene_hard_cap', type=float, default=30.0,
                        help='场景深度硬性上限（米），默认30m，玉米场景足够')
    parser.add_argument('--output', '-o', default=None)
    args = parser.parse_args()

    results = process_folders(
        args.folders,
        output_base_dir=args.output,
        max_depth_override=args.max_depth,
        scene_hard_cap=args.scene_hard_cap,
    )
    ok = sum(1 for r in results if r['success'])
    print(f"\n🎉 完成！成功: {ok}/{len(results)}")


if __name__ == "__main__":
    main()
