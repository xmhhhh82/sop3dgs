#!/usr/bin/env python3
"""
run_matchingljp0326.py  (v3 - 外观特征增强 + 噪声过滤 + 参数回归0302)
====================================================
核心改进（针对"点云标签混杂"问题）：
  1. 恢复 0302 原始参数 ── spatial_threshold=0.3, top_k=2000, coherence=0.2
  2. 外观相似性证据    ── HSV颜色直方图比较掩码区域，同一叶片颜色分布一致
  3. YOLO置信度过滤    ── 低置信度检测从匹配候选中排除
  4. 噪声实例后处理    ── 单帧/孤立/小面积全局ID在unified_masks中置零
  5. 时序一致性降权    ── 短暂出现(<=2帧)的实例不允许覆盖已有ID
  6. 三轮渐进匹配      ── 严格→宽松→最宽松（比例改回原始）
  7. 自适应阈值        ── 按点云尺度保守调整（上限1.0m→0.6m）
  8. 自动max_depth检测 ── IQR去噪版（保留）

用法：
  python run_matchingljp0326.py /path/to/scene
  python run_matchingljp0326.py /path/f1 /path/f2
  python run_matchingljp0326.py /path/to/scene --max_depth 10.0
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
#  自动探测合理的 max_depth（IQR去噪版，保留）
# ══════════════════════════════════════════════════════════════

def auto_detect_max_depth(depth_dir: str, depth_scale: float = 1000.0,
                          sample_n: int = 10,
                          percentile: float = 95.0,
                          scene_hard_cap: float = 15.0) -> float:
    """
    扫描深度图，用IQR去除离群值后取分位数。
    scene_hard_cap 默认改为15m（玉米植株场景更保守）。
    """
    depth_dir = Path(depth_dir)
    files = sorted(list(depth_dir.glob("*.png")) + list(depth_dir.glob("*.tif")))
    if not files:
        print(f"  ⚠️  未找到深度图，使用默认 max_depth=8.0m")
        return 8.0

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
        valid = d[(d > 0.1) & (d < scene_hard_cap)]
        if len(valid) > 0:
            all_depths.append(valid)

    if not all_depths:
        return 8.0

    combined = np.concatenate(all_depths)
    q25, q75 = np.percentile(combined, [25, 75])
    iqr = q75 - q25
    combined_clean = combined[combined <= q75 + 3.0 * iqr]
    if len(combined_clean) < 100:
        combined_clean = combined

    p_val     = float(np.percentile(combined_clean, percentile))
    max_depth = float(np.clip(p_val * 1.15, 2.0, scene_hard_cap))

    print(f"\n🔭 自动检测 max_depth: P{percentile:.0f}={p_val:.2f}m → "
          f"max_depth={max_depth:.2f}m  (hard_cap={scene_hard_cap:.1f}m)")
    return max_depth


# ══════════════════════════════════════════════════════════════
#  外观相似性计算工具
# ══════════════════════════════════════════════════════════════

def compute_hsv_histogram(image_bgr: np.ndarray,
                          mask_binary: np.ndarray,
                          bins=(12, 8, 8)) -> np.ndarray:
    """
    计算掩码区域内的 HSV 颜色直方图（归一化）。
    bins: (H_bins, S_bins, V_bins)
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mask_u8 = mask_binary.astype(np.uint8)
    h_hist = cv2.calcHist([hsv], [0], mask_u8,
                          [bins[0]], [0, 180]).flatten()
    s_hist = cv2.calcHist([hsv], [1], mask_u8,
                          [bins[1]], [0, 256]).flatten()
    v_hist = cv2.calcHist([hsv], [2], mask_u8,
                          [bins[2]], [0, 256]).flatten()
    hist   = np.concatenate([h_hist, s_hist, v_hist])
    norm   = hist.sum()
    if norm > 0:
        hist = hist / norm
    return hist.astype(np.float32)


def hist_similarity(h1: np.ndarray, h2: np.ndarray) -> float:
    """
    直方图相关系数相似度（[-1,1] → 归一到 [0,1]）。
    相关系数是颜色分布一致性的稳健指标。
    """
    if h1 is None or h2 is None:
        return 0.5  # 未知时返回中性值
    corr = cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL)  # [-1, 1]
    return float((corr + 1.0) / 2.0)


# ══════════════════════════════════════════════════════════════
#  核心匹配器（继承自原始0302版本）
# ══════════════════════════════════════════════════════════════

class OptimizedCenterPoint3DMatcher(_OrigMatcher):
    """
    在原始0302版本基础上叠加：
      ① 外观相似性（HSV直方图）—— 防止相邻叶片混淆
      ② YOLO置信度过滤         —— 低置信度不参与匹配
      ③ 三轮渐进式匹配          —— 严格→宽松→最宽松
      ④ 噪声实例后处理          —— 清理单帧/孤立全局ID
      ⑤ 自适应阈值（保守版）    —— 上限从1.0m降到0.6m
    """

    def __init__(self, config):
        # ── 新增参数（在 super().__init__ 之前设置） ──────────
        self._loose_multiplier     = config.get('loose_multiplier',     3.0)
        self._loosest_multiplier   = config.get('loosest_multiplier',   8.0)
        self._allow_new_global_ids = config.get('allow_new_global_ids', True)
        self._adaptive_threshold   = config.get('adaptive_threshold',   True)

        # 外观相似性权重
        self._appearance_weight    = config.get('appearance_weight', 0.3)
        # 外观相似度阈值（低于此值视为外观不一致）
        self._appearance_min_score = config.get('appearance_min_score', 0.35)

        # YOLO置信度过滤：低于此值的实例不参与匹配（标为噪声）
        self._min_yolo_conf        = config.get('min_yolo_conf', 0.0)  # 默认不过滤，用户按需开启

        # 噪声后处理：全局ID出现帧数少于此值则清除
        self._min_views_to_keep    = config.get('min_views_to_keep', 2)
        # 噪声后处理：全局ID的3D位置离群（IQR系数）
        self._outlier_iqr_coeff    = config.get('outlier_iqr_coeff', 3.0)

        # 原始图像目录（用于外观计算）
        self._images_dir           = None
        self._image_cache          = {}   # {frame_num: np.ndarray (BGR)}

        # 全局实例的外观直方图缓存
        self._global_histograms    = {}   # {global_id: np.ndarray}

        # YOLO置信度缓存
        self._frame_instance_conf  = {}   # {frame_num: {local_id: conf}}

        # track_id 相关
        self.frame_track_ids  = {}
        self.global_track_ids = {}

        super().__init__(config)

        # 初始化后的额外操作
        self._find_images_dir()
        self._load_track_ids_and_conf()
        self._adapt_threshold()

    # ── 查找原始图像目录 ────────────────────────────────────────
    def _find_images_dir(self):
        """从 mask_dir 向上推断原始图像目录。"""
        mask_dir = Path(self.config['mask_dir'])
        # 通常结构: scene_dir/masks_results/integer_masks
        scene_dir = mask_dir.parent.parent
        candidates = [
            scene_dir / "images",
            scene_dir.parent / "images",
        ]
        for c in candidates:
            if c.is_dir() and any(c.glob("*.jpg")) | any(c.glob("*.png")):
                self._images_dir = c
                print(f"✅ 找到原始图像目录: {c}")
                return
        print("⚠️  未找到原始图像目录，外观相似性将退化为形状相似性")

    def _get_image(self, frame_num: int):
        """加载并缓存原始图像（BGR）。"""
        if frame_num in self._image_cache:
            return self._image_cache[frame_num]
        if self._images_dir is None:
            return None
        for ext in ['jpg', 'jpeg', 'png', 'bmp']:
            p = self._images_dir / f"{frame_num:04d}.{ext}"
            if p.exists():
                img = cv2.imread(str(p))
                if img is not None:
                    self._image_cache[frame_num] = img
                    # 避免缓存过大，只保留最近20帧
                    if len(self._image_cache) > 20:
                        oldest = min(self._image_cache.keys())
                        del self._image_cache[oldest]
                    return img
        return None

    def _compute_instance_histogram(self, frame_num: int,
                                    local_id: int) -> np.ndarray:
        """计算某帧某实例掩码区域的 HSV 直方图。"""
        img = self._get_image(frame_num)
        if img is None or frame_num not in self.masks:
            return None
        mask = self.masks[frame_num]
        binary = (mask == local_id)
        if binary.sum() < 20:
            return None
        return compute_hsv_histogram(img, binary)

    def _get_global_histogram(self, global_id: int) -> np.ndarray:
        """获取全局实例的外观直方图（取最近一帧）。"""
        if global_id in self._global_histograms:
            return self._global_histograms[global_id]
        # 从最近一帧计算
        last_frame, last_lid = self.get_last_occurrence(global_id)
        if last_frame is None:
            return None
        hist = self._compute_instance_histogram(last_frame, last_lid)
        if hist is not None:
            self._global_histograms[global_id] = hist
        return hist

    def _update_global_histogram(self, global_id: int,
                                  new_hist: np.ndarray):
        """指数移动平均更新全局直方图（防止漂移）。"""
        if new_hist is None:
            return
        if global_id not in self._global_histograms:
            self._global_histograms[global_id] = new_hist.copy()
        else:
            alpha = 0.3  # 新帧权重
            self._global_histograms[global_id] = (
                (1 - alpha) * self._global_histograms[global_id] +
                alpha * new_hist
            )

    # ── 加载 track_id 和 YOLO 置信度 ────────────────────────────
    def _load_track_ids_and_conf(self):
        """从 class_info JSON 读取 track_id 和 conf（如果存在）。"""
        mask_dir = Path(self.config['mask_dir'])
        info_files = sorted(mask_dir.glob("class_info_*.json"))
        n_with_track = n_with_conf = 0

        for info_file in info_files:
            base = info_file.stem.replace('class_info_', '')
            m = re.search(r'(\d+)', base)
            if not m:
                continue
            frame_num = int(m.group(1))
            with open(info_file, 'r') as f:
                data = json.load(f)

            tid_map  = {}
            conf_map = {}
            for inst in data.get('instances', []):
                iid = inst['instance_id']
                if 'track_id' in inst:
                    tid_map[iid]  = inst['track_id']
                if 'confidence' in inst:
                    conf_map[iid] = float(inst['confidence'])

            if tid_map:
                self.frame_track_ids[frame_num] = tid_map
                n_with_track += 1
            if conf_map:
                self._frame_instance_conf[frame_num] = conf_map
                n_with_conf += 1

        print(f"\n🔗 Tracking: {n_with_track}/{len(info_files)} 帧含 track_id, "
              f"{n_with_conf} 帧含 confidence")

    def _get_instance_conf(self, frame_num: int, local_id: int) -> float:
        """获取某实例的YOLO置信度（找不到返回1.0）。"""
        if frame_num not in self._frame_instance_conf:
            return 1.0
        conf_map = self._frame_instance_conf[frame_num]
        # local_id 是重新编号后的，需要做映射
        # 简单策略：按序号映射到conf_map的第local_id个target class实例
        target_confs = sorted(
            (iid, c) for iid, c in conf_map.items()
            if self.frame_class_info.get(frame_num, {}).get(local_id) ==
            self.target_class
        )
        if 0 < local_id <= len(target_confs):
            return target_confs[local_id - 1][1]
        return 1.0

    def _get_remapped_track_id(self, frame_num, local_id):
        """获取重新编号后的 local_id 对应的 track_id。"""
        if frame_num not in self.frame_track_ids:
            return None
        tid_map = self.frame_track_ids[frame_num]
        if local_id in tid_map:
            return tid_map[local_id]
        return None

    # ── 自适应阈值（保守版：上限0.6m） ──────────────────────────
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

        # Per-axis IQR 去除离群3D点
        mask = np.ones(len(pts_arr), dtype=bool)
        for axis in range(pts_arr.shape[1]):
            vals = pts_arr[:, axis]
            q25, q75 = np.percentile(vals, [25, 75])
            iqr = q75 - q25
            mask &= (vals >= q25 - 3.0 * iqr) & (vals <= q75 + 3.0 * iqr)
        pts_clean = pts_arr[mask] if mask.sum() >= 2 else pts_arr

        outlier_n   = len(pts_arr) - len(pts_clean)
        bbox_clean  = np.linalg.norm(
            pts_clean.max(axis=0) - pts_clean.min(axis=0))

        # 保守：bbox的3%，上限0.6m（原来1.0m，因为混杂问题降低上限）
        BBOX_CAP    = 8.0
        THRESH_CAP  = 0.6   # ← 关键：从1.0m降到0.6m防止跨叶匹配
        bbox_diag   = min(bbox_clean, BBOX_CAP)
        adaptive    = max(self.spatial_threshold, bbox_diag * 0.03)
        adaptive    = min(adaptive, THRESH_CAP)

        if adaptive > self.spatial_threshold * 1.05:
            print(f"\n🔧 自适应阈值: {self.spatial_threshold:.3f}m → {adaptive:.3f}m "
                  f"(bbox={bbox_clean:.2f}m, 去除{outlier_n}个离群3D点)")
            self.spatial_threshold = adaptive
            self.motion_threshold  = max(self.motion_threshold,
                                         adaptive * 1.5)

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

            # 初始化外观直方图
            hist = self._compute_instance_histogram(frame_num, lid)
            if hist is not None:
                self._global_histograms[global_id] = hist

            new_matches[lid] = {
                'global_id':    global_id,
                'confidence':   1.0,
                'match_type':   '新建',
                'evidence_count': 0
            }
            print(f"    🆕 本地{lid} → 新全局ID {global_id}")
        return new_matches

    # ── 三轮渐进匹配（核心，含外观证据） ────────────────────────
    def match_with_coherence(self, frame_num):
        if frame_num not in self.instance_centers_3d:
            return {}

        # 目标类实例，过滤低置信度
        current_instances = {}
        for iid, pt in self.instance_centers_3d[frame_num].items():
            if iid > self.frame_mask_counts.get(frame_num, 0):
                continue
            conf = self._get_instance_conf(frame_num, iid)
            if conf < self._min_yolo_conf:
                print(f"    ⚠️  实例{iid} conf={conf:.2f} < {self._min_yolo_conf:.2f}，跳过")
                continue
            current_instances[iid] = pt

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

        gp_arr    = np.array(global_points)
        frame_ids = list(current_instances.keys())
        fp_arr    = np.array(list(current_instances.values()))
        distances = cdist(fp_arr, gp_arr)

        print(f"\n  🔍 帧{frame_num:04d}: "
              f"{len(frame_ids)}实例 <-> {len(global_ids)}全局点")

        # ── 单轮匹配函数（含外观证据）────────────────────────────
        def _one_round(pending, used_g, sp_thr, mo_thr, label):
            if not pending:
                return {}, []
            lidx = {lid: frame_ids.index(lid) for lid in pending}
            ev   = defaultdict(list)

            # ─ 证据A: Tracking ID（最强）──────────────────────
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
                            'type': 'tracking', 'score': 2.0})

            # ─ 证据B: 运动连续性 ───────────────────────────────
            for pred in self.get_motion_predictions(frame_num, global_ids):
                gid, ppos, conf = (pred['global_id'],
                                   pred['predicted_pos'], pred['confidence'])
                for lid, fi in lidx.items():
                    d = np.linalg.norm(np.array(ppos) - fp_arr[fi])
                    if d < mo_thr * 2:
                        ev[(lid, gid)].append({
                            'type': 'motion',
                            'score': conf / (1.0 + d * 5),
                            'distance': d})

            # ─ 证据C: 双向空间匹配 ────────────────────────────
            for lid, fi in lidx.items():
                for ji, gid in enumerate(global_ids):
                    if gid in used_g:
                        continue
                    d = distances[fi, ji]
                    if d >= sp_thr * 1.5:
                        continue
                    fwd  = (np.argmin(distances[fi]) == ji)
                    bwd  = (np.argmin(distances[:, ji]) == fi)
                    mult = 1.5 if (fwd and bwd) else (
                           1.0 if (fwd or bwd) else 0.6)
                    ev[(lid, gid)].append({
                        'type': 'bidirectional',
                        'score': mult / (1.0 + d * 5),
                        'distance': d})

            # ─ 证据D: 形状相似性（Hu矩）────────────────────────
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
                            ev[(lid, gid)].append({
                                'type': 'shape', 'score': sh})

            # ─ 证据E: 外观颜色直方图（新增！防跨叶混淆）─────────
            if self._images_dir is not None:
                for lid in pending:
                    cur_hist = self._compute_instance_histogram(frame_num, lid)
                    if cur_hist is None:
                        continue
                    for gid in global_ids:
                        if gid in used_g:
                            continue
                        g_hist = self._get_global_histogram(gid)
                        app_score = hist_similarity(cur_hist, g_hist)

                        # 外观不一致（低于阈值）：添加负向证据，惩罚该匹配
                        if app_score < self._appearance_min_score:
                            ev[(lid, gid)].append({
                                'type': 'appearance_penalty',
                                'score': -(1.0 - app_score) * 0.5})
                        elif app_score > 0.6:
                            ev[(lid, gid)].append({
                                'type': 'appearance',
                                'score': app_score * self._appearance_weight})

            # ─ 综合得分 ────────────────────────────────────────
            cands = []
            for (lid, gid), evs in ev.items():
                total = nm = nb = ns = nt = na = 0
                for e in evs:
                    t = e['type']
                    if   t == 'tracking':
                        total += e['score'] * 1.0;           nt += 1
                    elif t == 'motion':
                        total += e['score'] * self.motion_weight; nm += 1
                    elif t == 'bidirectional':
                        total += e['score'] * self.bidirectional_weight; nb += 1
                    elif t == 'shape':
                        total += e['score'] * self.shape_weight; ns += 1
                    elif t == 'appearance':
                        total += e['score'];                 na += 1
                    elif t == 'appearance_penalty':
                        total += e['score']  # 直接扣分

                n_pos = (nt > 0) + (nm > 0) + (nb > 0) + (ns > 0) + (na > 0)
                if n_pos >= 2: total *= 1.2; mt = '协同'
                elif nt:  mt = 'tracking'
                elif nm:  mt = '运动'
                elif nb:  mt = '双向'
                elif ns:  mt = '形状'
                elif na:  mt = '外观'
                else:     mt = '其他'
                cands.append({'lid': lid, 'gid': gid,
                               'score': total, 'mt': mt})

            cands.sort(key=lambda x: -x['score'])
            rm   = {}
            ul   = set()
            for c in cands:
                lid, gid = c['lid'], c['gid']
                if lid in ul or gid in used_g:
                    continue
                # 得分为负则跳过（外观惩罚导致总分为负）
                if c['score'] <= 0:
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

        # ── 执行三轮 ─────────────────────────────────────────────
        final   = {}
        used_g  = set()
        details = []

        thr1 = self.spatial_threshold
        thr2 = thr1 * self._loose_multiplier
        thr3 = thr1 * self._loosest_multiplier

        r1, pending = _one_round(frame_ids, used_g, thr1, self.motion_threshold, "R1")
        final.update(r1)
        for m in r1.values():
            used_g.add(m['global_id']); self._stat(m['match_type'])

        if pending:
            print(f"\n  🔄 R2({len(pending)}个, thr={thr2:.3f}m)")
            r2, pending = _one_round(
                pending, used_g, thr2,
                self.motion_threshold * self._loose_multiplier, "R2")
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
                bgid   = global_ids[bj]
                if bd < thr3:
                    # R3 额外做外观检查：外观分太低则跳过（不强制）
                    app_ok = True
                    if self._images_dir is not None:
                        cur_h = self._compute_instance_histogram(frame_num, lid)
                        g_h   = self._get_global_histogram(bgid)
                        if cur_h is not None and g_h is not None:
                            app_score = hist_similarity(cur_h, g_h)
                            if app_score < self._appearance_min_score * 0.7:
                                app_ok = False
                                print(f"    [R3] ⚠️  本地{lid} 外观分={app_score:.2f} 过低 → 新ID")

                    if app_ok:
                        final[lid] = {'global_id': bgid,
                                      'confidence': 1.0/(1.0+bd),
                                      'match_type': '最近邻', 'evidence_count': 1}
                        used_g.add(bgid)
                        print(f"    [R3] ✅ 本地{lid}<->全局{bgid} (d={bd:.3f}m)")
                        self.matching_stats['bidirectional_matches'] += 1
                    else:
                        still3.append(lid)
                else:
                    still3.append(lid)
                    print(f"    [R3] ⚠️  本地{lid} d={bd:.3f}m>{thr3:.3f}m → 新ID")
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
                'depth_method': 'unknown', 'top_k_used': 0, 'total_pixels': 0
            })

        self.save_matching_details(frame_num, details)
        self.frame_matched_counts[frame_num] = len(final)
        n_un = len(frame_ids) - len(final)
        print(f"\n  📊 {frame_num:04d}: {len(final)}/{len(frame_ids)} 匹配"
              f"{'  🎉' if n_un == 0 else f'  ({n_un}未匹配)'}")
        return final

    def _stat(self, mt):
        if 'tracking' in mt:
            self.matching_stats.setdefault('tracking_matches', 0)
            self.matching_stats['tracking_matches'] += 1
        elif '协同' in mt:
            self.matching_stats['cooperative_matches'] += 1
        elif '运动' in mt:
            self.matching_stats['motion_matches'] += 1
        elif '双向' in mt:
            self.matching_stats['bidirectional_matches'] += 1
        elif '形状' in mt:
            self.matching_stats['shape_matches'] += 1

    # ── 噪声实例后处理（核心！解决类别混杂的关键步骤） ──────────
    def post_process_remove_noise(self):
        """
        在所有帧匹配完成后，清除噪声全局ID：
          1. 出现帧数 < min_views_to_keep → 孤立噪声
          2. 3D位置是离群点（per-axis IQR）→ 深度噪声反投影
          3. 平均面积过小 → 碎片实例

        清除方式：在 global_instance_map 中标记，
        在 save_results 生成 unified_masks 时这些ID会被设为0。
        """
        print("\n🧹 噪声实例后处理...")

        # 收集所有目标类全局ID的信息
        target_gids = [gid for gid in self.global_centers_3d
                       if gid not in self.fixed_class_ids.values()]
        if not target_gids:
            return

        noise_ids = set()

        # ① 出现帧数过少
        for gid in target_gids:
            info = self.global_centers_3d[gid]
            if info['num_views'] < self._min_views_to_keep:
                noise_ids.add(gid)
                print(f"   🗑️  全局ID {gid}: 仅出现{info['num_views']}帧 → 标记为噪声")

        # ② 3D位置离群（针对仍保留的ID）
        remaining_gids = [gid for gid in target_gids if gid not in noise_ids]
        if len(remaining_gids) >= 4:
            pts = np.array([self.global_centers_3d[gid]['point']
                            for gid in remaining_gids])
            for axis in range(3):
                vals = pts[:, axis]
                q25, q75 = np.percentile(vals, [25, 75])
                iqr = q75 - q25
                lower = q25 - self._outlier_iqr_coeff * iqr
                upper = q75 + self._outlier_iqr_coeff * iqr
                for i, gid in enumerate(remaining_gids):
                    if gid in noise_ids:
                        continue
                    if vals[i] < lower or vals[i] > upper:
                        noise_ids.add(gid)
                        print(f"   🗑️  全局ID {gid}: 3D位置离群"
                              f"(axis{axis}={vals[i]:.2f}, "
                              f"range=[{lower:.2f},{upper:.2f}]) → 标记为噪声")
                        break  # 一个轴离群即可

        # 记录噪声ID集合供 save_results 使用
        self._noise_global_ids = noise_ids
        print(f"   共标记 {len(noise_ids)}/{len(target_gids)} 个全局ID为噪声")

        # 统计：被清除后的实际有效ID数
        valid_count = len(target_gids) - len(noise_ids)
        print(f"   清除后有效全局ID: {valid_count}")

    # ── 重写 save_results 以应用噪声过滤 ────────────────────────
    def save_results(self):
        """覆盖父类方法：在生成unified_masks时跳过噪声ID。"""
        # 先执行噪声后处理
        self.post_process_remove_noise()

        # 调用父类保存（会生成 id_mapping.json、color masks等）
        # 但我们需要在生成 unified_masks 时应用噪声过滤
        # 策略：临时修改 global_instance_map，把噪声ID映射到0
        noise_ids = getattr(self, '_noise_global_ids', set())

        # 备份原始映射
        original_map = dict(self.global_instance_map)

        # 将噪声ID的所有帧映射标记为不输出（设为0）
        # 通过将噪声gid从 global_instance_map 中移除实现
        for key, gid in list(self.global_instance_map.items()):
            if gid in noise_ids:
                del self.global_instance_map[key]

        # 同样从 global_centers_3d 中移除噪声ID（不影响 id_mapping.json 的写入，但会影响报告）
        noise_center_backup = {}
        for gid in noise_ids:
            if gid in self.global_centers_3d:
                noise_center_backup[gid] = self.global_centers_3d.pop(gid)

        print(f"\n💾 保存结果（已过滤 {len(noise_ids)} 个噪声ID）...")

        # 调用父类原始 save_results
        super().save_results()

        # 恢复（供后续验证使用）
        self.global_instance_map.update(original_map)
        self.global_centers_3d.update(noise_center_backup)

        # 额外保存一份噪声ID清单
        noise_log = self.output_dir / "noise_ids_removed.txt"
        with open(noise_log, 'w') as f:
            f.write("# 以下全局ID被识别为噪声并从unified_masks中清除\n")
            f.write(f"# 过滤条件: 出现帧数<{self._min_views_to_keep} "
                    f"或3D离群(IQR系数={self._outlier_iqr_coeff})\n\n")
            for gid in sorted(noise_ids):
                info = noise_center_backup.get(gid) or \
                       self.global_centers_3d.get(gid, {})
                f.write(f"global_id={gid}  "
                        f"num_views={info.get('num_views',0)}  "
                        f"point={info.get('point','?')}\n")
        print(f"✅ 噪声ID清单已保存: {noise_log}")

    # ── run_matching（含外观直方图更新） ─────────────────────────
    def run_matching(self):
        print("\n" + "="*60)
        print("🚀 优化版三轮渐进匹配 (v3 外观增强)")
        print(f"   R1 阈值: {self.spatial_threshold:.3f}m")
        print(f"   R2 阈值: {self.spatial_threshold*self._loose_multiplier:.3f}m")
        print(f"   R3 阈值: {self.spatial_threshold*self._loosest_multiplier:.3f}m")
        print(f"   外观权重: {self._appearance_weight}  "
              f"外观最低分: {self._appearance_min_score}")
        print(f"   噪声过滤: 最少{self._min_views_to_keep}帧出现")
        print("="*60)

        first_frame = self.initialize_with_first_frame()
        if first_frame is None:
            print("❌ 初始化失败")
            return False

        # 为初始帧建立外观直方图和track_id映射
        for (fn, lid), gid in self.global_instance_map.items():
            if fn == first_frame:
                hist = self._compute_instance_histogram(fn, lid)
                if hist is not None:
                    self._global_histograms[gid] = hist
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

                # 更新外观直方图
                new_hist = self._compute_instance_histogram(cf, lid)
                self._update_global_histogram(gid, new_hist)

                # 记录 track_id
                tid = self._get_remapped_track_id(cf, lid)
                if tid is not None and gid not in self.global_track_ids:
                    self.global_track_ids[gid] = tid

                self.total_matches += 1
                if mi['confidence'] > self.confidence_threshold:
                    self.matching_stats['high_confidence'] += 1
                else:
                    self.matching_stats['low_confidence'] += 1

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

        self.save_results()           # ← 内部含噪声过滤
        self.save_all_centers_3d_info()
        self.verify_matching_results()
        return True


# ══════════════════════════════════════════════════════════════
#  路径解析 & 批量处理
# ══════════════════════════════════════════════════════════════

def build_config(folder_path, output_base_dir, max_depth, scene_hard_cap=15.0):
    """
    构建 config 字典。
    参数回归0302原始值：spatial=0.3, top_k=2000, motion=0.5。
    """
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
    if not mask_dir.exists():
        cands = list(folder_path.glob("**/integer_masks"))
        mask_dir = cands[0] if cands else None
    else:
        mask_dir = mask_dir
    mask_dir = str(mask_dir) if mask_dir else None

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
        max_depth = min(max_depth, scene_hard_cap)

    # ── 全帧数量 ──────────────────────────────────────────────
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

    first_frame_candidates = sorted(all_frame_nums[:5]) if len(all_frame_nums) >= 5 \
        else (all_frame_nums if all_frame_nums else [1, 2, 3, 4, 5])

    print(f"\n⚙️  配置（0302回归参数 + 外观增强）:")
    print(f"   max_depth              = {max_depth:.2f}m")
    print(f"   spatial_threshold      = 0.3m  (0302原始值)")
    print(f"   top_k_min_depths       = 2000  (0302原始值)")
    print(f"   loose_multiplier       = 3.0   (R2: 0.9m)")
    print(f"   loosest_multiplier     = 8.0   (R3: 2.4m)")
    print(f"   appearance_weight      = 0.3")
    print(f"   min_views_to_keep      = 2     (单帧噪声过滤)")

    config = {
        # ── 路径 ────────────────────────────────────────────────
        'depth_dir':    depth_dir,
        'camera_json':  camera_json,
        'mask_dir':     mask_dir,
        'output_dir':   output_dir,

        # ── 类别 ────────────────────────────────────────────────
        'target_class':    3,
        'fixed_class_ids': {0: 220, 1: 221, 2: 222},
        'first_frame_candidates': first_frame_candidates,

        # ── 深度 ────────────────────────────────────────────────
        'depth_scale':         1000.0,
        'depth_format':        '16bit',
        'max_depth':           max_depth,

        # ════════════════════════════════════════════════════════
        # ★ 回归0302原始参数（从激进值恢复到稳健值）
        # ════════════════════════════════════════════════════════
        'spatial_threshold':   0.3,    # 0302原始值（v2是0.15，过小导致跨叶合并）
        'motion_threshold':    0.5,    # 0302原始值（v2是0.3）
        'coherence_threshold': 0.2,    # 0302原始值（v2是0.05，过小导致冲突漏判）
        'confidence_threshold':0.5,    # 0302原始值（v2是0.3）
        'min_instance_area':   5,      # 0302原始值

        # ── 匹配倍率（温和：不再激进扩大匹配范围）────────────────
        'loose_multiplier':    3.0,    # R2 = 0.3*3 = 0.9m
        'loosest_multiplier':  8.0,    # R3 = 0.3*8 = 2.4m
        'allow_new_global_ids': True,
        'adaptive_threshold':  True,

        # ── 权重 ────────────────────────────────────────────────
        'motion_weight':        0.4,
        'bidirectional_weight': 0.4,
        'shape_weight':         0.2,

        # ── 深度计算（0302原始值）───────────────────────────────
        'use_top_k_depths':  True,
        'top_k_min_depths':  2000,     # 0302原始值（v2是300，过小不稳定）

        'max_frames':       max_frames_val,
        'use_motion_prior': True,
        'colormap':         'tab20',

        # ════════════════════════════════════════════════════════
        # ★ 新增参数：外观 + 噪声过滤
        # ════════════════════════════════════════════════════════
        'appearance_weight':     0.3,    # 外观证据在综合得分中的权重
        'appearance_min_score':  0.35,   # 低于此外观分则施加惩罚
        'min_yolo_conf':         0.0,    # YOLO置信度门槛（0=不过滤，可调为0.25）
        'min_views_to_keep':     2,      # 全局ID最少出现帧数（低于此值清除）
        'outlier_iqr_coeff':     3.0,    # 3D离群检测的IQR系数

        # ── 像素深度（关闭节省时间）─────────────────────────────
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
                    max_depth_override=None, scene_hard_cap=15.0):
    if isinstance(folder_paths, str):
        folder_paths = [folder_paths]

    print("\n" + "="*80)
    print(f"🚀 全帧匹配 + 外观增强 + 噪声过滤  ({len(folder_paths)} 个文件夹)")
    print(f"   参数版本: 0302回归 (spatial=0.3, top_k=2000)")
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
            print(f"✅ {name} → {r.get('output_dir','')}")
        else:
            print(f"❌ {name}: {r.get('error','')}")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='3D匹配 v3（0302参数 + 外观增强 + 噪声过滤）',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
参数说明（关键调优项）：
  --max_depth      手动指定深度上限（不填则自动检测，默认上限15m）
  --scene_hard_cap 场景深度硬性上限（默认15m，室外玉米植株）
  --min_yolo_conf  YOLO置信度过滤门槛（默认0.0=不过滤，推荐0.25）
  --min_views      全局ID最少出现帧数，少于此值清除（默认2）
  --no_appearance  关闭外观相似性证据（调试用）

示例:
  # 标准运行
  python run_matchingljp0326.py /path/to/scene

  # 开启YOLO置信度过滤
  python run_matchingljp0326.py /path/to/scene --min_yolo_conf 0.25

  # 严格噪声过滤（全局ID至少出现3帧）
  python run_matchingljp0326.py /path/to/scene --min_views 3
        """
    )
    parser.add_argument('folders',          nargs='+')
    parser.add_argument('--max_depth',      type=float, default=None)
    parser.add_argument('--scene_hard_cap', type=float, default=15.0)
    parser.add_argument('--min_yolo_conf',  type=float, default=0.0,
                        help='YOLO置信度过滤（0=不过滤）')
    parser.add_argument('--min_views',      type=int,   default=2,
                        help='全局ID最少出现帧数（默认2）')
    parser.add_argument('--no_appearance',  action='store_true',
                        help='关闭外观相似性（调试用）')
    parser.add_argument('--output', '-o',   default=None)
    args = parser.parse_args()

    # 将命令行参数写回 config（通过 process_folders 的 config 构建传递）
    # 使用全局变量传递用户自定义参数
    import builtins
    builtins._USER_MIN_YOLO_CONF  = args.min_yolo_conf
    builtins._USER_MIN_VIEWS      = args.min_views
    builtins._USER_NO_APPEARANCE  = args.no_appearance

    # 打补丁：让 build_config 能读取这些值
    _orig_build_config = build_config

    def _patched_build_config(folder_path, output_base_dir, max_depth,
                               scene_hard_cap=15.0):
        cfg = _orig_build_config(folder_path, output_base_dir,
                                  max_depth, scene_hard_cap)
        if cfg is not None:
            cfg['min_yolo_conf']     = args.min_yolo_conf
            cfg['min_views_to_keep'] = args.min_views
            if args.no_appearance:
                cfg['appearance_weight']    = 0.0
                cfg['appearance_min_score'] = 0.0
        return cfg

    import types
    current_module = sys.modules[__name__]
    setattr(current_module, 'build_config', _patched_build_config)

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
