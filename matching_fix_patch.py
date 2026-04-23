"""
matching_fix_patch.py
=====================
针对"距离太远导致未匹配"问题的修复补丁。

修改说明：
  1. match_with_coherence       —— 三轮匹配（严格 → 宽松 → 最宽松）
  2. assign_new_global_ids_for_unmatched —— 对仍未匹配的实例分配新全局 ID
  3. run_matching               —— 在每帧匹配后调用新函数兜底
  4. 自适应阈值：初始化时根据点云尺度自动放大 spatial_threshold

使用方法：
  将本文件中的三个方法替换到
  0302直接3D匹配_xin2_0305_使用位姿信息进行运动预测+双向匹配_0311_不跑全部帧.py
  的 CenterPoint3DMatcher 类中，
  同时把 __init__ 里的参数段替换为下方的"__init__ 参数段"。
"""

# ═══════════════════════════════════════════════════════════════
# 0. __init__ 参数段（替换原来的 spatial_threshold / motion_threshold 等赋值行）
# ═══════════════════════════════════════════════════════════════
INIT_PATCH = """
        # ---------- 阈值配置（支持自适应放大） ----------
        self.spatial_threshold      = config.get('spatial_threshold', 0.3)
        self.motion_threshold       = config.get('motion_threshold', 0.5)
        # 第二轮（宽松）倍率
        self.loose_multiplier       = config.get('loose_multiplier', 3.0)
        # 第三轮（最宽松）倍率
        self.loosest_multiplier     = config.get('loosest_multiplier', 8.0)
        # 是否允许对未匹配实例创建新全局 ID
        self.allow_new_global_ids   = config.get('allow_new_global_ids', True)
        # 自适应阈值：若点云 bbox 较大则自动放大基础阈值
        self.adaptive_threshold     = config.get('adaptive_threshold', True)
"""

# ═══════════════════════════════════════════════════════════════
# 1. 自适应阈值计算（在 convert_to_3d_centers 之后调用一次）
#    将此方法添加到类中，并在 __init__ 最后调用 self._adapt_threshold()
# ═══════════════════════════════════════════════════════════════
def _adapt_threshold(self):
    """
    根据所有 3D 中心点的包围盒尺度自动放大 spatial_threshold，
    避免固定阈值在不同场景尺度下失效。
    """
    if not self.adaptive_threshold:
        return

    all_pts = []
    for frame_pts in self.instance_centers_3d.values():
        for inst_id, pt in frame_pts.items():
            if inst_id <= self.frame_mask_counts.get(
                    next(iter(self.instance_centers_3d)), 0):
                all_pts.append(pt)

    # 兼容：收集所有目标类点
    all_pts = []
    for frame_num, frame_pts in self.instance_centers_3d.items():
        max_id = self.frame_mask_counts.get(frame_num, 0)
        for inst_id, pt in frame_pts.items():
            if inst_id <= max_id:
                all_pts.append(pt)

    if len(all_pts) < 2:
        return

    import numpy as np
    pts_arr = np.array(all_pts)
    bbox_diag = np.linalg.norm(pts_arr.max(axis=0) - pts_arr.min(axis=0))

    # 经验公式：bbox 对角线的 5% 作为基础阈值下限
    adaptive_base = max(self.spatial_threshold, bbox_diag * 0.05)

    if adaptive_base > self.spatial_threshold * 1.2:
        print(f"\n🔧 自适应阈值: spatial_threshold {self.spatial_threshold:.3f}m "
              f"→ {adaptive_base:.3f}m  (bbox_diag={bbox_diag:.3f}m)")
        self.spatial_threshold = adaptive_base
        self.motion_threshold  = max(self.motion_threshold,
                                     adaptive_base * 1.5)


# ═══════════════════════════════════════════════════════════════
# 2. 三轮匹配核心方法（替换原 match_with_coherence）
# ═══════════════════════════════════════════════════════════════
def match_with_coherence(self, frame_num):
    """
    三轮协同匹配：
      Round-1  严格阈值  (spatial_threshold)
      Round-2  宽松阈值  (spatial_threshold * loose_multiplier)
      Round-3  最宽松阈值 (spatial_threshold * loosest_multiplier)
    每轮对"仍未匹配"的实例再尝试一次，逐步降低要求。
    """
    import numpy as np
    from collections import defaultdict
    from scipy.spatial.distance import cdist

    if frame_num not in self.instance_centers_3d:
        return {}

    # ---------- 目标类实例 ----------
    current_instances = {
        inst_id: point
        for inst_id, point in self.instance_centers_3d[frame_num].items()
        if inst_id <= self.frame_mask_counts.get(frame_num, 0)
    }
    if not current_instances:
        return {}

    # ---------- 全局点（跳过固定类别） ----------
    global_ids, global_points = [], []
    for gid, info in self.global_centers_3d.items():
        if gid in self.fixed_class_ids.values():
            continue
        global_ids.append(gid)
        global_points.append(info['point'])

    if not global_points:
        return {}

    global_points_arr = np.array(global_points)
    frame_ids         = list(current_instances.keys())
    frame_points_arr  = np.array(list(current_instances.values()))

    distances = cdist(frame_points_arr, global_points_arr)

    print(f"\n  🔍 协同匹配: 帧 {frame_num:04d} "
          f"({len(frame_ids)} 目标类实例) <-> 全局点 ({len(global_ids)} 个)")

    # ─── 公共函数：单轮匹配 ────────────────────────────────────
    def _run_one_round(pending_local_ids, used_globals,
                       spatial_thr, motion_thr, label):
        """
        对 pending_local_ids 中的实例进行一轮匹配。
        返回 (round_matches: dict, still_unmatched: list)
        """
        if not pending_local_ids:
            return {}, []

        # 当前轮次的局部索引
        local_idx_map = {lid: frame_ids.index(lid) for lid in pending_local_ids}

        match_evidence = defaultdict(list)

        # 证据1：运动连续性
        motion_predictions = self.get_motion_predictions(frame_num, global_ids)
        for pred in motion_predictions:
            gid          = pred['global_id']
            predicted_pos = pred['predicted_pos']
            confidence    = pred['confidence']
            for lid, fi in local_idx_map.items():
                dist = np.linalg.norm(
                    np.array(predicted_pos) - frame_points_arr[fi])
                if dist < motion_thr * 2:
                    score = confidence * (1.0 / (1.0 + dist * 5))
                    match_evidence[(lid, gid)].append({
                        'type': 'motion', 'score': score,
                        'distance': dist,
                        'details': f'运动({confidence:.2f},{dist:.3f}m)'
                    })

        # 证据2：双向空间匹配
        for lid, fi in local_idx_map.items():
            for ji, gid in enumerate(global_ids):
                if gid in used_globals:
                    continue
                dist = distances[fi, ji]
                if dist < spatial_thr * 1.5:
                    fwd = (np.argmin(distances[fi]) == ji)
                    bwd = (np.argmin(distances[:, ji]) == fi)
                    if fwd and bwd:
                        score = 1.0 / (1.0 + dist * 5) * 1.5
                        sub   = 'perfect_bidirectional'
                    elif fwd:
                        score = 1.0 / (1.0 + dist * 5) * 1.0
                        sub   = 'forward_nn'
                    elif bwd:
                        score = 1.0 / (1.0 + dist * 5) * 1.0
                        sub   = 'backward_nn'
                    else:
                        score = 1.0 / (1.0 + dist * 5) * 0.6
                        sub   = 'distance'
                    match_evidence[(lid, gid)].append({
                        'type': 'bidirectional', 'subtype': sub,
                        'distance': dist, 'score': score,
                        'details': f'{sub}({dist:.3f}m)'
                    })

        # 证据3：形状相似性
        for lid in pending_local_ids:
            cur_mask = (self.masks[frame_num] == lid)
            for gid in global_ids:
                if gid in used_globals:
                    continue
                last_frame, last_lid = self.get_last_occurrence(gid)
                if (last_frame in self.masks and
                        last_lid in np.unique(self.masks[last_frame])):
                    last_mask = (self.masks[last_frame] == last_lid)
                    shape_s   = self.compute_shape_similarity(
                        cur_mask, last_mask)
                    if shape_s > 0.4:   # 宽松一点
                        match_evidence[(lid, gid)].append({
                            'type': 'shape', 'score': shape_s,
                            'details': f'shape({shape_s:.2f})'
                        })

        # 计算综合得分
        candidates = []
        for (lid, gid), evs in match_evidence.items():
            total  = 0
            n_m = n_b = n_s = 0
            for ev in evs:
                if   ev['type'] == 'motion':
                    total += ev['score'] * self.motion_weight;      n_m += 1
                elif ev['type'] == 'bidirectional':
                    total += ev['score'] * self.bidirectional_weight; n_b += 1
                elif ev['type'] == 'shape':
                    total += ev['score'] * self.shape_weight;        n_s += 1
            n_types = (n_m > 0) + (n_b > 0) + (n_s > 0)
            if n_types >= 2:
                total      *= 1.2
                match_type  = '协同'
            elif n_m: match_type = '运动'
            elif n_b: match_type = '双向'
            elif n_s: match_type = '形状'
            else:     match_type = '其他'
            candidates.append({
                'local_id': lid, 'global_id': gid,
                'score': total, 'match_type': match_type
            })

        candidates.sort(key=lambda x: -x['score'])

        round_matches = {}
        used_locals   = set()
        for cand in candidates:
            lid = cand['local_id']
            gid = cand['global_id']
            if lid in used_locals or gid in used_globals:
                continue
            # 冲突检查
            conflict = False
            for sel_lid in round_matches:
                fi1 = frame_ids.index(sel_lid)
                fi2 = local_idx_map[lid]
                if np.linalg.norm(
                        frame_points_arr[fi1] - frame_points_arr[fi2]
                ) < self.coherence_threshold:
                    conflict = True
                    break
            if not conflict:
                round_matches[lid] = {
                    'global_id':  gid,
                    'confidence': cand['score'],
                    'match_type': cand['match_type'],
                    'evidence_count': 1
                }
                used_locals.add(lid)
                used_globals.add(gid)
                print(f"    [{label}] ✅ 本地{lid} <-> 全局{gid} "
                      f"(得分:{cand['score']:.3f}, {cand['match_type']})")

        still_unmatched = [lid for lid in pending_local_ids
                           if lid not in used_locals]
        return round_matches, still_unmatched

    # ─── 三轮匹配 ─────────────────────────────────────────────
    final_matches = {}
    used_globals  = set()
    matching_details = []

    # Round-1：严格阈值
    r1_matches, unmatched_after_r1 = _run_one_round(
        frame_ids, used_globals,
        self.spatial_threshold,
        self.motion_threshold,
        label="R1-严格"
    )
    final_matches.update(r1_matches)
    for m in r1_matches.values():
        used_globals.add(m['global_id'])
    # 统计
    for m in r1_matches.values():
        mt = m['match_type']
        if '协同' in mt:  self.matching_stats['cooperative_matches'] += 1
        elif '运动' in mt: self.matching_stats['motion_matches']      += 1
        elif '双向' in mt: self.matching_stats['bidirectional_matches'] += 1
        elif '形状' in mt: self.matching_stats['shape_matches']        += 1

    # Round-2：宽松阈值
    if unmatched_after_r1:
        thr2 = self.spatial_threshold * self.loose_multiplier
        mth2 = self.motion_threshold  * self.loose_multiplier
        print(f"\n  🔄 Round-2 宽松匹配 ({len(unmatched_after_r1)} 个实例, "
              f"阈值={thr2:.3f}m)")
        r2_matches, unmatched_after_r2 = _run_one_round(
            unmatched_after_r1, used_globals,
            thr2, mth2, label="R2-宽松"
        )
        final_matches.update(r2_matches)
        for m in r2_matches.values():
            used_globals.add(m['global_id'])
        for m in r2_matches.values():
            mt = m['match_type']
            if '协同' in mt:  self.matching_stats['cooperative_matches'] += 1
            elif '运动' in mt: self.matching_stats['motion_matches']      += 1
            elif '双向' in mt: self.matching_stats['bidirectional_matches'] += 1
            elif '形状' in mt: self.matching_stats['shape_matches']        += 1
    else:
        unmatched_after_r2 = []

    # Round-3：最宽松阈值（纯距离最近邻）
    if unmatched_after_r2:
        thr3 = self.spatial_threshold * self.loosest_multiplier
        print(f"\n  🔄 Round-3 最宽松匹配 ({len(unmatched_after_r2)} 个实例, "
              f"阈值={thr3:.3f}m)")

        still_unmatched_r3 = []
        for lid in unmatched_after_r2:
            fi  = frame_ids.index(lid)
            row = distances[fi].copy()
            # 排除已被占用的全局点
            for ji, gid in enumerate(global_ids):
                if gid in used_globals:
                    row[ji] = np.inf
            best_ji  = int(np.argmin(row))
            best_dist = row[best_ji]
            best_gid  = global_ids[best_ji]

            if best_dist < thr3:
                final_matches[lid] = {
                    'global_id':    best_gid,
                    'confidence':   1.0 / (1.0 + best_dist),
                    'match_type':   '最近邻',
                    'evidence_count': 1
                }
                used_globals.add(best_gid)
                print(f"    [R3-最近邻] ✅ 本地{lid} <-> 全局{best_gid} "
                      f"(距离:{best_dist:.3f}m)")
                self.matching_stats['bidirectional_matches'] += 1
            else:
                still_unmatched_r3.append(lid)
                print(f"    [R3] ⚠️  本地{lid} 仍未匹配 "
                      f"(最近全局:{best_gid}, 距离:{best_dist:.3f}m) → 分配新ID")

        unmatched_final = still_unmatched_r3
    else:
        unmatched_final = []

    # ─── 对仍未匹配的实例分配新全局 ID ───────────────────────
    if self.allow_new_global_ids and unmatched_final:
        new_ids = self._assign_new_global_ids(frame_num, unmatched_final)
        final_matches.update(new_ids)
        print(f"\n  🆕 为 {len(new_ids)} 个未匹配实例分配了新全局 ID: "
              f"{[v['global_id'] for v in new_ids.values()]}")

    # ─── 保存匹配详情（只记录最终仍未匹配的） ────────────────
    for lid in frame_ids:
        if lid not in final_matches:
            fi       = frame_ids.index(lid)
            min_dist = float(np.min(distances[fi]))
            min_gid  = global_ids[int(np.argmin(distances[fi]))]
            matching_details.append({
                'frame': frame_num, 'local_id': lid,
                'global_id': None, 'match_status': 'UNMATCHED',
                'reason': f'全部轮次均未匹配 (最近全局:{min_gid}, 距离:{min_dist:.3f}m)',
                'depth_method': self.instance_stats[frame_num].get(lid, {}).get(
                    'depth_calculation_method', 'unknown'),
                'top_k_used': self.instance_stats[frame_num].get(lid, {}).get('top_k_used', 0),
                'total_pixels': self.instance_stats[frame_num].get(lid, {}).get('num_valid_depths', 0)
            })

    self.save_matching_details(frame_num, matching_details)
    self.frame_matched_counts[frame_num] = len(final_matches)

    total_unmatched = len(frame_ids) - len(final_matches)
    print(f"\n  📊 最终: {len(final_matches)}/{len(frame_ids)} 匹配成功"
          f"{'，无未匹配 🎉' if total_unmatched == 0 else f'，{total_unmatched} 个未匹配'}")

    return final_matches


# ═══════════════════════════════════════════════════════════════
# 3. 为未匹配实例分配新全局 ID（新增方法）
# ═══════════════════════════════════════════════════════════════
def _assign_new_global_ids(self, frame_num, unmatched_local_ids):
    """
    对无法与任何现有全局点匹配的实例，创建新的全局点。
    返回 {local_id: {'global_id': ..., 'confidence': 1.0, ...}}
    """
    new_matches = {}

    for lid in unmatched_local_ids:
        if lid not in self.instance_centers_3d.get(frame_num, {}):
            continue
        if lid not in self.instance_stats.get(frame_num, {}):
            continue

        point = self.instance_centers_3d[frame_num][lid]
        stats = self.instance_stats[frame_num][lid]

        global_id = self.global_instance_counter
        self.global_instance_counter += 1

        key = (frame_num, lid)
        self.global_instance_map[key] = global_id

        self.global_centers_3d[global_id] = {
            'point':      point,
            'frames':     [frame_num],
            'instances':  [(frame_num, lid)],
            'distances':  [],
            'num_views':  1,
            'confidence': 1.0,
            'is_fixed':   False,
            'class_id':   self.target_class
        }

        # 保存掩码
        import numpy as np
        if frame_num in self.masks and lid in np.unique(self.masks[frame_num]):
            mask = (self.masks[frame_num] == lid)
            self.global_masks[global_id] = {frame_num: mask}

        self.global_stats[global_id] = {
            'avg_depth': stats.get('depth_median', 0),
            'total_area': stats.get('area', 0),
            'num_views':  1,
            'class_id':   self.target_class
        }

        # 初始化运动轨迹
        self.motion_trajectories[global_id].append((frame_num, point))

        # 更新颜色映射
        max_id = max(self.global_id_colors.keys()) if self.global_id_colors else 0
        if global_id > max_id:
            self.global_id_colors.update(
                self.generate_colormap(global_id))

        new_matches[lid] = {
            'global_id':    global_id,
            'confidence':   1.0,
            'match_type':   '新建',
            'evidence_count': 0
        }

    return new_matches


# ═══════════════════════════════════════════════════════════════
# 4. run_matching 中需要补充的修改（在 update_global_point 调用之后）
#    将原来的 run_matching 里匹配结果处理段替换如下
# ═══════════════════════════════════════════════════════════════
RUN_MATCHING_PATCH = """
            if matches:
                for local_id, match_info in matches.items():
                    global_id = match_info['global_id']
                    distance  = match_info.get('distance', 0.3)
                    point     = np.array(
                        self.instance_centers_3d[current_frame][local_id])

                    key = (current_frame, local_id)
                    # 已由 _assign_new_global_ids 写入的不重复写
                    if key not in self.global_instance_map:
                        self.global_instance_map[key] = global_id

                    # 新建类型的实例已在 _assign_new_global_ids 中完成初始化，
                    # 无需再调 update_global_point
                    if match_info.get('match_type') != '新建':
                        self.update_global_point(
                            global_id, point, current_frame, local_id, distance)

                    self.total_matches += 1
                    if match_info['confidence'] > self.confidence_threshold:
                        self.matching_stats['high_confidence'] += 1
                    else:
                        self.matching_stats['low_confidence'] += 1
"""

# ═══════════════════════════════════════════════════════════════
# 5. 推荐的 config 参数（替换原有的同名字段）
# ═══════════════════════════════════════════════════════════════
RECOMMENDED_CONFIG = {
    'spatial_threshold':    0.5,   # 原 0.3 → 放宽基础阈值
    'motion_threshold':     0.8,   # 原 0.5
    'loose_multiplier':     3.0,   # Round-2 放大倍率（0.5*3=1.5m）
    'loosest_multiplier':   8.0,   # Round-3 放大倍率（0.5*8=4.0m）
    'allow_new_global_ids': True,  # 未匹配实例自动建新 ID
    'adaptive_threshold':   True,  # 根据点云尺度自动调整阈值
    'coherence_threshold':  0.1,   # 原 0.2 → 减小冲突过滤半径
}

# ═══════════════════════════════════════════════════════════════
# 如何集成到原脚本：
#   1. 把 RECOMMENDED_CONFIG 中的键值合并到 main() 的 config 字典
#   2. 在 __init__ 的阈值赋值段（INIT_PATCH 所示位置）添加新参数读取
#   3. 将 _adapt_threshold、match_with_coherence、_assign_new_global_ids
#      三个函数的函数体粘贴为类方法
#   4. 在 __init__ 最后的 self.select_best_first_frame() 前加一行：
#        self._adapt_threshold()
#   5. 在 run_matching 的匹配结果处理段用 RUN_MATCHING_PATCH 替换
# ═══════════════════════════════════════════════════════════════
