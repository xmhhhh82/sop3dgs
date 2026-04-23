
#!/usr/bin/env bash
# =============================================================================
# pipeline.sh — 3D 语义点云全自动处理流水线 (v3 外观增强版)
#
# 核心变化（对应匹配脚本 v3）：
#   1. STEP 5 新增 --min_yolo_conf / --min_views 参数透传
#   2. --scene_hard_cap 默认从30m改为15m（玉米场景更保守）
#   3. 注释中标注参数回归原因（0302原始值）
#   4. 保留所有断点续跑控制
#
# 用法:
#   bash pipeline.sh --video /path/to/video.mp4 [选项]
#   bash pipeline.sh --scene_dir /path/to/scene [选项]
# =============================================================================
set -euo pipefail

# ─────────────────────────── 颜色输出与日志系统 ────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'

log_step() { echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}";
             echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}";
             echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"; }
log_ok()   { echo -e "${GREEN}✅ $1${NC}"; }
log_warn() { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err()  { echo -e "${RED}❌ $1${NC}"; exit 1; }
log_info() { echo -e "${CYAN}ℹ  $1${NC}"; }
log_skip() { echo -e "${YELLOW}⏭  $1 (已满足条件，跳过)${NC}"; }
log_env()  { echo -e "${BOLD}${YELLOW}🐍 conda env: $1${NC}"; }

# =============================================================================
# ██  1. 初始化 Conda  ██
# =============================================================================
CONDA_BASE="/datashare/dir_liuzy0/anaconda3"
if [[ ! -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    log_err "找不到 conda 初始化脚本: ${CONDA_BASE}/etc/profile.d/conda.sh"
fi
source "${CONDA_BASE}/etc/profile.d/conda.sh"
log_ok "Conda 初始化完成 (Base: ${CONDA_BASE})"

# =============================================================================
# ██  2. 核心路径与环境配置  ██
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_VGGSFM="vggsfm"
ENV_GS="gaussian_splatting"

VGGSFM_DIR="/extdatashare/liuzy0/code/vggsfm"
YOLO_MODEL="/extdatashare/liuzy0/ma_dataljp0330/runs/segment/corn-seg-ljp04162/weights/best.pt"

GS_ITERATIONS=30000
VIDEO_FPS=2
YOLO_CONF=0.30

# =============================================================================
# ██  3. Conda Run 执行器封装  ██
# =============================================================================
_find_libittnotify() {
    local env_lib="${CONDA_BASE}/envs/${ENV_VGGSFM}/lib"
    local found
    found=$(find "$env_lib" -maxdepth 3 -name "libittnotify*.so*" 2>/dev/null | head -1)
    if [[ -z "$found" ]]; then
        found=$(ldconfig -p 2>/dev/null | awk '/libittnotify/{print $NF}' | head -1)
    fi
    echo "$found"
}

_LIBITT=$(_find_libittnotify)
if [[ -n "$_LIBITT" ]]; then
    log_info "找到 libittnotify: ${_LIBITT}，已设置 LD_PRELOAD"
    RUN_VGGSFM() {
        LD_PRELOAD="${_LIBITT}${LD_PRELOAD:+:$LD_PRELOAD}" \
        conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"
    }
else
    log_warn "未找到 libittnotify.so，尝试清空 LD_PRELOAD 规避冲突..."
    RUN_VGGSFM() {
        LD_PRELOAD="" \
        conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"
    }
fi

RUN_GS() { conda run --no-capture-output -n "$ENV_GS" python "$@"; }

# =============================================================================
# ██  4. 命令行参数解析  ██
# =============================================================================
VIDEO=""
SCENE_DIR=""

# STEP 5 匹配参数（★ v3新增 / 调整默认值）
MAX_DEPTH=""
SCENE_HARD_CAP="15.0"        # ★ 从30m改为15m（玉米植株场景更保守）
MIN_YOLO_CONF="0.3"          # ★ 新增：YOLO置信度过滤（0=不过滤）
MIN_VIEWS="2"                # ★ 新增：全局ID最少出现帧数（单帧噪声过滤）
NO_APPEARANCE=false          # ★ 新增：是否关闭外观相似性（调试用）

SKIP_VGGSFM=false; SKIP_YOLO=false; SKIP_TRAIN=false
SKIP_DEPTH=false;  SKIP_MATCH=false; SKIP_RENDER=false

usage() {
    cat <<EOF
用法: bash pipeline.sh [选项]

必选（二选一）:
  --video      <path>   输入视频文件路径 (.mp4)
  --scene_dir  <path>   已有场景目录（用于跳过初期步骤）

通用参数:
  --fps         <int>   抽帧帧率 (默认: $VIDEO_FPS)
  --yolo_conf   <float> YOLO推理置信度 (默认: $YOLO_CONF)
  --iterations  <int>   2D-GS训练次数 (默认: $GS_ITERATIONS)

STEP 5 匹配参数（★ v3 外观增强版）:
  --max_depth       <float> 手动指定深度上限（米），不填则自动检测
  --scene_hard_cap  <float> 场景深度硬性上限 (默认: $SCENE_HARD_CAP m)
                            玉米场景建议15m，室内场景可调小
  --min_yolo_conf   <float> YOLO置信度过滤门槛 (默认: $MIN_YOLO_CONF)
                            推荐0.25以过滤低质量检测
  --min_views       <int>   全局ID最少出现帧数 (默认: $MIN_VIEWS)
                            增大可过滤更多单帧噪声，推荐2~3
  --no_appearance           关闭外观相似性证据（调试用，默认开启）

断点续跑控制:
  --skip_vggsfm   跳过 STEP 1
  --skip_yolo     跳过 STEP 2
  --skip_train    跳过 STEP 3
  --skip_depth    跳过 STEP 4
  --skip_match    跳过 STEP 5
  --skip_render   跳过 STEP 6

示例:
  # 全新处理（标准）
  bash pipeline.sh --video /data/corn.mp4

  # 开启YOLO置信度过滤 + 严格噪声过滤
  bash pipeline.sh --video /data/corn.mp4 \\
      --min_yolo_conf 0.25 --min_views 3

  # 只重跑匹配（调试外观参数）
  bash pipeline.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_yolo --skip_train --skip_depth \\
      --min_yolo_conf 0.25 --min_views 2

  # 关闭外观（纯空间匹配，对比测试）
  bash pipeline.sh --scene_dir /data/corn_frames \\
      --skip_vggsfm --skip_yolo --skip_train --skip_depth \\
      --no_appearance
EOF
    exit 0
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)           VIDEO="$2";           shift 2 ;;
        --scene_dir)       SCENE_DIR="$2";       shift 2 ;;
        --max_depth)       MAX_DEPTH="$2";       shift 2 ;;
        --scene_hard_cap)  SCENE_HARD_CAP="$2";  shift 2 ;;
        --min_yolo_conf)   MIN_YOLO_CONF="$2";   shift 2 ;;
        --min_views)       MIN_VIEWS="$2";       shift 2 ;;
        --no_appearance)   NO_APPEARANCE=true;   shift ;;
        --fps)             VIDEO_FPS="$2";       shift 2 ;;
        --yolo_conf)       YOLO_CONF="$2";       shift 2 ;;
        --iterations)      GS_ITERATIONS="$2";   shift 2 ;;
        --skip_vggsfm)     SKIP_VGGSFM=true;     shift ;;
        --skip_yolo)       SKIP_YOLO=true;       shift ;;
        --skip_train)      SKIP_TRAIN=true;      shift ;;
        --skip_depth)      SKIP_DEPTH=true;      shift ;;
        --skip_match)      SKIP_MATCH=true;      shift ;;
        --skip_render)     SKIP_RENDER=true;     shift ;;
        -h|--help)         usage ;;
        *) log_err "未知参数: $1" ;;
    esac
done

# ─── 路径推断与创建 ───
[[ -z "$VIDEO" && -z "$SCENE_DIR" ]] && log_err "请提供 --video 或 --scene_dir"

if [[ -n "$VIDEO" && -z "$SCENE_DIR" ]]; then
    VIDEO_ABS="$(realpath "$VIDEO")"
    SCENE_NAME="$(basename "$VIDEO_ABS" .mp4)"
    DIR_NAME="$(basename "$(dirname "$VIDEO_ABS")")"
    if [[ "$DIR_NAME" == "video" ]]; then
        MAIN_DIR="$(dirname "$(dirname "$VIDEO_ABS")")"
    else
        MAIN_DIR="$(dirname "$VIDEO_ABS")"
    fi
    SCENE_DIR="${MAIN_DIR}/${SCENE_NAME}"
fi

SCENE_DIR="$(realpath -m "$SCENE_DIR")"
mkdir -p "$SCENE_DIR"

# ─── 校验脚本 ───
for f in "vggsfm_cli.py" "reason_cli.py" "train_0211之前.py" \
         "pirender深度图.py" "run_matchingljp0326.py" "pirenderljp0326.py"; do
    [[ -f "${SCRIPT_DIR}/$f" ]] || log_err "在 ${SCRIPT_DIR} 找不到依赖脚本: $f"
done
[[ -f "$YOLO_MODEL" ]] || log_err "YOLO 模型不存在: $YOLO_MODEL"

# =============================================================================
# ██  5. 全局日志  ██
# =============================================================================
LOG_FILE="${SCENE_DIR}/pipeline_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo -e "${BOLD}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║    corn-seg 3D Semantic Pipeline (v3 外观增强版)     ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-20s: %s\n" "场景目录"         "$SCENE_DIR"
printf "  %-20s: %s\n" "输入视频"         "${VIDEO:-[使用已有场景目录]}"
printf "  %-20s: %s\n" "max_depth"        "${MAX_DEPTH:-自动检测}"
printf "  %-20s: %s\n" "scene_hard_cap"   "${SCENE_HARD_CAP}m"
printf "  %-20s: %s\n" "min_yolo_conf"    "$MIN_YOLO_CONF"
printf "  %-20s: %s\n" "min_views"        "$MIN_VIEWS"
printf "  %-20s: %s\n" "appearance"       "$( $NO_APPEARANCE && echo '关闭(调试)' || echo '开启')"
printf "  %-20s: %s\n" "日志文件"         "$LOG_FILE"
echo ""

PIPELINE_START=$(date +%s)
elapsed() { local s=$(( $(date +%s) - $1 )); printf "%dm%02ds" $(( s/60 )) $(( s%60 )); }

# =============================================================================
# STEP 1: VGGSfM
# =============================================================================
log_step 1 "VGGSfM 视频抽帧 + 稀疏重建"
log_env "$ENV_VGGSFM"
T1=$(date +%s)

if $SKIP_VGGSFM; then
    log_skip "VGGSfM (--skip_vggsfm)"
else
    if [[ -d "${SCENE_DIR}/sparse" ]]; then
        log_skip "sparse/ 已存在"
    else
        [[ -n "$VIDEO" ]] || log_err "未提供视频路径"
        cd "$VGGSFM_DIR"
        RUN_VGGSFM "${SCRIPT_DIR}/vggsfm_cli.py" \
            --video "$VIDEO_ABS" --scene_dir "$SCENE_DIR" \
            --fps "$VIDEO_FPS" --vggsfm_dir "$VGGSFM_DIR"
        cd - > /dev/null
    fi
fi

N_FRAMES=$(find "${SCENE_DIR}/images" -maxdepth 1 \
    \( -name "*.jpg" -o -name "*.png" \) 2>/dev/null | wc -l || echo 0)
log_info "图像总帧数: ${N_FRAMES}"

# sparse/0 目录对齐
SPARSE_DIR="${SCENE_DIR}/sparse"
SPARSE_0_DIR="${SPARSE_DIR}/0"
if [[ -d "$SPARSE_DIR" && ! -d "$SPARSE_0_DIR" ]]; then
    log_info "对齐 sparse/ → sparse/0/ ..."
    HAS_COLMAP_FILES=false
    for ext in bin txt; do
        for f in cameras images points3D; do
            [[ -f "${SPARSE_DIR}/${f}.${ext}" ]] && HAS_COLMAP_FILES=true && break 2
        done
    done
    if $HAS_COLMAP_FILES; then
        mkdir -p "$SPARSE_0_DIR"
        for ext in bin txt; do
            for f in cameras images points3D; do
                SRC="${SPARSE_DIR}/${f}.${ext}"
                [[ -f "$SRC" ]] && mv "$SRC" "${SPARSE_0_DIR}/"
            done
        done
        log_ok "sparse/0/ 对齐完成"
    fi
fi
log_ok "STEP 1 完成 (用时 $(elapsed $T1))"

# =============================================================================
# STEP 2: YOLO 分割
# =============================================================================
log_step 2 "YOLO 2D 实例分割"
log_env "$ENV_GS"
T2=$(date +%s)

MASK_DIR="${SCENE_DIR}/masks_results/integer_masks"

if $SKIP_YOLO; then
    log_skip "YOLO (--skip_yolo)"
else
    EXISTING_MASKS=$(find "$MASK_DIR" -name "mask_*.png" 2>/dev/null | wc -l || echo 0)
    if [[ "$EXISTING_MASKS" -ge "$N_FRAMES" && "$EXISTING_MASKS" -gt 0 ]]; then
        log_skip "掩码已完整 ($EXISTING_MASKS 帧)"
    else
        RUN_GS "${SCRIPT_DIR}/reason_cli.py" \
            --scene_dir "$SCENE_DIR" \
            --model     "$YOLO_MODEL" \
            --conf      "$YOLO_CONF" \
            --save_viz
    fi
fi
log_ok "STEP 2 完成 (用时 $(elapsed $T2))"

# =============================================================================
# STEP 3: 2D-GS 训练
# =============================================================================
log_step 3 "2D-GS 高斯重建 (iter=$GS_ITERATIONS)"
log_env "$ENV_GS"
T3=$(date +%s)

SCENE_BASENAME="$(basename "$SCENE_DIR")"
BASE_NAME="${SCENE_BASENAME%_frames}"
OUTPUT_DIR="${SCENE_DIR}/output_${BASE_NAME}"

if $SKIP_TRAIN; then
    log_skip "2D-GS 训练 (--skip_train)"
else
    if [[ -d "$OUTPUT_DIR" && -f "${OUTPUT_DIR}/cameras.json" ]]; then
        log_skip "output_*/cameras.json 已存在"
    else
        RUN_GS "${SCRIPT_DIR}/train_0211之前.py" \
            --source_path "$SCENE_DIR" \
            -m            "$OUTPUT_DIR" \
            --iterations  "$GS_ITERATIONS"
    fi
fi
log_ok "STEP 3 完成 (用时 $(elapsed $T3))"

# =============================================================================
# STEP 4: 深度图渲染（渲染全部帧）
# =============================================================================
log_step 4 "渲染全部深度图"
log_env "$ENV_GS"
T4=$(date +%s)

DEPTH_DIR="${OUTPUT_DIR}/train/ours_${GS_ITERATIONS}/depth"

if $SKIP_DEPTH; then
    log_skip "深度图渲染 (--skip_depth)"
else
    N_DEPTH=$(find "$DEPTH_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
    if [[ "$N_DEPTH" -ge "$N_FRAMES" && "$N_DEPTH" -gt 0 ]]; then
        log_skip "深度图已全部生成 ($N_DEPTH 帧)"
    else
        RUN_GS "${SCRIPT_DIR}/pirender深度图.py" \
            "$OUTPUT_DIR" \
            --iteration "$GS_ITERATIONS" \
            --skip_mesh \
            --resolution 1
        N_DEPTH=$(find "$DEPTH_DIR" -name "*.png" 2>/dev/null | wc -l || echo 0)
        log_info "渲染完成，共 ${N_DEPTH} 帧深度图"
    fi
fi
log_ok "STEP 4 完成 (用时 $(elapsed $T4))"

# =============================================================================
# STEP 5: 3D 语义匹配（v3：外观增强 + 噪声过滤 + 0302参数）
# =============================================================================
log_step 5 "3D 语义匹配 v3（外观增强 + 噪声过滤）"
log_env "$ENV_GS"
T5=$(date +%s)

MATCH_DIR="${SCENE_DIR}/数据驱动匹配"

if $SKIP_MATCH; then
    log_skip "3D匹配 (--skip_match)"
else
    if [[ -f "${MATCH_DIR}/id_mapping.json" ]]; then
        log_skip "id_mapping.json 已存在"
    else
        # ── 构建匹配命令 ────────────────────────────────────────
        MATCH_CMD=(
            "${SCRIPT_DIR}/run_matchingljp0326.py"
            "$SCENE_DIR"
            "--scene_hard_cap" "$SCENE_HARD_CAP"
            "--min_yolo_conf"  "$MIN_YOLO_CONF"
            "--min_views"      "$MIN_VIEWS"
        )

        # 手动指定 max_depth
        if [[ -n "$MAX_DEPTH" ]]; then
            MATCH_CMD+=("--max_depth" "$MAX_DEPTH")
            log_info "使用手动 max_depth=${MAX_DEPTH}m"
        else
            log_info "max_depth 自动检测 (hard_cap=${SCENE_HARD_CAP}m)"
        fi

        # 关闭外观（调试选项）
        if $NO_APPEARANCE; then
            MATCH_CMD+=("--no_appearance")
            log_warn "外观相似性已关闭（调试模式）"
        fi

        log_info "匹配参数: min_yolo_conf=${MIN_YOLO_CONF}, min_views=${MIN_VIEWS}"
        log_info "参数版本: 0302回归 (spatial=0.3, top_k=2000)"

        export ORIGINAL_SCRIPT_DIR="${SCRIPT_DIR}"
        RUN_GS "${MATCH_CMD[@]}"
    fi
fi
log_ok "STEP 5 完成 (用时 $(elapsed $T5))"

# =============================================================================
# STEP 6: 最终渲染
# =============================================================================
log_step 6 "带全局语义 ID 的最终渲染"
log_env "$ENV_GS"
T6=$(date +%s)

if $SKIP_RENDER; then
    log_skip "最终渲染 (--skip_render)"
else
    RUN_GS "${SCRIPT_DIR}/pirenderljp0326.py" \
        "$SCENE_DIR" \
        --gs_id \
        --skip_train \
        --skip_test \
        --skip_mesh \
        --iteration "$GS_ITERATIONS"
fi
log_ok "STEP 6 完成 (用时 $(elapsed $T6))"

# =============================================================================
# 完成汇总
# =============================================================================
TOTAL_SEC=$(( $(date +%s) - PIPELINE_START ))

echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║         全流程完美结束 🎉  (v3 外观增强版)          ║${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════════╝${NC}"
printf "  %-20s: %s\n" "总耗时"          "$(printf "%dm%02ds" $(( TOTAL_SEC/60 )) $(( TOTAL_SEC%60 )))"
printf "  %-20s: %s\n" "场景目录"        "$SCENE_DIR"
printf "  %-20s: %s\n" "匹配参数版本"    "0302回归 + 外观增强"
printf "  %-20s: %s\n" "min_views过滤"   "$MIN_VIEWS 帧"
printf "  %-20s: %s\n" "3D匹配输出"      "$MATCH_DIR"
printf "  %-20s: %s\n" "最终渲染"        "$OUTPUT_DIR"
printf "  %-20s: %s\n" "噪声ID清单"      "${MATCH_DIR}/noise_ids_removed.txt"
echo -e "\n${CYAN}日志: ${LOG_FILE}${NC}"
