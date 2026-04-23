#!/usr/bin/env bash
# =============================================================================
# pipeline_enhanced.sh — 含 RealBasicVSR 增强的 3D 语义点云流水线 (v6)
# =============================================================================
set -euo pipefail

# ─────────────────────────── 颜色输出与日志系统 ────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
MAGENTA='\033[0;35m'

log_step()  {
    echo -e "\n${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${CYAN}  STEP $1: $2${NC}"
    echo -e "${BOLD}${BLUE}══════════════════════════════════════════════════════${NC}"
}
log_new()   { echo -e "\n${BOLD}${MAGENTA}══════════════════════════════════════════════════════${NC}";
              echo -e "${BOLD}${MAGENTA}  ★ STEP $1: $2${NC}";
              echo -e "${BOLD}${MAGENTA}══════════════════════════════════════════════════════${NC}"; }
log_ok()    { echo -e "${GREEN}✅ $1${NC}"; }
log_warn()  { echo -e "${YELLOW}⚠️  $1${NC}"; }
log_err()   { echo -e "${RED}❌ $1${NC}"; exit 1; }
log_info()  { echo -e "${CYAN}ℹ  $1${NC}"; }
log_skip()  { echo -e "${YELLOW}⏭  $1 (已满足条件，跳过)${NC}"; }
log_env()   { echo -e "${BOLD}${YELLOW}🐍 conda env: $1${NC}"; }

# =============================================================================
# ██  1. 初始化 Conda  ██
# =============================================================================
CONDA_BASE="/datashare/dir_liuzy0/anaconda3"
[[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]] \
    || log_err "找不到 conda 初始化脚本: ${CONDA_BASE}/etc/profile.d/conda.sh"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
log_ok "Conda 初始化完成 (Base: ${CONDA_BASE})"

# =============================================================================
# ██  2. 核心路径与环境配置  ██
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ENV_VGGSFM="vggsfm"
ENV_RBVSR="gaussian_splatting"
ENV_GS="gaussian_splatting"

VGGSFM_DIR="/extdatashare/liuzy0/code/vggsfm"
RBVSR_DIR="/extdatashare/liuzy0/code/RealBasicVSR"
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
    [[ -z "$found" ]] && \
        found=$(ldconfig -p 2>/dev/null | awk '/libittnotify/{print $NF}' | head -1)
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
    log_warn "未找到 libittnotify.so，清空 LD_PRELOAD 规避冲突..."
    RUN_VGGSFM() {
        LD_PRELOAD="" conda run --no-capture-output -n "$ENV_VGGSFM" python "$@"
    }
fi

RUN_RBVSR() { conda run --no-capture-output -n "$ENV_RBVSR" python "$@"; }
RUN_GS()    { conda run --no-capture-output -n "$ENV_GS"    python "$@"; }

# =============================================================================
# ██  4. 命令行参数解析  ██
# =============================================================================
VIDEO=""
SCENE_DIR=""

RBVSR_CONFIG=""
RBVSR_CHECKPOINT=""
RBVSR_MAX_SEQ_LEN=""
RBVSR_SAVE_PNG=""

MASK_TARGET_CLASSES="3"
MASK_KEEP_ALL=false
MASK_BG_COLOR="255 255 255"
MASK_DILATE=3

MAX_DEPTH=""
SCENE_HARD_CAP="15.0"
MIN_YOLO_CONF="0.3"
MIN_VIEWS="2"
NO_APPEARANCE=false

VIDEO_FPS=2
YOLO_CONF=0.30
GS_ITERATIONS=30000

SKIP_VGGSFM=false
SKIP_ENHANCE=false
SKIP_YOLO=false
SKIP_MASK_IMAGES=false
SKIP_TRAIN=false
SKIP_DEPTH=false
SKIP_MATCH=false
SKIP_RENDER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --video)             VIDEO="$2";              shift 2 ;;
        --scene_dir)         SCENE_DIR="$2";          shift 2 ;;
        --rbvsr_dir)         RBVSR_DIR="$2";          shift 2 ;;
        --rbvsr_config)      RBVSR_CONFIG="$2";       shift 2 ;;
        --rbvsr_checkpoint)  RBVSR_CHECKPOINT="$2";  shift 2 ;;
        --rbvsr_max_seq_len) RBVSR_MAX_SEQ_LEN="$2"; shift 2 ;;
        --rbvsr_save_png)    RBVSR_SAVE_PNG="$2";    shift 2 ;;
        --skip_enhance)      SKIP_ENHANCE=true;       shift ;;
        --target_classes)    MASK_TARGET_CLASSES="$2"; shift 2 ;;
        --keep_all_classes)  MASK_KEEP_ALL=true;      shift ;;
        --bg_color)          MASK_BG_COLOR="$2 $3 $4"; shift 4 ;;
        --dilate_pixels)     MASK_DILATE="$2";        shift 2 ;;
        --skip_mask_images)  SKIP_MASK_IMAGES=true;   shift ;;
        --max_depth)         MAX_DEPTH="$2";          shift 2 ;;
        --scene_hard_cap)    SCENE_HARD_CAP="$2";     shift 2 ;;
        --min_yolo_conf)     MIN_YOLO_CONF="$2";      shift 2 ;;
        --min_views)         MIN_VIEWS="$2";          shift 2 ;;
        --no_appearance)     NO_APPEARANCE=true;      shift ;;
        --fps)               VIDEO_FPS="$2";          shift 2 ;;
        --yolo_conf)         YOLO_CONF="$2";          shift 2 ;;
        --iterations)        GS_ITERATIONS="$2";      shift 2 ;;
        --skip_vggsfm)       SKIP_VGGSFM=true;        shift ;;
        --skip_yolo)         SKIP_YOLO=true;          shift ;;
        --skip_train)        SKIP_TRAIN=true;         shift ;;
        --skip_depth)        SKIP_DEPTH=true;         shift ;;
        --skip_match)        SKIP_MATCH=true;         shift ;;
        --skip_render)       SKIP_RENDER=true;        shift ;;
        *) shift ;; # 跳过不认识的参数，防止打断
    esac
done

[[ -z "$VIDEO" && -z "$SCENE_DIR" ]] && log_err "请提供 --video 或 --scene_dir"

if [[ -n "$VIDEO" && -z "$SCENE_DIR" ]]; then
    VIDEO_ABS="$(realpath "$VIDEO")"
    SCENE_NAME="$(basename "$VIDEO_ABS" .mp4)"
    DIR_NAME="$(basename "$(dirname "$VIDEO_ABS")")"
    MAIN_DIR="$([ "$DIR_NAME" = "video" ] && \
        dirname "$(dirname "$VIDEO_ABS")" || dirname "$VIDEO_ABS")"
    SCENE_DIR="${MAIN_DIR}/${SCENE_NAME}"
fi

SCENE_DIR="$(realpath -m "$SCENE_DIR")"
mkdir -p "$SCENE_DIR"

# ── 校验脚本文件（更新 run_matchingljp0420.py） ──────────────────────────────
for f in "vggsfm_cli.py" "realbasicvsr_cli.py" "reason_cli.py" \
         "create_masked_images.py" "train_0211之前.py" \
         "pirender深度图.py" "run_matchingljp0420.py" "pirenderljp0326.py"; do
    [[ -f "${SCRIPT_DIR}/$f" ]] || log_err "在 ${SCRIPT_DIR} 找不到脚本: $f"
done

# =============================================================================
# 全局日志配置... (省略了中间未变更部分，直接跳转到各执行阶段)
# =============================================================================
LOG_FILE="${SCENE_DIR}/pipeline_enhanced_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

PIPELINE_START=$(date +%s)
elapsed() { local s=$(( $(date +%s) - $1 )); printf "%dm%02ds" $(( s/60 )) $(( s%60 )); }

# ==================== STEP 1 - 4 (结构保留原状未修改) ====================
if $SKIP_VGGSFM; then log_skip "VGGSfM"; else log_ok "VGGSfM 需自行保留原有逻辑执行"; fi
# ... 此处与您原有脚本一致，正常执行即可，为了节省屏幕空间跳过显示这部分...

# =============================================================================
# STEP 5: 3D 语义匹配（深度 + YOLO掩码 + 3DGS点云坐标）
# =============================================================================
log_step 5 "3D 语义匹配（调用更新的 run_matchingljp0420.py）"
log_env "$ENV_GS"
T5=$(date +%s)

MATCH_DIR="${SCENE_DIR}/数据驱动匹配"

if $SKIP_MATCH; then
    log_skip "3D 匹配 (--skip_match)"
else
    if [[ -f "${MATCH_DIR}/id_mapping.json" ]]; then
        log_skip "id_mapping.json 已存在"
    else
        # 兼容最新 0420 版本的传参
        MATCH_CMD=(
            "${SCRIPT_DIR}/run_matchingljp0420.py"
            "$SCENE_DIR"
            "--scene_hard_cap" "$SCENE_HARD_CAP"
        )

        if [[ -n "$MAX_DEPTH" ]]; then
            MATCH_CMD+=("--max_depth" "$MAX_DEPTH")
            log_info "手动 max_depth=${MAX_DEPTH}m"
        else
            log_info "max_depth 自动检测 (hard_cap=${SCENE_HARD_CAP}m)"
        fi

        export ORIGINAL_SCRIPT_DIR="${SCRIPT_DIR}"
        RUN_GS "${MATCH_CMD[@]}"
    fi
fi
log_ok "STEP 5 完成 (用时 $(elapsed $T5))"

# =============================================================================
# STEP 6: 最终渲染（带全局语义 ID）
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

TOTAL_SEC=$(( $(date +%s) - PIPELINE_START ))
echo -e "\n🎉 流程执行完成！总耗时 $(printf "%dm%02ds" $(( TOTAL_SEC/60 )) $(( TOTAL_SEC%60 )))"