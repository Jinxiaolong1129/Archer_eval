#!/bin/bash
#
# 批量模型合并与 AIME 评估脚本 - Part 2 (3 A100 GPU)
# 对包含 50/90/105 checkpoint 的实验进行 merge 并评估 AIME24 和 AIME25
# 优先级顺序: 105 -> 50 -> 90
# GPU: 0,1,2 (3个A100)
#

set -e

# ============ 环境配置 ============
export CUDA_VISIBLE_DEVICES=0,1,2
export PYTHONPATH=/scratch/jin509/self_RL/Archer2.0:$PYTHONPATH
PYTHON=/scratch/jin509/miniconda3/envs/archer/bin/python

# ============ 路径配置 ============
BASE_DIR=/scratch/jin509/self_RL/Archer2.0
OUTPUT_ROOT=${BASE_DIR}/output/ArcherCodeR
DATA_DIR=${BASE_DIR}/data/test

# ============ 评估参数 ============
n_gpus=3
tp_size=1
n_samples=32           # pass@32
temperature=0.6
top_p=0.95
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response
batch_size=2048

# ============ 日志配置 ============
LOG_DIR=${BASE_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/batch_merge_eval_part2_$(date +%Y%m%d_%H%M%S).log

# ============ 实验列表 Part 2 (5个实验) ============
EXPERIMENTS=(
    "Archer-Intuitor-Qwen2.5-1.5B-2k-8k-batch64-no-kl-n12"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-kl005-v2"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl-v2"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-16resp-no-kl-temp0.8-v2"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-n12-no-kl-v2"
)

# 按优先级排序的 checkpoint steps
STEPS=(105 50 90)

# ============ 辅助函数 ============

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

# 检查 checkpoint 是否存在 actor 目录
check_checkpoint_exists() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    
    if [ -d "$ckpt_path" ] && [ -f "$ckpt_path/config.json" ]; then
        return 0
    else
        return 1
    fi
}

# 检查 HF 模型是否已存在
check_hf_model_exists() {
    local exp_name=$1
    local step=$2
    local hf_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    
    if [ -d "$hf_path" ] && [ -f "$hf_path/config.json" ]; then
        return 0
    else
        return 1
    fi
}

# 检查评估结果是否已存在
check_eval_exists() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local result_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/output/${dataset}.parquet"
    
    if [ -f "$result_path" ]; then
        return 0
    else
        return 1
    fi
}

# 合并模型
merge_model() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    local hf_path="${ckpt_path}/hf_model"
    
    log "🔧 开始合并模型: ${exp_name} step ${step}"
    
    # 检查是否已合并
    if check_hf_model_exists "$exp_name" "$step"; then
        log "✓ HF模型已存在，跳过合并: ${hf_path}"
        return 0
    fi
    
    # 执行合并
    local start_time=$(date +%s)
    
    $PYTHON -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${ckpt_path}" \
        --target_dir "${hf_path}" 2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${hf_path}/config.json" ]; then
        log "✓ 模型合并成功 (耗时: ${duration}s): ${exp_name} step ${step}"
        return 0
    else
        log "✗ 模型合并失败: ${exp_name} step ${step}"
        # 清理失败的合并结果
        [ -d "${hf_path}" ] && rm -rf "${hf_path}"
        return 1
    fi
}

# 运行评估
run_eval() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local model_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    local output_dir="${model_path}/output"
    
    log "📊 开始评估: ${exp_name} step ${step} - ${dataset}"
    
    # 检查是否已评估
    if check_eval_exists "$exp_name" "$step" "$dataset"; then
        log "✓ 评估结果已存在，跳过: ${output_dir}/${dataset}.parquet"
        return 0
    fi
    
    # 创建输出目录
    mkdir -p "${output_dir}"
    
    local start_time=$(date +%s)
    
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Eval \
        +trainer.experiment_name=${exp_name} \
        +trainer.task_name=${dataset} \
        +trainer.global_step=${step} \
        +trainer.use_wandb=False \
        model.path=${model_path} \
        data.path=${DATA_DIR}/${dataset}.parquet \
        data.output_path=${output_dir}/${dataset}.parquet \
        data.batch_size=${batch_size} \
        data.n_samples=${n_samples} \
        rollout.name=vllm \
        rollout.gpu_memory_utilization=0.9 \
        rollout.enforce_eager=False \
        rollout.free_cache_engine=False \
        rollout.disable_log_stats=False \
        rollout.tensor_model_parallel_size=${tp_size} \
        rollout.temperature=${temperature} \
        rollout.top_k=-1 \
        rollout.top_p=${top_p} \
        rollout.prompt_length=${max_prompt_length} \
        rollout.response_length=${max_response_length} \
        rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
        2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${output_dir}/${dataset}.parquet" ]; then
        log "✓ 评估完成 (耗时: ${duration}s): ${exp_name} step ${step} - ${dataset}"
        return 0
    else
        log "✗ 评估失败: ${exp_name} step ${step} - ${dataset}"
        return 1
    fi
}

# 处理单个 checkpoint
process_checkpoint() {
    local exp_name=$1
    local step=$2
    
    log "============================================================"
    log "处理: ${exp_name} - global_step_${step}"
    log "============================================================"
    
    # 检查 checkpoint 是否存在
    if ! check_checkpoint_exists "$exp_name" "$step"; then
        log "⚠ Checkpoint 不存在，跳过: ${exp_name} step ${step}"
        return 0
    fi
    
    # 1. 合并模型
    if ! merge_model "$exp_name" "$step"; then
        log "⚠ 合并失败，跳过评估: ${exp_name} step ${step}"
        return 1
    fi
    
    # 2. 评估 AIME24
    run_eval "$exp_name" "$step" "aime2024"
    
    # 3. 评估 AIME25
    run_eval "$exp_name" "$step" "aime2025"
    
    log "✓ 完成处理: ${exp_name} - global_step_${step}"
}

# ============ 主流程 ============

main() {
    log "============================================================"
    log "批量模型合并与 AIME 评估脚本 - Part 2 (3 A100)"
    log "============================================================"
    log "实验数量: ${#EXPERIMENTS[@]}"
    log "Checkpoint steps: ${STEPS[*]}"
    log "GPU: 0,1,2 (${n_gpus} GPUs)"
    log "n_samples: ${n_samples} (pass@32)"
    log "batch_size: ${batch_size}"
    log "max_response_length: ${max_response_length}"
    log "============================================================"
    
    # 统计计数
    local total_tasks=0
    local completed_tasks=0
    local failed_tasks=0
    
    # 按优先级顺序处理: 105 -> 50 -> 90
    for step in "${STEPS[@]}"; do
        log ""
        log "########################################################"
        log "开始处理 Step ${step} (所有实验)"
        log "########################################################"
        
        for exp_name in "${EXPERIMENTS[@]}"; do
            ((total_tasks+=1))
            
            if process_checkpoint "$exp_name" "$step"; then
                ((completed_tasks+=1))
            else
                ((failed_tasks+=1))
            fi
            
            log ""
        done
    done
    
    # 打印统计
    log "============================================================"
    log "🎉 批量处理完成 (Part 2)!"
    log "============================================================"
    log "总任务数: ${total_tasks}"
    log "成功: ${completed_tasks}"
    log "失败: ${failed_tasks}"
    log "日志文件: ${MAIN_LOG}"
    log "============================================================"
}

# 运行主流程
cd ${BASE_DIR}
main

