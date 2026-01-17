#!/bin/bash
#
# 测试脚本 - pass@1 快速验证
# 只测试 2 个实验的 step 105
#

set -e

# ============ 环境配置 ============
export CUDA_VISIBLE_DEVICES=3,4,5,6,7
export PYTHONPATH=/scratch/jin509/self_RL/Archer2.0:$PYTHONPATH
PYTHON=/scratch/jin509/miniconda3/envs/archer/bin/python

# ============ 路径配置 ============
BASE_DIR=/scratch/jin509/self_RL/Archer2.0
OUTPUT_ROOT=${BASE_DIR}/output/ArcherCodeR
DATA_DIR=${BASE_DIR}/data/test

# ============ 评估参数 (pass@1 快速测试) ============
n_gpus=5
tp_size=1
n_samples=1            # pass@1 快速测试
temperature=0.6
top_p=0.95
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response
batch_size=2048

# ============ 日志配置 ============
LOG_DIR=${BASE_DIR}/tools/logs
mkdir -p ${LOG_DIR}
MAIN_LOG=${LOG_DIR}/test_merge_eval_$(date +%Y%m%d_%H%M%S).log

# ============ 测试实验列表 (只选2个) ============
EXPERIMENTS=(
    "Archer-TokenEntropy-Qwen2.5-1.5B-2k-8k-batch64-no-kl-simple"
    "Pure-GRPO-Qwen2.5-1.5B-2K-8K-n8-no-kl-v2"
)

# 只测试 step 105
STEPS=(105)

# ============ 辅助函数 ============

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a ${MAIN_LOG}
}

check_checkpoint_exists() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    [ -d "$ckpt_path" ] && [ -f "$ckpt_path/config.json" ]
}

check_hf_model_exists() {
    local exp_name=$1
    local step=$2
    local hf_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    [ -d "$hf_path" ] && [ -f "$hf_path/config.json" ]
}

check_eval_exists() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local result_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model/output_test/${dataset}.parquet"
    [ -f "$result_path" ]
}

merge_model() {
    local exp_name=$1
    local step=$2
    local ckpt_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor"
    local hf_path="${ckpt_path}/hf_model"
    
    log "🔧 开始合并模型: ${exp_name} step ${step}"
    
    if check_hf_model_exists "$exp_name" "$step"; then
        log "✓ HF模型已存在，跳过合并: ${hf_path}"
        return 0
    fi
    
    local start_time=$(date +%s)
    
    $PYTHON -m tools.model_merge merge \
        --backend fsdp \
        --local_dir "${ckpt_path}" \
        --target_dir "${hf_path}" 2>&1 | tee -a ${MAIN_LOG}
    
    local exit_code=${PIPESTATUS[0]}
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    if [ $exit_code -eq 0 ] && [ -f "${hf_path}/config.json" ]; then
        log "✓ 模型合并成功 (耗时: ${duration}s)"
        return 0
    else
        log "✗ 模型合并失败"
        [ -d "${hf_path}" ] && rm -rf "${hf_path}"
        return 1
    fi
}

run_eval() {
    local exp_name=$1
    local step=$2
    local dataset=$3
    local model_path="${OUTPUT_ROOT}/${exp_name}/global_step_${step}/actor/hf_model"
    local output_dir="${model_path}/output_test"  # 测试用单独目录
    
    log "📊 开始评估: ${exp_name} step ${step} - ${dataset} (pass@${n_samples})"
    
    if check_eval_exists "$exp_name" "$step" "$dataset"; then
        log "✓ 评估结果已存在，跳过"
        return 0
    fi
    
    mkdir -p "${output_dir}"
    
    local start_time=$(date +%s)
    
    $PYTHON -m verl.trainer.main_generation \
        trainer.nnodes=1 \
        trainer.n_gpus_per_node=${n_gpus} \
        +trainer.project_name=ArcherCodeR_Test \
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
        log "✓ 评估完成 (耗时: ${duration}s)"
        return 0
    else
        log "✗ 评估失败"
        return 1
    fi
}

process_checkpoint() {
    local exp_name=$1
    local step=$2
    
    log "============================================================"
    log "处理: ${exp_name} - global_step_${step}"
    log "============================================================"
    
    if ! check_checkpoint_exists "$exp_name" "$step"; then
        log "⚠ Checkpoint 不存在，跳过"
        return 0
    fi
    
    if ! merge_model "$exp_name" "$step"; then
        log "⚠ 合并失败，跳过评估"
        return 1
    fi
    
    run_eval "$exp_name" "$step" "aime2024"
    run_eval "$exp_name" "$step" "aime2025"
    
    log "✓ 完成处理: ${exp_name} - global_step_${step}"
}

# ============ 主流程 ============

main() {
    log "============================================================"
    log "🧪 测试脚本 - pass@1 快速验证"
    log "============================================================"
    log "测试实验: ${EXPERIMENTS[*]}"
    log "测试 Steps: ${STEPS[*]}"
    log "n_samples: ${n_samples} (pass@1)"
    log "GPU 数量: ${n_gpus}"
    log "============================================================"
    
    for step in "${STEPS[@]}"; do
        for exp_name in "${EXPERIMENTS[@]}"; do
            process_checkpoint "$exp_name" "$step"
            log ""
        done
    done
    
    log "============================================================"
    log "🎉 测试完成!"
    log "日志: ${MAIN_LOG}"
    log "============================================================"
}

cd ${BASE_DIR}
main

