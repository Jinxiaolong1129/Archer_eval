#!/bin/bash
#
# 5卡 GPU 评估脚本 - AIME 2025 pass@8
# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Dataset: AIME 2025
#

set -e

# 设置 CUDA 设备 (使用 GPU 3, 4, 5, 6, 7)
export CUDA_VISIBLE_DEVICES=3,4,5,6,7

# 使用 archer 环境的 Python
PYTHON=/scratch/jin509/miniconda3/envs/archer/bin/python
export PYTHONPATH=/scratch/jin509/self_RL/Archer2.0:$PYTHONPATH

# ============ 配置 ============
nnodes=1
n_gpus=5   # 使用 5 张 GPU
tp_size=1  # 数据并行模式

# Model: 直接使用 HuggingFace 模型
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 数据路径
base_dir=/scratch/jin509/self_RL/Archer2.0
data_dir=${base_dir}/data/test
dataset=aime2025

# 评估参数 (AIME math task)
n_samples=8          # pass@8
temperature=0.6      # 数学任务温度稍低
top_p=0.95
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response
batch_size=10

# 输出路径
response_len_k=$((max_response_length / 1024))
output_dir=${base_dir}/output/eval_results/${dataset}_${response_len_k}k_n${n_samples}
# 删除之前的结果
rm -rf ${output_dir}
mkdir -p ${output_dir}

# 项目配置
project_name=AIME_Eval
experiment_name=DeepSeek-R1-Distill-Qwen-1.5B

echo "============================================================"
echo "AIME 2025 Evaluation - pass@${n_samples}"
echo "============================================================"
echo "Model: ${model_path}"
echo "Dataset: ${data_dir}/${dataset}.parquet"
echo "Output: ${output_dir}/${dataset}.parquet"
echo "GPU: CUDA:3,4,5,6,7 (${n_gpus} GPUs, Data Parallel)"
echo "n_samples: ${n_samples}"
echo "batch_size: ${batch_size}"
echo "max_response_length: ${max_response_length}"
echo "temperature: ${temperature}"
echo "============================================================"

# 运行评估
$PYTHON -m verl.trainer.main_generation \
    trainer.nnodes=${nnodes} \
    trainer.n_gpus_per_node=${n_gpus} \
    +trainer.project_name=${project_name} \
    +trainer.experiment_name=${experiment_name} \
    +trainer.task_name=${dataset} \
    +trainer.global_step=0 \
    +trainer.use_wandb=False \
    model.path=${model_path} \
    data.path=${data_dir}/${dataset}.parquet \
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
    rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))

echo ""
echo "============================================================"
echo "✓ Evaluation complete!"
echo "Results saved to: ${output_dir}/${dataset}.parquet"
echo "============================================================"

