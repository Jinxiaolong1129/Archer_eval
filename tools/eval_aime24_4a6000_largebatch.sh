#!/bin/bash
#
# 4卡 A6000 评估脚本 - AIME24 pass@32 大 batch 测试
# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Dataset: AIME 2024
#

set -e

# 设置 CUDA 设备 (使用 GPU 3, 4, 5, 6 - 4x A6000)
export CUDA_VISIBLE_DEVICES=3,4,5,6

# 使用 archer 环境的 Python
PYTHON=/scratch/jin509/miniconda3/envs/archer/bin/python
export PYTHONPATH=/scratch/jin509/self_RL/Archer2.0:$PYTHONPATH

# ============ 配置 ============
nnodes=1
n_gpus=4   # 使用 4 张 A6000
tp_size=1  # 数据并行模式

# Model: 直接使用 HuggingFace 模型
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 数据路径
base_dir=/scratch/jin509/self_RL/Archer2.0
data_dir=${base_dir}/data/test
dataset=aime2024

# 评估参数 (AIME math task) - pass@32
n_samples=32         # 每个问题生成 32 个回答
temperature=0.6      
top_p=0.95
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response
batch_size=2048      # 大 batch，让 vLLM continuous batching 自己调度

# 输出路径
response_len_k=$((max_response_length / 1024))
output_dir=${base_dir}/output/eval_results/${dataset}_${response_len_k}k_n${n_samples}_4a6000_largebatch
# 删除之前的结果
rm -rf ${output_dir}
mkdir -p ${output_dir}

# 项目配置
project_name=AIME_Eval
experiment_name=DeepSeek-R1-Distill-Qwen-1.5B-4A6000-largebatch

# 预估时间
total_responses=$((30 * n_samples))

echo "============================================================"
echo "AIME 2024 Evaluation - pass@${n_samples} (4x A6000, Large Batch)"
echo "============================================================"
echo "Model: ${model_path}"
echo "Dataset: ${data_dir}/${dataset}.parquet"
echo "Output: ${output_dir}/${dataset}.parquet"
echo "GPU: CUDA:3,4,5,6 (${n_gpus} x A6000, Data Parallel)"
echo "n_samples: ${n_samples}"
echo "Total responses: ${total_responses}"
echo "batch_size: ${batch_size} (large batch for vLLM continuous batching)"
echo "max_response_length: ${max_response_length}"
echo "temperature: ${temperature}"
echo "============================================================"
echo "Baseline: 5x A6000 = 637.3s"
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

