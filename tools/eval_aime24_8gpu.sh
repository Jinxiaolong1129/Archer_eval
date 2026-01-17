#!/bin/bash
#
# 8卡 GPU 评估脚本 - 测试速度
# Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
# Dataset: AIME 2024
#

set -e

# 设置 CUDA 设备 (使用全部 8 张 GPU)
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 使用 archer 环境的 Python
PYTHON=/scratch/jin509/miniconda3/envs/archer/bin/python
export PYTHONPATH=/scratch/jin509/self_RL/Archer2.0:$PYTHONPATH

# ============ 配置 ============
nnodes=1
n_gpus=8   # 使用 8 张 GPU
tp_size=1  # 数据并行模式 (每张GPU独立处理，速度更快)

# Model: 直接使用 HuggingFace 模型
model_path=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

# 数据路径
base_dir=/scratch/jin509/self_RL/Archer2.0
data_dir=${base_dir}/data/test
dataset=aime2024

# 评估参数 (AIME math task)
n_samples=8          # 每个问题生成 8 个回答用于计算 pass@K
temperature=0.6      # 数学任务温度稍低
top_p=0.95
max_prompt_length=$((1024 * 2))      # 2K prompt
max_response_length=$((1024 * 8))    # 8K response (测试速度)
batch_size=10        # 小批次，方便观察进度

# 输出路径 (包含响应长度标识)
response_len_k=$((max_response_length / 1024))
output_dir=${base_dir}/output/eval_results/${dataset}_${response_len_k}k_n${n_samples}_8gpu
# 删除之前的结果
rm -rf ${output_dir}
mkdir -p ${output_dir}

# 项目配置
project_name=AIME_Eval
experiment_name=DeepSeek-R1-Distill-Qwen-1.5B-8gpu

echo "============================================================"
echo "AIME 2024 Evaluation - 8 GPU Speed Test"
echo "============================================================"
echo "Model: ${model_path}"
echo "Dataset: ${data_dir}/${dataset}.parquet"
echo "Output: ${output_dir}/${dataset}.parquet"
echo "GPU: CUDA:0-7 (${n_gpus} GPUs, Data Parallel)"
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

