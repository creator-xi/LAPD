#!/bin/bash
set -e  
export CUDA_VISIBLE_DEVICES=''

data_source="main"
dataset="xsum"
source_model="gpt4o"

max_samples=100

python -m method.lapd_multi_gpu \
    --data_source "$data_source" \
    --sampling_model llama2-7b \
    --base_model llama2-7b \
    --instruct_model llama2-7b-instruct \
    --dataset "$dataset" \
    --source_model "$source_model" \
    --max_samples $max_samples \
    --device cuda \
    --time_compute

echo "Computation of our method time efficiency complete!"

METHODS=("dna_detectllm" "binoculars" "lastde" "fast_origin" "entropy" "logrank" "likelihood" "detectgpt" "rai_double_gpu")
for method in "${METHODS[@]}"; do
    python -m method.baselines.run_baselines \
        --method "$method" \
        --data_source "$data_source" \
        --dataset "$dataset" \
        --source_model "$source_model" \
        --sampling_model_name llama2-7b \
        --scoring_model_name llama2-7b-instruct \
        --max_samples "$max_samples" \
        --time_compute
done
echo "Time efficiency computation of all methods complete!"
