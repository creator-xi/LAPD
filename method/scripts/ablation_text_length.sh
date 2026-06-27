#!/bin/bash
set -e
set -u
export CUDA_VISIBLE_DEVICES=''

data_source="main"
dataset="xsum"
source_model="gpt4o"
max_samples=300
sampling_model="llama2-7b"
scoring_model="llama2-7b-instruct"

METHODS=("rai" "dna_detectllm" "binoculars" "lastde" "fast")
WORD_COUNTS=(20 40 60 80 100 120 140 160 180 200)

echo "========================================="
echo "Text Length Ablation Study"
echo "========================================="
echo "Dataset: $dataset"
echo "Source model: $source_model"
echo "Samples: $max_samples"
echo "Word counts: ${WORD_COUNTS[@]} (10 points)"
echo "Methods: ${METHODS[@]}"
echo "Total experiments: ${#METHODS[@]} methods x ${#WORD_COUNTS[@]} word counts = $(( ${#METHODS[@]} * ${#WORD_COUNTS[@]} )) runs"
echo "========================================="

for method in "${METHODS[@]}"; do
    echo ""
    echo "========================================="
    echo "Testing method: $method"
    echo "========================================="

    for max_words in "${WORD_COUNTS[@]}"; do
        echo ""
        echo "Running $method with max_words=$max_words"

        python -m method.baselines.run_baselines \
            --method "$method" \
            --data_source "$data_source" \
            --dataset "$dataset" \
            --source_model "$source_model" \
            --sampling_model_name "$sampling_model" \
            --scoring_model_name "$scoring_model" \
            --max_samples "$max_samples" \
            --max_words "$max_words"

        echo "Completed $method with max_words=$max_words"
    done

    echo "Completed all word counts for $method"
done

echo ""
echo "========================================="
echo "Text Length Ablation Study Complete!"
echo "========================================="
echo "All results saved to formal_exp/*/maxwords/*/"