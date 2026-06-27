#!/bin/bash
set -e  
export CUDA_VISIBLE_DEVICES=''

DATA_SOURCE=("main" "m4" "detectrl_multidomain" "detectrl_multillm" "raid" "realdet" "text_attack" "base" "detectrl_attack" "raid_attack" "cred")
MAX_SAMPLES=-1

# DATASETS=("xsum" "wp" "arxiv")
# SOURCE_MODELS=("gpt4o" "claude3.7" "gemini2.0")

SAMPLING_MODELS=("llama2-7b")
BASE_MODELS=("llama2-7b")
INSTRUCT_MODELS=("llama2-7b-instruct")

ATTACK_TYPES=("delete" "dipper" "insert" "replace")

# options for raid_attack dataset
RAID_ATTACK_TYPES=("none" "whitespace" "synonym" "perplexity_misspelling" "paraphrase" "homoglyph" "zero_width_space")

# CReD specific configurations
CRED_DOMAINS=("composition" "film_review" "news" "paper" "question_answer" "TC_news")
CRED_MODELS=("claude-3.5-haiku" "deepseek-r1" "deepseek-v3" "doubao-1.5-pro" "gemini-2.5-flash" "gpt-3.5-turbo" "gpt-4o" "qwen-2.5" "qwen-3")

CACHE_DIR="./cache"

echo "Starting LAPD Experiments"
echo "Data source: ${data_source}"
echo "Model Pairs: ${#BASE_MODELS[@]} configurations"
echo "Source Models: ${#SOURCE_MODELS[@]} models"
echo "Max Samples: $MAX_SAMPLES"

for data_source in "${DATA_SOURCE[@]}"; do
    if [ "$data_source" = "m4" ] || [ "$data_source" = "realdet" ] || [[ "$data_source" == "detectrl_multidomain" || "$data_source" == "detectrl_multillm" ]] || [[ "$data_source" == "raid" ]]; then
        total_experiments=$((${#BASE_MODELS[@]}))
        current_exp=0

        echo ""
        echo "Processing Dataset: $data_source"

        for model_idx in "${!BASE_MODELS[@]}"; do
            sampling_model=${SAMPLING_MODELS[$model_idx]}
            base_model=${BASE_MODELS[$model_idx]}
            instruct_model=${INSTRUCT_MODELS[$model_idx]}

            current_exp=$((current_exp + 1))

            echo ""
            echo "------------------------------------------"
            echo "Experiment $current_exp/$total_experiments"
            echo "  Data Source: $data_source"
            echo "  Sampling Model: $sampling_model"
            echo "  Base Model: $base_model"
            echo "  Instruct Model: $instruct_model"
            echo "------------------------------------------"

            python -m method.lapd \
                --data_source "$data_source" \
                --sampling_model "$sampling_model" \
                --base_model "$base_model" \
                --instruct_model "$instruct_model" \
                --max_samples $MAX_SAMPLES \
                --device cuda \
                --cache_dir "$CACHE_DIR"

            echo "Experiment completed for $base_model vs $instruct_model sampled with $sampling_model on $data_source dataset"
            python -c "import torch; torch.cuda.empty_cache()"
        done
    elif [ "$data_source" = "text_attack" ]; then
        total_experiments=$((${#BASE_MODELS[@]} * ${#SOURCE_MODELS[@]} * ${#ATTACK_TYPES[@]}))
        current_exp=0

        for source_idx in "${!SOURCE_MODELS[@]}"; do
            source_model=${SOURCE_MODELS[$source_idx]}

            echo ""
            echo "=========================================="
            echo "Processing Source Model ($((source_idx+1))/${#SOURCE_MODELS[@]}): $source_model"
            echo "=========================================="

            for attack_idx in "${!ATTACK_TYPES[@]}"; do
                attack_type=${ATTACK_TYPES[$attack_idx]}

                echo ""
                echo "--- Processing Attack Type ($((attack_idx+1))/${#ATTACK_TYPES[@]}): $attack_type ---"

                for model_idx in "${!BASE_MODELS[@]}"; do
                    sampling_model=${SAMPLING_MODELS[$model_idx]}
                    base_model=${BASE_MODELS[$model_idx]}
                    instruct_model=${INSTRUCT_MODELS[$model_idx]}

                    current_exp=$((current_exp + 1))

                    echo ""
                    echo "------------------------------------------"
                    echo "Experiment $current_exp/$total_experiments"
                    echo "  Data Source: $data_source"
                    echo "  Source Model: $source_model"
                    echo "  Attack Type: $attack_type"
                    echo "  Sampling Model: $sampling_model"
                    echo "  Base Model: $base_model"
                    echo "  Instruct Model: $instruct_model"
                    echo "------------------------------------------"

                    python -m method.lapd \
                        --data_source "$data_source" \
                        --sampling_model "$sampling_model" \
                        --base_model "$base_model" \
                        --instruct_model "$instruct_model" \
                        --source_model "$source_model" \
                        --attack_type "$attack_type" \
                        --max_samples $MAX_SAMPLES \
                        --device cuda \
                        --cache_dir "$CACHE_DIR"

                    echo "Experiment completed for $source_model + $attack_type with $base_model vs $instruct_model sampled with $sampling_model on $data_source dataset"
                    python -c "import torch; torch.cuda.empty_cache()"
                done
            done
        done
    elif [ "$data_source" = "main" ]; then
        total_experiments=$((${#BASE_MODELS[@]} * ${#SOURCE_MODELS[@]} * ${#DATASETS[@]}))
        current_exp=0

        for dataset_idx in "${!DATASETS[@]}"; do
            dataset=${DATASETS[$dataset_idx]}

            echo ""
            echo "=========================================="
            echo "Processing Dataset ($((dataset_idx+1))/${#DATASETS[@]}): $dataset"
            echo "=========================================="

            for source_idx in "${!SOURCE_MODELS[@]}"; do
                source_model=${SOURCE_MODELS[$source_idx]}

                echo ""
                echo "--- Processing Source Model: $source_model ---"

                for model_idx in "${!BASE_MODELS[@]}"; do
                    sampling_model=${SAMPLING_MODELS[$model_idx]}
                    base_model=${BASE_MODELS[$model_idx]}
                    instruct_model=${INSTRUCT_MODELS[$model_idx]}

                    current_exp=$((current_exp + 1))

                    echo ""
                    echo "------------------------------------------"
                    echo "Experiment $current_exp/$total_experiments"
                    echo "  Dataset: $dataset"
                    echo "  Source Model: $source_model"
                    echo "  Sampling Model: $sampling_model"
                    echo "  Base Model: $base_model"
                    echo "  Instruct Model: $instruct_model"
                    echo "------------------------------------------"

                    python -m method.lapd \
                        --data_source "$data_source" \
                        --sampling_model "$sampling_model" \
                        --base_model "$base_model" \
                        --instruct_model "$instruct_model" \
                        --dataset "$dataset" \
                        --source_model "$source_model" \
                        --max_samples $MAX_SAMPLES \
                        --device cuda \
                        --cache_dir "$CACHE_DIR"

                    echo "Experiment completed for $base_model vs $instruct_model sampled with $sampling_model on $dataset dataset with $source_model"
                    python -c "import torch; torch.cuda.empty_cache()"
                done
            done
        done

    elif [ "$data_source" = "raid_attack" ]; then
        total_experiments=$((${#BASE_MODELS[@]} * ${#RAID_ATTACK_TYPES[@]}))
        current_exp=0

        for raid_attack_idx in "${!RAID_ATTACK_TYPES[@]}"; do
            raid_attack_type=${RAID_ATTACK_TYPES[$raid_attack_idx]}

            echo ""
            echo "=========================================="
            echo "Processing RAID Attack Type ($((raid_attack_idx+1))/${#RAID_ATTACK_TYPES[@]}): $raid_attack_type"
            echo "=========================================="

            for model_idx in "${!BASE_MODELS[@]}"; do
                sampling_model=${SAMPLING_MODELS[$model_idx]}
                base_model=${BASE_MODELS[$model_idx]}
                instruct_model=${INSTRUCT_MODELS[$model_idx]}

                current_exp=$((current_exp + 1))

                echo ""
                echo "------------------------------------------"
                echo "Experiment $current_exp/$total_experiments"
                echo "  Data Source: $data_source"
                echo "  RAID Attack Type: $raid_attack_type"
                echo "  Sampling Model: $sampling_model"
                echo "  Base Model: $base_model"
                echo "  Instruct Model: $instruct_model"
                echo "------------------------------------------"

                python -m method.lapd \
                    --data_source "$data_source" \
                    --sampling_model "$sampling_model" \
                    --base_model "$base_model" \
                    --instruct_model "$instruct_model" \
                    --raid_attack_type "$raid_attack_type" \
                    --max_samples $MAX_SAMPLES \
                    --device cuda \
                    --cache_dir "$CACHE_DIR"

                echo "Experiment completed for raid_attack=$raid_attack_type with $base_model vs $instruct_model sampled with $sampling_model"
                python -c "import torch; torch.cuda.empty_cache()"
            done
        done
    elif [ "$data_source" = "cred" ]; then
        total_experiments=$((${#BASE_MODELS[@]} * ${#CRED_DOMAINS[@]} * ${#CRED_MODELS[@]}))
        current_exp=0

        for domain_idx in "${!CRED_DOMAINS[@]}"; do
            cred_domain=${CRED_DOMAINS[$domain_idx]}

            echo ""
            echo "=========================================="
            echo "Processing CReD Domain ($((domain_idx+1))/${#CRED_DOMAINS[@]}): $cred_domain"
            echo "=========================================="

            for model_idx in "${!CRED_MODELS[@]}"; do
                cred_model=${CRED_MODELS[$model_idx]}

                echo ""
                echo "--- Processing CReD Model ($((model_idx+1))/${#CRED_MODELS[@]}): $cred_model ---"

                for detect_model_idx in "${!BASE_MODELS[@]}"; do
                    sampling_model=${SAMPLING_MODELS[$detect_model_idx]}
                    base_model=${BASE_MODELS[$detect_model_idx]}
                    instruct_model=${INSTRUCT_MODELS[$detect_model_idx]}

                    current_exp=$((current_exp + 1))

                    echo ""
                    echo "------------------------------------------"
                    echo "Experiment $current_exp/$total_experiments"
                    echo "  CReD Domain: $cred_domain"
                    echo "  CReD Model: $cred_model"
                    echo "  Sampling Model: $sampling_model"
                    echo "  Base Model: $base_model"
                    echo "  Instruct Model: $instruct_model"
                    echo "------------------------------------------"

                    python -m method.lapd \
                        --data_source "$data_source" \
                        --sampling_model "$sampling_model" \
                        --base_model "$base_model" \
                        --instruct_model "$instruct_model" \
                        --cred_domain "$cred_domain" \
                        --cred_model "$cred_model" \
                        --max_samples $MAX_SAMPLES \
                        --device cuda \
                        --cache_dir "$CACHE_DIR"

                    echo "Experiment completed for cred=$cred_domain/$cred_model with $base_model vs $instruct_model sampled with $sampling_model"
                    python -c "import torch; torch.cuda.empty_cache()"
                done
            done
        done
    fi
done

echo "All LAPD experiments completed!"
