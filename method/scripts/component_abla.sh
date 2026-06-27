#!/bin/bash

export CUDA_VISIBLE_DEVICES=''

DATA_SOURCES=("detectrl_multidomain" "detectrl_multillm" "raid" "m4" "realdet")

# DATASETS=("xsum" "wp" "arxiv")
# SOURCE_MODELS=("gpt4o" "claude3.7" "gemini2.0")

SAMPLING_MODELS=("llama2-7b")
BASE_MODELS=("llama2-7b")
INSTRUCT_MODELS=("llama2-7b-instruct")

COMPARISON_METHODS=("base" "base-inst" "d_base-inst" "inst_mul_base-inst")

MAX_SAMPLES=-1
CACHE_DIR="./cache"

echo "=========================================="
echo "Comparison Methods Experiment"
echo "Comparison Methods: ${#COMPARISON_METHODS[@]} methods (${COMPARISON_METHODS[*]})"
echo "Max Samples: $MAX_SAMPLES"
echo "=========================================="

for comp_idx in "${!COMPARISON_METHODS[@]}"; do
    comparison_method=${COMPARISON_METHODS[$comp_idx]}

    for data_source in "${DATA_SOURCES[@]}"; do
        # benchmarks
        if [ "$data_source" = "m4" ] || [ "$data_source" = "realdet" ] || [[ "$data_source" == "detectrl_multidomain" || "$data_source" == "detectrl_multillm" || "$data_source" == "raid" ]]; then
            total_experiments=$((${#COMPARISON_METHODS[@]} * ${#BASE_MODELS[@]}))
            current_exp=0

            echo ""
            echo "=========================================="
            echo "Processing Dataset: $data_source"
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
                echo "  Sampling Model: $sampling_model"
                echo "  Base Model: $base_model"
                echo "  Instruct Model: $instruct_model"
                echo "------------------------------------------"

                python -m method.component_abla \
                    --data_source "$data_source" \
                    --sampling_model "$sampling_model" \
                    --base_model "$base_model" \
                    --instruct_model "$instruct_model" \
                    --max_samples $MAX_SAMPLES \
                    --device cuda \
                    --cache_dir "$CACHE_DIR" \
                    --comparison_method "$comparison_method"

                echo "Experiment completed for $base_model vs $instruct_model sampled with $sampling_model on $data_source dataset"
                python -c "import torch; torch.cuda.empty_cache()"
            done
        else
            # datasets with single domain each
            total_experiments=$((${#COMPARISON_METHODS[@]} * ${#BASE_MODELS[@]} * ${#SOURCE_MODELS[@]} * ${#DATASETS[@]}))
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

                        python -m method.component_abla \
                            --data_source "$data_source" \
                            --sampling_model "$sampling_model" \
                            --base_model "$base_model" \
                            --instruct_model "$instruct_model" \
                            --dataset "$dataset" \
                            --source_model "$source_model" \
                            --max_samples $MAX_SAMPLES \
                            --device cuda \
                            --cache_dir "$CACHE_DIR" \
                            --comparison_method "$comparison_method"

                        echo "Experiment completed for $base_model vs $instruct_model sampled with $sampling_model on $dataset dataset with $source_model"
                        python -c "import torch; torch.cuda.empty_cache()"
                    done
                done
            done
        fi
    done
done

echo ""
echo "=========================================="
echo "All Comparison Methods experiments completed!"
echo "=========================================="
