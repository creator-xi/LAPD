#!/bin/bash
export CUDA_VISIBLE_DEVICES=''

# DATA_SOURCES=("main" "detectrl_multidomain" "detectrl_multillm" "raid" "text_attack" "m4" "realdet" "detectrl_attack" "raid_attack" "base")
DATA_SOURCES=("detectrl_multidomain" "detectrl_multillm" "raid" "m4" "realdet")
MAX_SAMPLES=-1

# Options: likelihood, logrank, entropy, fast, fast_origin, detectgpt, lastde, binoculars, dna_detectllm, roberta_base, roberta_large, rai
METHODS=("dna_detectllm" "binoculars" "lastde" "fast" "entropy" "logrank" "likelihood")

DATASETS=("xsum" "wp" "arxiv")
SOURCE_MODELS=("gpt4o" "claude3.7" "gemini2.0")


ATTACK_TYPES=("delete" "dipper" "insert" "replace")

# options for raid_attack dataset
RAID_ATTACK_TYPES=("none" "whitespace" "synonym" "perplexity_misspelling" "paraphrase" "homoglyph" "zero_width_space")

SCORING_MODELS=("llama2-7b-instruct" "falcon-7b-instruct")
SAMPLING_MODELS=("llama2-7b" "falcon-7b")

echo "Starting Baseline Experiments"
echo "Data Sources: ${DATA_SOURCES[*]}"
echo "Methods: ${METHODS[*]}"
echo "Max Samples: $MAX_SAMPLES"

# 1. data source
for DATA_SOURCE in "${DATA_SOURCES[@]}"; do
    echo ""
    echo "Processing Data Source: $DATA_SOURCE"

    # 2. method
    for method in "${METHODS[@]}"; do
        echo ""
        echo "Method: $method"

        if [ "$DATA_SOURCE" = "m4" ] || [ "$DATA_SOURCE" = "realdet" ] || [[ "$DATA_SOURCE" == "detectrl_multidomain" || "$DATA_SOURCE" == "detectrl_multillm" ]] || [[ "$DATA_SOURCE" == "raid" ]]; then
            # M4 / RealDet / DetectRL
            for idx in "${!SCORING_MODELS[@]}"; do
                scoring_model=${SCORING_MODELS[$idx]}
                sampling_model=${SAMPLING_MODELS[$idx]}
                echo "Running: Method=$method | DataSource=$DATA_SOURCE | Scoring=$scoring_model"
                python -m method.baselines.run_baselines \
                    --method "$method" \
                    --data_source "$DATA_SOURCE" \
                    --scoring_model_name "$scoring_model" \
                    --sampling_model_name "$sampling_model" \
                    --max_samples "$MAX_SAMPLES"
            done

        elif [ "$DATA_SOURCE" = "text_attack" ]; then
            # Text Attack
            for source_model in "${SOURCE_MODELS[@]}"; do
                for attack_type in "${ATTACK_TYPES[@]}"; do
                    for idx in "${!SCORING_MODELS[@]}"; do
                        scoring_model=${SCORING_MODELS[$idx]}
                        sampling_model=${SAMPLING_MODELS[$idx]}
                        echo "Running: Method=$method | Source=$source_model | Attack=$attack_type | Scoring=$scoring_model"
                        python -m method.baselines.run_baselines \
                            --method "$method" \
                            --data_source "$DATA_SOURCE" \
                            --source_model "$source_model" \
                            --attack_type "$attack_type" \
                            --scoring_model_name "$scoring_model" \
                            --sampling_model_name "$sampling_model" \
                            --max_samples "$MAX_SAMPLES"
                    done
                done
            done
        elif [ "$DATA_SOURCE" = "raid_attack" ]; then
            # RAID Attack
            for raid_attack_type in "${RAID_ATTACK_TYPES[@]}"; do
                for idx in "${!SCORING_MODELS[@]}"; do
                    scoring_model=${SCORING_MODELS[$idx]}
                    sampling_model=${SAMPLING_MODELS[$idx]}
                    echo "Running: Method=$method | RAID Attack=$raid_attack_type | Scoring=$scoring_model"
                    python -m method.baselines.run_baselines \
                        --method "$method" \
                        --data_source "$DATA_SOURCE" \
                        --raid_attack_type "$raid_attack_type" \
                        --scoring_model_name "$scoring_model" \
                        --sampling_model_name "$sampling_model" \
                        --max_samples "$MAX_SAMPLES"
                done
            done
        else
            # main
            for dataset in "${DATASETS[@]}"; do
                for source_model in "${SOURCE_MODELS[@]}"; do
                    for idx in "${!SCORING_MODELS[@]}"; do
                        scoring_model=${SCORING_MODELS[$idx]}
                        sampling_model=${SAMPLING_MODELS[$idx]}
                        echo "Running: Method=$method | Dataset=$dataset | Source=$source_model | Scoring=$scoring_model"
                        python -m method.baselines.run_baselines \
                            --method "$method" \
                            --data_source "$DATA_SOURCE" \
                            --dataset "$dataset" \
                            --source_model "$source_model" \
                            --scoring_model_name "$scoring_model" \
                            --sampling_model_name "$sampling_model" \
                            --max_samples "$MAX_SAMPLES"
                    done
                done
            done
        fi
    done
done

echo "All experiments completed."