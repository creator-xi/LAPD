# Unified Baseline Detection Framework

This directory provides a unified interface for running various baseline detection methods for AI-generated text detection.

## Overview

The framework unifies the following baseline methods:

1. **Simple Baselines** (likelihood, rank, logrank, entropy)
   - Single-model methods using statistical measures
   - Fast and simple to compute
   - Good baseline comparisons

2. **Sampling Discrepancy** (sampling_discrepancy)
   - Fast-DetectGPT method
   - Can use Monte Carlo or analytic computation
   - Uses reference and scoring models

3. **DetectGPT** (detectgpt)
   - Perturbation-based detection
   - Uses T5 mask-filling model
   - Computes likelihood differences with perturbations

4. **LastDE** (lastde)
   - Uses fastMDE (fast Multiscale Dimensionality Estimation)
   - Computes LastDE metric with sampling
   - Requires fastMDE library

5. **DNA-DetectLLM** (dna_detectllm)
   - DetectLLM method
   - Uses observer and performer models
   - Computes perplexity/entropy ratios

6. **IRM**

7. **Roberta**

## Usage

### Basic Usage

Run a specific method:

```bash
# Simple baselines
python run_baselines.py --method likelihood --dataset xsum --source_model gpt4o
python run_baselines.py --method rank --dataset xsum --source_model gpt4o
python run_baselines.py --method entropy --dataset xsum --source_model gpt4o

# fast (analytic version)
python scripts/baselines/run_baselines.py --method sampling_discrepancy --dataset xsum --source_model gpt4o \
    --scoring_model_name falcon-7b-instruct --sampling_model_name falcon-7b \
    --discrepancy_analytic

# DetectGPT
python run_baselines.py --method detectgpt --dataset xsum --source_model gpt4o \
    --scoring_model_name gpt2 --mask_filling_model_name t5-small \
    --pct_words_masked 0.3 --n_perturbations 10

# LastDE
python run_baselines.py --method lastde --dataset xsum --source_model gpt4o \
    --reference_model_name falcon --embed_size 4 --epsilon 8 --tau_prime 15

# DNA-GPT
python run_baselines.py --method dna_gpt --dataset xsum --source_model gpt4o
```

### Advanced Usage

#### Custom Model Paths

```bash
# For methods that support custom model paths
python run_baselines.py --method dna_gpt --dataset xsum \
    --observer_model_path /path/to/observer/model \
    --performer_model_path /path/to/performer/model

python run_baselines.py --method lastde --dataset xsum \
    --reference_model_path /path/to/reference/model \
    --scoring_model_path /path/to/scoring/model
```

#### Output Directory

```bash
python run_baselines.py --method sampling_discrepancy \
    --output_file ./results/my_experiment \
    --dataset xsum --source_model gpt4o
```

This will save results to `./results/my_experiment.sampling_discrepancy.json`

#### Limit Number of Samples

```bash
python run_baselines.py --method sampling_discrepancy \
    --max_samples 50 \
    --dataset xsum --source_model gpt4o
```

### Command-Line Options

#### Common Options

- `--method`: Detection method to run (required)
- `--output_file`: Output file prefix (default: `./exp_baselines/results/xsum_gpt4o`)
- `--seed`: Random seed (default: 0)
- `--device`: Device to use (default: `cuda`)
- `--cache_dir`: Model cache directory (default: `./cache`)
- `--max_samples`: Maximum number of samples to use

#### Model Options

- `--scoring_model_name`: Scoring model name (default: `gpt2`)
- `--sampling_model_name`: Sampling model name (default: `gpt2`)

#### DetectGPT-Specific Options

- `--pct_words_masked`: Percentage of words masked (default: 0.3)
- `--mask_top_p`: Top-p for mask filling (default: 1.0)
- `--span_length`: Span length for masking (default: 2)
- `--n_perturbations`: Number of perturbations (default: 10)
- `--mask_filling_model_name`: Mask filling model (default: `t5-small`)

#### Sampling Discrepancy-Specific Options

- `--discrepancy_analytic`: Use analytic version instead of Monte Carlo

#### LastDE-Specific Options

- `--reference_model_name`: Reference model name (default: `falcon`)
- `--embed_size`: Embedding size (default: 4)
- `--epsilon`: Epsilon parameter (default: 8)
- `--tau_prime`: Tau prime parameter (default: 15)
- `--n_samples`: Number of samples (default: 100)

## Data Format

### Standard Format

Most methods expect data in the format:
```python
{
    'original': ['human text 1', 'human text 2', ...],
    'sampled': ['AI text 1', 'AI text 2', ...]
}
```

### DetectGPT Format

DetectGPT expects a JSON file with this structure. Use `--dataset_file` to specify the path.

### LastDE Format

LastDE expects specific JSON files with `human_text` and `machine_text` keys. The framework automatically loads from hardcoded paths (modify in `_lastde_doubleplus.py` if needed).

## Output Format

All methods save results to JSON files with the following structure:

```python
{
    'name': 'method_name',
    'info': {
        'n_samples': 100,
        # method-specific info
    },
    'predictions': {
        'real': [human_scores],
        'samples': [ai_scores]
    },
    'raw_results': [
        {
            'original': 'text',
            'original_crit': score,
            'sampled': 'text',
            'sampled_crit': score
        },
        ...
    ],
    'metrics': {
        'roc_auc': 0.95,
        'fpr': [...],
        'tpr': [...]
    },
    'pr_metrics': {
        'pr_auc': 0.93,
        'precision': [...],
        'recall': [...]
    },
    'loss': 0.07
}
```

## Running Multiple Methods

You can easily run multiple methods by executing the script multiple times:

```bash
# Run a battery of tests
for method in likelihood rank logrank entropy sampling_discrepancy; do
    python run_baselines.py --method $method \
        --dataset xsum --source_model gpt4o \
        --output_file ./results/xsum_gpt4o
done
```

Or use a loop in bash:

```bash
# Run all simple baselines
for method in likelihood rank logrank entropy; do
    echo "Running $method..."
    python run_baselines.py --method $method \
        --dataset xsum --source_model gpt4o \
        --output_file ./results/xsum_gpt4o
done
```

## File Structure

```
scripts/
├── run_baselines.py           # Main entry point
├── _baseline_methods.py       # Simple baselines (likelihood, rank, etc.)
├── _detect_gpt.py            # DetectGPT implementation
├── _fast_detect_gpt_low_gpu.py # Sampling discrepancy
├── _lastde_doubleplus.py     # LastDE implementation
├── _dna_detectllm.py         # DNA-GPT implementation
└── BASELINES_README.md       # This file
```

## Dependencies

All methods require:
- PyTorch
- Transformers
- NumPy
- Scikit-learn

Additional dependencies:
- **LastDE**: `fastMDE` library (install separately)
- **DetectGPT**: No additional dependencies (uses built-in T5)

## Performance Tips

1. **Use analytic version** for sampling discrepancy when possible:
   - Add `--discrepancy_analytic` flag
   - Much faster than Monte Carlo sampling

2. **Cache logits** for sampling discrepancy:
   - The framework automatically caches logits
   - Saves time when running multiple iterations

3. **Limit samples** for quick testing:
   - Use `--max_samples 50` for quick experiments
   - Full dataset for final results

4. **GPU memory**:
   - Methods load models to GPU
   - Framework automatically clears memory between stages
   - Use `--device cuda:0` to specify GPU

## Troubleshooting

### LastDE Method Not Working

If you get an ImportError for `fastMDE`, install it:
```bash
pip install fastmde
# or
pip install git+https://github.com/.../fastmde
```

### Out of Memory Errors

1. Reduce `--max_samples`
2. Use analytic version: `--discrepancy_analytic`
3. Use a smaller model
4. Clear GPU memory: Restart the script

### Model Not Found

Check:
1. Model name spelling
2. Model path exists in `--cache_dir`
3. For LastDE, check hardcoded paths in `_lastde_doubleplus.py`

## Examples

### Example 1: Compare Simple Baselines

```bash
# Run all simple baselines on XSum with GPT-4o generated texts
for method in likelihood rank logrank entropy; do
    python run_baselines.py \
        --method $method \
        --dataset xsum \
        --source_model gpt4o \
        --scoring_model_name gpt2 \
        --output_file ./results/xsum_baselines \
        --max_samples 100
done
```

### Example 2: Sampling Discrepancy with Different Models

```bash
# Base model only
python run_baselines.py \
    --method sampling_discrepancy \
    --dataset xsum \
    --source_model gpt4o \
    --scoring_model_name falcon-7b \
    --sampling_model_name falcon-7b \
    --discrepancy_analytic \
    --output_file ./results/xsum_falcon_base

# Base + Instruct models
python run_baselines.py \
    --method sampling_discrepancy \
    --dataset xsum \
    --source_model gpt4o \
    --scoring_model_name falcon-7b-instruct \
    --sampling_model_name falcon-7b \
    --discrepancy_analytic \
    --output_file ./results/xsum_falcon_combo
```

### Example 3: DetectGPT with Different Perturbation Rates

```bash
# Low perturbation rate
python run_baselines.py \
    --method detectgpt \
    --dataset xsum \
    --source_model gpt4o \
    --pct_words_masked 0.2 \
    --n_perturbations 10 \
    --output_file ./results/xsum_detectgpt_low

# High perturbation rate
python run_baselines.py \
    --method detectgpt \
    --dataset xsum \
    --source_model gpt4o \
    --pct_words_masked 0.5 \
    --n_perturbations 10 \
    --output_file ./results/xsum_detectgpt_high
```

## Notes

- All methods use the same random seed for reproducibility
- Results are automatically saved to JSON files
- Methods handle GPU memory management automatically
- Some methods (LastDE) may require custom paths or additional setup
