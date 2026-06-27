**English** | [简体中文](README_zh.md)

# LAPD

This repository contains the code and data for the paper **"Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy"**.

We provide LAPD and RAI, together with a set of zero-shot baselines (fast-detectgpt, lastde++, binoculars, dna-detectllm, ...) in a unified framework, and benchmark data covering M4, DetectRL, RAID, RealDet, XSum, WritingPrompts, and CReD.

**Paper**: [arXiv:2604.16923](https://arxiv.org/abs/2604.16923) | **Code**: [creator-xi/LAPD](https://github.com/creator-xi/LAPD)

- 4/18/2026: The LAPD preprint is available on arXiv.
- 6/27/2026: This repository provides the LAPD implementation, baseline runners, benchmark data, robustness evaluation scripts, and ablation utilities.

## Brief Introduction

Detecting AI-generated text is difficult because modern LLMs produce fluent, coherent, and increasingly human-like passages. Traditional likelihood-based detectors are often confounded by intrinsic content complexity: a simple human sentence can receive higher likelihood than a complex AI-generated paragraph.

LAPD starts from a different observation: modern user-facing LLMs are aligned through supervised fine-tuning and preference tuning. This alignment process leaves a measurable distributional imprint. By comparing the log-likelihood assigned by an aligned model and its corresponding base model, we can capture how much extra probability a text receives from alignment-induced preferences rather than from general language modeling alone.

The paper derives this signal as **Alignment Imprint** and proposes **Log-likelihood Alignment Preference Discrepancy (LAPD)**, a standardized information-weighted statistic that improves robustness in high-entropy regions.

| Method | Main Benchmarks AUROC | Cross-Domain AUROC | TPR @ 0.5% FPR | Time / Text |
| --- | ---: | ---: | ---: | ---: |
| Fast-DetectGPT | 82.26 | 95.36 | 51.79 | 0.5578s |
| DNA-DetectLLM | 86.12 | 98.36 | 59.56 | 0.7696s |
| RAI | 85.60 | 95.69 | 69.92 | 0.3506s |
| LAPD | **92.37** | **99.49** | **92.27** | 0.5792s |

The main benchmark results are averaged over M4, DetectRL Multi-LLM, DetectRL Multi-Domain, RAID, and RealDet. Cross-domain results are averaged over XSum, WritingPrompts, and Arxiv with GPT-4 Turbo, Gemini-2.0 Flash, and Claude-3.7 Sonnet generations.

## Project Overview

LAPD is a zero-shot detector. It does not train a classifier on labeled AI/human examples. Given an input text, it:

1. Computes token-level likelihoods under a base model.
2. Computes token-level likelihoods under the corresponding aligned or instruction-tuned model.
3. Forms the Raw Alignment Imprint (RAI) from the base-aligned likelihood discrepancy.
4. Applies information weighting and perturbation-based standardization to obtain the LAPD score.

The implementation supports:

- LAPD and RAI evaluation.
- Standard zero-shot baselines including likelihood, entropy, LogRank, DetectGPT, Fast-DetectGPT, Lastde++, Binoculars, DNA-DetectLLM, RoBERTa baselines, and IRM.
- Main benchmark evaluation on M4, DetectRL, RAID, RealDet, XSum, WritingPrompts, Arxiv, text-attack data, and CReD.
- Ablations for LAPD components, text length, low-FPR behavior, time efficiency, and model-pair sensitivity.

## Environment

Create a Python 3.10 environment first and then install the requirements.

```bash
conda create -n lapd python=3.10
conda activate lapd
pip install -r requirements.txt
```

After installation, verify the environment:

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"
python -c "import method.lapd; print('lapd import ok')"
```

Models are loaded through `method/model.py`. The default cache directory is `./cache`. Local model aliases are also defined in `method/model.py`; for example, `llama2-7b` maps to a local Llama-2-7B checkpoint under the cache directory.

Recommended default models:

- Sampling model: `llama2-7b`
- Base model: `llama2-7b`
- Aligned model: `llama2-7b-instruct`

## Running Guide

Run a small LAPD evaluation from the repository root:

```bash
python -m method.lapd \
  --data_source main \
  --dataset xsum \
  --source_model gpt4o \
  --sampling_model llama2-7b \
  --base_model llama2-7b \
  --instruct_model llama2-7b-instruct \
  --max_samples 10 \
  --device cuda \
  --cache_dir ./cache
```

The results will be saved under:

```text
./formal_exp/lapd/
```

Run the full LAPD batch script:

```bash
./method/scripts/run_lapd.sh
```

Run baseline methods:

```bash
./method/scripts/run_baselines
```

Available baseline methods include:

```text
likelihood, logrank, entropy, fast, fast_origin, detectgpt,
lastde, dna_detectllm, binoculars, binoculars_zscore,
roberta_base, roberta_large, rai, rai_double_gpu
```

Useful experiment scripts:

```bash
# LAPD main evaluation
bash method/scripts/run_lapd.sh

# Baseline evaluation
bash method/scripts/run_baselines.sh

# Component ablation
bash method/scripts/component_abla.sh

# Text-length ablation
bash method/scripts/ablation_text_length.sh

# Time-efficiency evaluation
bash method/scripts/time_efficiency.sh
```

## Recommended Directory Structure

```text
LAPD/
├── cache/                     # model checkpoints
│   ├── llama-2-7b/
│   ├── llama-2-7b-instruct/
│   ├── falcon-7b/
│   └── falcon-7b-instruct/
├── data/
│   ├── Collected data/        # XSum, WritingPrompts, Arxiv data
│   ├── M4/                    # M4 benchmark
│   ├── DetectRL/              # DetectRL benchmark
│   ├── RAID/                  # RAID benchmark
│   ├── RealDet/               # RealDet benchmark
│   ├── Text_attack/           # Random and realistic attack data
│   └── CReD/                  # Chinese benchmark data
├── method/
│   ├── lapd.py                # Main LAPD entry point
│   ├── lapd_multi_gpu.py      # Multi-GPU LAPD variant
│   ├── component_abla.py      # Component ablation entry point
│   ├── args.py                # Centralized argument parser
│   ├── model.py               # Model and tokenizer loading
│   ├── core/                  # Core likelihood and discrepancy computation
│   ├── dataloader/            # Unified benchmark loaders
│   ├── baselines/             # Baseline detector implementations
│   ├── scripts/               # Batch experiment scripts
│   └── utils/                 # Utilities for saving, seeding, and processing
├── formal_exp/                # Saved experiment outputs
├── LAPD.pdf                   # Paper draft
├── README.md
└── README_zh.md
```

Please download detector models into `./cache`. For local model aliases in `method/model.py`, the expected directory names are:

```text
llama2-7b          -> ./cache/llama-2-7b
llama2-7b-instruct -> ./cache/llama-2-7b-instruct
llama3.1-8b        -> ./cache/Meta-Llama-3.1-8B
llama3.1-8b-instruct -> ./cache/Meta-Llama-3.1-8B-Instruct
gpt-j-6b           -> ./cache/gpt-j-6b
gpt-j-6b-instruct  -> ./cache/gpt-j-6b-instruct
```

You can override the data and output roots with:

```bash
export DATA_ROOT=/path/to/data
export FORMAL_EXP_ROOT=/path/to/outputs
```

## Citation

If you find this work useful, please cite:

```bibtex
@article{wu2026alignment,
  title={Alignment Imprint: Zero-Shot AI-Generated Text Detection via Provable Preference Discrepancy},
  author={Wu, Junxi and Huang, Kailin and Hu, Dongjian and Chen, Bin and Wu, Hao and Xia, Shu-Tao and Zou, Changliang},
  journal={arXiv preprint arXiv:2604.16923},
  year={2026}
}
```

## Reference Repositories

This project borrows ideas, baselines, or evaluation conventions from the following works:

- [Fast-DetectGPT](https://github.com/baoguangsheng/fast-detect-gpt)
- [Binoculars](https://arxiv.org/abs/2401.12070)
- [DNA-DetectLLM](https://github.com/Xiaoweizhu57/DNA-DetectLLM)
