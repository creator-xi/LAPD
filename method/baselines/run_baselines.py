#!/usr/bin/env python3
"""
Unified Baseline Detection Framework

This script provides a unified interface for running various baseline detection methods:
- likelihood, rank, logrank, entropy: Simple statistical methods
- fast: Fast-DetectGPT sampling discrepancy method
- detectgpt: DetectGPT perturbation-based method
- lastde: LastDE method with fastMDE
- dna_detectllm: DNA-GPT method

Usage:
    python run_baselines.py --method fast --dataset xsum --source_model gpt4o
"""

import os
import sys
import argparse
import json
import random
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from method.model import load_tokenizer, load_model, get_model_fullname
from method.metrics import get_roc_metrics, get_precision_recall_metrics
from method.utils import exp_root_baseline, save_results, setup_seed
from method.args import create_baselines_parser
from method.dataloader import load_data

def handle_methods(args, data, n_samples):
    # Import method-specific handlers
    from .baseline_methods import handle_simple_baselines
    from .detect_gpt import handle_detectgpt
    from .fast_detect_gpt import handle_fast
    from .fast_detect_gpt_double_gpu import handle_fast_origin
    from .lastde.lastde_doubleplus import handle_lastde
    from .binoculars.detector import handle_binoculars
    from .dna_detectllm.detector import handle_dna_detectllm
    from .roberta import handle_roberta
    from .rai import handle_rai
    from .rai_double_gpu import handle_rai_double_gpu

    criterion_fn = {
        'likelihood': handle_simple_baselines,
        'logrank': handle_simple_baselines,
        'entropy': handle_simple_baselines,
        'fast': handle_fast,
        'fast_origin': handle_fast_origin,  # 2 gpu usage with same results as 'fast'
        'detectgpt': handle_detectgpt,
        'lastde': handle_lastde,
        'binoculars': handle_binoculars,
        'binoculars_zscore': handle_binoculars,
        'dna_detectllm': handle_dna_detectllm,
        'roberta_base': handle_roberta,
        'roberta_large': handle_roberta,
        'rai': handle_rai,
        'rai_double_gpu': handle_rai_double_gpu  # 2 gpu usage with same results as 'rai', for speed testing
    }

    results = criterion_fn[args.method](args, data, n_samples)
    return results

def main():
    """Main entry point"""
    parser = create_baselines_parser()
    args = parser.parse_args()
    # Load data
    data, n_samples = load_data(args)
    if n_samples == 0:
        print(f"[INFO] No data found for data_source={args.data_source}. skip")
        return

    setup_seed(args.seed)
    args.output_dir = exp_root_baseline(args, args.method)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Running baseline method: {args.method.upper()}")
    try:
        # Dispatch to appropriate handler
        results = handle_methods(args, data, n_samples)
        save_results(args, results, args.method)

        # Print summary
        if 'roc_auc' in results.get('metrics', {}):
            roc_auc = results['metrics']['roc_auc']
            pr_auc = results['pr_metrics']['pr_auc']
            print(f"\n{'='*80}")
            print(f"FINAL RESULTS - {args.method.upper()}")
            print(f"ROC AUC: {roc_auc:.4f}")
            print(f"PR AUC:  {pr_auc:.4f}")

    except Exception as e:
        print(f"Error running {args.method}: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == '__main__':
    main()
