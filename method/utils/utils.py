from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import torch
import random
import json
from method.core.agg_strategy import sample_ll_add, token_level_sample_ll_mul

def clear_gpu_memory():
    gc.collect()
    torch.cuda.empty_cache()

def plot_results(fpr, tpr, roc_auc, output_path=None):
    """Plot and optionally save a ROC curve."""
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, bbox_inches="tight")
    return plt.gcf()

FORMAL_EXP_ROOT = os.environ.get("FORMAL_EXP_ROOT", "./formal_exp")

def exp_root(args, method):
    base_dir = os.path.join(FORMAL_EXP_ROOT, method, args.data_source)

    # Add max_words subdirectory if specified for ablation study
    if hasattr(args, 'max_words') and args.max_words is not None:
        max_words_subdir = os.path.join("maxwords", str(args.max_words))
    else:
        max_words_subdir = None

    if args.data_source == 'main':
        base_path = os.path.join(base_dir, f"{args.dataset}_{args.source_model}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'text_attack':
        base_path = os.path.join(base_dir, f"{args.source_model}_{args.attack_type}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'raid_attack':
        base_path = os.path.join(base_dir, args.raid_attack_type, f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'cred':
        base_path = os.path.join(base_dir, f"{args.cred_domain}_{args.cred_model}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    else:
        base_path = os.path.join(base_dir, f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")

    # Append max_words subdirectory if needed
    if max_words_subdir:
        output_dir = os.path.join(base_path, max_words_subdir)
    else:
        output_dir = base_path

    return output_dir

def exp_root_baseline(args, method):
    base_dir = os.path.join(FORMAL_EXP_ROOT, method, args.data_source)
    suffix = f"{args.sampling_model_name}_{args.scoring_model_name}"

    # Add max_words subdirectory if specified for ablation study
    if hasattr(args, 'max_words') and args.max_words is not None:
        max_words_subdir = os.path.join("maxwords", str(args.max_words))
    else:
        max_words_subdir = None

    if args.data_source == 'main':
        base_path = os.path.join(base_dir, f"{args.dataset}_{args.source_model}", suffix)
    elif args.data_source == 'text_attack':
        base_path = os.path.join(base_dir, f"{args.source_model}_{args.attack_type}", suffix)
    elif args.data_source == 'raid_attack':
        base_path = os.path.join(base_dir, args.raid_attack_type, suffix)
    elif args.data_source == 'cred':
        base_path = os.path.join(base_dir, f"{args.cred_domain}_{args.cred_model}", suffix)
    else:
        base_path = os.path.join(base_dir, suffix)

    # Append max_words subdirectory if needed
    if max_words_subdir:
        output_dir = os.path.join(base_path, max_words_subdir)
    else:
        output_dir = base_path

    return output_dir


def save_results(args, results: Dict, method: str):
    """Save results to JSON file"""
    if hasattr(args, 'time_compute') and args.time_compute:
        output_file = os.path.join(args.output_dir, f"{method}_time_compute.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        time_dict = results['time']
        with open(output_file, 'w') as fout:
            json.dump(time_dict, fout, indent=4)
    else:
        output_file = os.path.join(args.output_dir, f"{method}.json")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w') as fout:
            json.dump(results, fout, indent=4)
    print(f"Results saved to {output_file}")

def setup_seed(seed):
    """Set random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def choose_agg_strategy(args):
    if hasattr(args, 'add') and args.add:
        return sample_ll_add
    return token_level_sample_ll_mul
