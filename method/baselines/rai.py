"""
RAI (Implicit Reward Model) Detection Method

This method computes the difference between base and instruct model log-likelihoods.
Field mapping for RAI:
    - scoring_model -> instruct_model (phi)
    - sampling_model -> base_model (theta)

RAI Score = ll_base - ll_instruct
"""

import torch
import tqdm
import time
from method.utils import setup_seed
from method.core import get_likelihood, compute_logits_cache
from method.metrics import get_roc_metrics, get_precision_recall_metrics


def handle_rai(args, data, n_samples):
    """
    Handle RAI detection

    RAI computes: score = ll_base - ll_instruct

    Args:
        args: Command-line arguments
            - scoring_model: Maps to instruct_model (phi)
            - sampling_model: Maps to base_model (theta)
        data: Dict with 'original' and 'sampled' keys
        n_samples: Number of samples

    Returns:
        Dict with results
    """
    setup_seed(args.seed)

    # Field mapping: scoring->instruct, sampling->base
    instruct_model = args.scoring_model_name
    base_model = args.sampling_model_name

    print("=" * 50)
    print("RAI Detection Method")
    print("=" * 50)
    print(f"Instruct Model (phi): {instruct_model}")
    print(f"Base Model (theta): {base_model}")
    print("Field Mapping: scoring_model->instruct, sampling_model->base")
    print("=" * 50)

    # Stage 1: Compute instruct model logits
    print("\n[Stage 1] Computing instruct model logits...")
    instruct_cache, instruct_time = compute_logits_cache(
        args, data, instruct_model, model_type="instruct"
    )

    # Stage 2: Compute base model logits (if different)
    if base_model != instruct_model:
        print("\n[Stage 2] Computing base model logits...")
        base_cache, base_time = compute_logits_cache(
            args, data, base_model, model_type="base"
        )
    else:
        print("\n[Stage 2] Base model = instruct model, reusing cache...")
        base_cache = instruct_cache
        base_time = 0

    # Stage 3: Compute RAI scores
    print("\n[Stage 3] Computing RAI scores...")
    print("RAI formula: score = ll_base - ll_instruct")

    results = []
    crit_time_start = time.time()

    for idx in tqdm.tqdm(range(n_samples), desc="Computing RAI scores"):
        # Original text (human)
        logits_orig_inst = instruct_cache["original"][idx]["logits"].to(args.device)
        labels_orig_inst = instruct_cache["original"][idx]["labels"].to(args.device)
        logits_orig_base = base_cache["original"][idx]["logits"].to(args.device)
        labels_orig_base = base_cache["original"][idx]["labels"].to(args.device)

        with torch.no_grad():
            ll_human_inst = get_likelihood(logits_orig_inst, labels_orig_inst)
            ll_human_base = get_likelihood(logits_orig_base, labels_orig_base)
            original_crit = ll_human_inst - ll_human_base

        # Sampled text (AI)
        logits_samp_inst = instruct_cache["sampled"][idx]["logits"].to(args.device)
        labels_samp_inst = instruct_cache["sampled"][idx]["labels"].to(args.device)
        logits_samp_base = base_cache["sampled"][idx]["logits"].to(args.device)
        labels_samp_base = base_cache["sampled"][idx]["labels"].to(args.device)

        with torch.no_grad():
            ll_ai_inst = get_likelihood(logits_samp_inst, labels_samp_inst)
            ll_ai_base = get_likelihood(logits_samp_base, labels_samp_base)
            sampled_crit = ll_ai_inst - ll_ai_base

        results.append({
            "original": data["original"][idx],
            "original_crit": original_crit.item() if torch.is_tensor(original_crit) else original_crit,
            "sampled": data["sampled"][idx],
            "sampled_crit": sampled_crit.item() if torch.is_tensor(sampled_crit) else sampled_crit
        })

    crit_time = time.time() - crit_time_start
    total_time = instruct_time + base_time + crit_time
    time_per_sample = total_time / n_samples

    print(f"\n[INFO] instruct time: {instruct_time:.4f}, base time: {base_time:.4f}, "
          f"crit time: {crit_time:.4f}, total time: {total_time:.4f}")
    print(f"[INFO] time per sample: {time_per_sample:.4f}")

    # Compute metrics
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }

    import numpy as np
    print(f"\nReal mean/std: {np.mean(predictions['real']):.4f}/{np.std(predictions['real']):.4f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.4f}/{np.std(predictions['samples']):.4f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    print(f"\nRAI ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Return results in standard format
    results_dict = {
        'name': 'rai',
        'info': {
            'n_samples': n_samples,
            'instruct_model': instruct_model,
            'base_model': base_model
        },
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc,
        'time': {
            'instruct_time': instruct_time,
            'base_time': base_time,
            'crit_time': crit_time,
            'total_time': total_time,
            'n_samples': n_samples,
            'time_per_sample': time_per_sample
        }
    }

    return results_dict
