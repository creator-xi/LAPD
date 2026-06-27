"""
Fast-DetectGPT Sampling Discrepancy Method

This method computes the sampling discrepancy between a scoring model and a sampling model.
It can use either Monte Carlo sampling or an analytic approximation.
"""

import torch
import tqdm
import numpy as np
import time
from method.utils import setup_seed
from method.core import get_samples, get_likelihood, compute_logits_cache
from method.metrics import get_roc_metrics, get_precision_recall_metrics

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    """
    Compute sampling discrepancy using Monte Carlo sampling

    The discrepancy is computed as: (ll - mean(ll_sampled)) / std(ll_sampled)
    where ll_sampled are log-likelihoods of samples from the reference distribution
    under the scoring model.

    Args:
        logits_ref: Logits from reference/sampling model
        logits_score: Logits from scoring model
        labels: Ground truth labels

    Returns:
        Tuple of (discrepancy, ll, mean, std)
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    # Generate samples from reference distribution
    samples = get_samples(logits_ref, labels)

    # Compute log-likelihoods
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)

    # Compute statistics
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde

    return (discrepancy.item(),
            log_likelihood_x.mean().item(),
            miu_tilde.mean().item(),
            sigma_tilde.mean().item())

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    """
    Compute sampling discrepancy analytically (no sampling needed)

    This is mathematically equivalent to the Monte Carlo version but computes
    the expectation and variance analytically.

    Args:
        logits_ref: Logits from reference/sampling model
        logits_score: Logits from scoring model
        labels: Ground truth labels

    Returns:
        Tuple of (discrepancy, ll, mean, std)
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = torch.log_softmax(logits_score, dim=-1)
    probs_ref = torch.softmax(logits_ref, dim=-1)

    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)

    discrepancy = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
    discrepancy = discrepancy.mean()

    return (discrepancy.item(),
            log_likelihood.mean().item(),
            mean_ref.mean().item(),
            var_ref.mean().sqrt().item())

def handle_fast(args, data, n_samples):
    """
    Handle sampling discrepancy detection

    Args:
        args: Command-line arguments
        data: Dict with 'original' and 'sampled' keys
        n_samples: Number of samples

    Returns:
        Dict with results
    """
    setup_seed(args.seed)
    # Choose criterion function
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy

    # Stage 1: Load scoring model and compute logits
    print("=" * 50)
    print("Stage 1: Computing scoring model logits...")
    print("=" * 50)

    scoring_cache, scoring_time = compute_logits_cache(args, data, args.scoring_model_name, model_type="scoring")

    sampling_time = 0
    # Stage 2: Load sampling model and compute logits (if different from scoring)
    if args.sampling_model_name != args.scoring_model_name:
        print("=" * 50)
        print("Stage 2: Computing sampling model logits...")
        print("=" * 50)
        sampling_cache, sampling_time = compute_logits_cache(args, data, args.sampling_model_name, model_type="sampling")
    else:
        sampling_cache = scoring_cache

    # Stage 3: Compute discrepancy scores using cached logits
    print("=" * 50)
    print("Stage 3: Computing discrepancy scores...")
    print("=" * 50)

    results = []
    crit_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        # Get logits from cache
        logits_score_orig = scoring_cache["original"][idx]["logits"].to(args.device)
        labels_orig = scoring_cache["original"][idx]["labels"].to(args.device)
        logits_ref_orig = sampling_cache["original"][idx]["logits"].to(args.device)

        logits_score_samp = scoring_cache["sampled"][idx]["logits"].to(args.device)
        labels_samp = scoring_cache["sampled"][idx]["labels"].to(args.device)
        logits_ref_samp = sampling_cache["sampled"][idx]["logits"].to(args.device)

        with torch.no_grad():
            # Original text
            original_crit, human_x, human_miu, human_sigma = criterion_fn(
                logits_ref_orig, logits_score_orig, labels_orig)

            # Sampled text
            sampled_crit, ai_x, ai_miu, ai_sigma = criterion_fn(
                logits_ref_samp, logits_score_samp, labels_samp)

        results.append({
            "original": data["original"][idx],
            "original_crit": original_crit,
            "sampled": data["sampled"][idx],
            "sampled_crit": sampled_crit
        })
    crit_time = time.time() - crit_time_start
    total_time = scoring_time + sampling_time + crit_time
    time_per_sample = total_time / n_samples
    print(f"[INFO] scoring time: {scoring_time:.4f}, sampling time: {sampling_time:.4f}, crit time: {crit_time:.4f}, total time: {total_time:.4f}")
    print(f"[INFO] time per sample of {args.method} method: {time_per_sample:.4f}")
    # Compute metrics
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }

    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results
    results_dict = {
        'name': f'{name}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc,
        'time': {
            'scoring_time': scoring_time,
            'sampling_time': sampling_time,
            'crit_time': crit_time,
            'total_time': total_time,
            'n_samples': n_samples,
            'time_per_sample': time_per_sample
        }
    }

    return results_dict
