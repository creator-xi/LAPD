"""
LastDE (Last Dimensionality Estimation) Detection Method

This method uses fastMDE (fast Multiscale Dimensionality Estimation) to compute
the LastDE metric, which measures the likelihood of text under different models.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import argparse
import json
import time
import os
import warnings
warnings.filterwarnings('ignore')
from .fastMDE import get_tau_multiscale_DE
from method.metrics import get_roc_metrics, get_precision_recall_metrics
from method.model import load_model, load_tokenizer
from method.core import get_samples, get_likelihood_no_mean
from method.utils import setup_seed


def get_lastde(log_likelihood, embed_size, epsilon, tau_prime):
    """
    Compute LastDE (Last Dimensionality Estimation) metric

    Args:
        log_likelihood: Log-likelihood tensor
        embed_size: Embedding size for MDE
        epsilon: Epsilon parameter (percentage of tokens)
        tau_prime: Tau prime parameter

    Returns:
        LastDE value
    """
    seq_length = log_likelihood.shape[1]
    effective_embed_size = min(embed_size, seq_length)
    templl = log_likelihood.mean(dim=1)
    epsilon_tokens = int(epsilon * log_likelihood.shape[1])
    aggmde = get_tau_multiscale_DE(
        ori_data=log_likelihood,
        embed_size=effective_embed_size,
        epsilon=epsilon_tokens,
        tau_prime=min(tau_prime, seq_length)
    )

    # LastDE = templl / aggmde
    lastde = templl / aggmde
    return lastde

def get_sampling_discrepancy(logits_ref, logits_score, labels, args):
    """
    Compute sampling discrepancy with LastDE

    This computes the LastDE metric for both the true sequence and samples,
    then normalizes by the statistics of sampled sequences.
    """
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1

    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    # Generate samples
    samples = get_samples(logits_ref, labels, 100)

    # Compute log-likelihoods
    log_likelihood_x = get_likelihood_no_mean(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood_no_mean(logits_score, samples)

    # Compute LastDE for both
    lastde_x = get_lastde(log_likelihood_x, args.embed_size, args.epsilon, args.tau_prime)
    sampled_lastde = get_lastde(log_likelihood_x_tilde, args.embed_size, args.epsilon, args.tau_prime)

    # Normalize
    miu_tilde = sampled_lastde.mean()
    sigma_tilde = sampled_lastde.std()
    discrepancy = (lastde_x - miu_tilde) / sigma_tilde

    return discrepancy.cpu().item()

def handle_lastde(args, data, n_samples):
    """
    Handle LastDE detection with dual GPU support

    Args:
        args: Command-line arguments
        data: Dict with 'original' and 'sampled' keys
        n_samples: Number of samples

    Returns:
        Dict with results
    """
    DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
    max_length = 1024

    print(f"Using devices: scoring_model -> {DEVICE_1}, sampling_model -> {DEVICE_2}")

    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, DEVICE_1, args.cache_dir)
    scoring_model.eval()

    if args.sampling_model_name != args.scoring_model_name:
        reference_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        reference_model = load_model(args.sampling_model_name, DEVICE_2, args.cache_dir)
        reference_model.eval()
    else:
        reference_tokenizer = scoring_tokenizer
        reference_model = scoring_model

    setup_seed(args.seed)
    results = []
    print(f"Processing {n_samples} samples with LastDE method...")
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc="Computing LastDE criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # Process original text (human)
        if original_text is None or isinstance(original_text, float):
            original_crit = 0.0  # Default value for invalid text
        else:
            # Ensure minimum length for original text
            while len(original_text.split()) < 10:
                original_text = original_text + " " + original_text

            # Original text tokenization on scoring model device
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True,
                                        truncation=True, max_length=max_length+1,
                                        return_token_type_ids=False).to(DEVICE_1)
            labels = tokenized.input_ids[:, 1:].to(DEVICE_2)

            with torch.no_grad():
                # Get logits from scoring model
                logits_score = scoring_model(**tokenized).logits[:, :-1].to(DEVICE_1)

                # Get logits from reference model
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized_ref = reference_tokenizer(original_text, return_tensors="pt", padding=True,
                                                      truncation=True, max_length=max_length+1,
                                                      return_token_type_ids=False).to(DEVICE_2)
                    assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer mismatch"
                    logits_ref = reference_model(**tokenized_ref).logits[:, :-1].to(DEVICE_1)

                labels = labels.to(DEVICE_1)
                original_crit = get_sampling_discrepancy(logits_ref, logits_score, labels, args)

        # Process sampled text (AI)
        if sampled_text is None or isinstance(sampled_text, float):
            sampled_crit = 0.0  # Default value for invalid text
        else:
            # Ensure minimum length for sampled text
            while len(sampled_text.split()) < 10:
                sampled_text = sampled_text + " " + sampled_text

            # Sampled text tokenization on scoring model device
            tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                        truncation=True, max_length=max_length+1,
                                        return_token_type_ids=False).to(DEVICE_1)
            labels = tokenized.input_ids[:, 1:].to(DEVICE_2)

            with torch.no_grad():
                # Get logits from scoring model
                logits_score = scoring_model(**tokenized).logits[:, :-1].to(DEVICE_1)

                # Get logits from reference model
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized_ref = reference_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                                      truncation=True, max_length=max_length+1,
                                                      return_token_type_ids=False).to(DEVICE_2)
                    assert torch.all(tokenized_ref.input_ids[:, 1:] == labels), "Tokenizer mismatch"
                    logits_ref = reference_model(**tokenized_ref).logits[:, :-1].to(DEVICE_1)

                labels = labels.to(DEVICE_1)
                sampled_crit = get_sampling_discrepancy(logits_ref, logits_score, labels, args)

        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })
    total_time = time.time() - total_time_start
    time_per_sample = total_time / n_samples
    print(f"[INFO] total time: {total_time}")
    print(f"[INFO] time per sample: {time_per_sample}")
    # Compute metrics
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"LastDE ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results dict
    results_dict = {
        'name': 'lastde',
        'info': {
            'n_samples': n_samples,
            'embed_size': getattr(args, 'embed_size', 4),
            'epsilon': getattr(args, 'epsilon', 8),
            'tau_prime': getattr(args, 'tau_prime', 15),
            'n_samples_param': 100
        },
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc,
        'time': {
            'total_time': total_time, 
            'n_samples': n_samples, 
            'time_per_sample': time_per_sample
        },
    }

    print(f"Processed {len(results)} samples with LastDE method")

    return results_dict
