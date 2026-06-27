import torch
import numpy as np
from typing import Dict, Tuple, List
from .compute import get_likelihood, get_samples


def compute_base_method(
    logits_base: torch.Tensor,
    labels_base: torch.Tensor
) -> float:
    """Method 1: base - using base model log-likelihood"""
    return get_likelihood(logits_base, labels_base).item()


def compute_base_inst_method(
    logits_base: torch.Tensor,
    logits_inst: torch.Tensor,
    labels_base: torch.Tensor,
    labels_inst: torch.Tensor
) -> float:
    """Method 2: base - inst - difference between base and instruct model log-likelihoods"""
    ll_base = get_likelihood(logits_base, labels_base)
    ll_inst = get_likelihood(logits_inst, labels_inst)
    diff = ll_base - ll_inst
    return -(diff).item()  # Flip sign so higher scores indicate AI text


def compute_d_base_inst_method(
    args,
    logits_ref: torch.Tensor,  # from sampling model for normalization
    logits_base: torch.Tensor,
    logits_inst: torch.Tensor,
    labels_base: torch.Tensor,
    labels_inst: torch.Tensor
) -> float:
    """Method 3: d(base - inst) - normalized difference using sampling distribution"""
    ll_base = get_likelihood(logits_base, labels_base)
    ll_inst = get_likelihood(logits_inst, labels_inst)
    diff = ll_base - ll_inst

    samples_ref = get_samples(logits_ref, labels_base)

    ll_base_sample = get_likelihood(logits_base, samples_ref)
    ll_inst_sample = get_likelihood(logits_inst, samples_ref)
    diff_sample = ll_base_sample - ll_inst_sample

    mu_tilde = diff_sample.mean(dim=-1)
    sigma_tilde = diff_sample.std(dim=-1)
    normalized_diff = (diff - mu_tilde) / sigma_tilde

    # Flip sign so higher scores indicate AI text
    return -(normalized_diff.item())


def compute_inst_mul_base_inst_method(
    args,
    logits_base: torch.Tensor,
    logits_inst: torch.Tensor,
    labels_base: torch.Tensor,
    labels_inst: torch.Tensor
) -> float:
    """Method 4: inst * (base - inst) - multiplicative interaction"""
    ll_base = get_likelihood(logits_base, labels_base)
    ll_inst = get_likelihood(logits_inst, labels_inst)
    diff = ll_base - ll_inst
    # Compute raw multiplicative result
    mul_result = ll_inst * diff

    return mul_result.item()


def process_samples_with_method(
    method: str,
    args,
    data: Dict,
    base_cache: Dict,
    instruct_cache: Dict,
    sampling_cache: Dict,
    n_samples: int,
    device: str = "cuda"
) -> Tuple[List[float], List[float]]:
    """
    Process all samples using the specified comparison method.

    Returns:
        Tuple of (original_scores, sampled_scores)
    """
    original_scores = []
    sampled_scores = []

    for idx in range(n_samples):
        # Original text (human)
        logits_orig_base = base_cache["original"][idx]["logits"].to(device)
        logits_orig_inst = instruct_cache["original"][idx]["logits"].to(device)
        logits_orig_ref = sampling_cache["original"][idx]["logits"].to(device)
        labels_orig_base = base_cache["original"][idx]["labels"].to(device)
        labels_orig_inst = instruct_cache["original"][idx]["labels"].to(device)

        # Sampled text (AI)
        logits_samp_base = base_cache["sampled"][idx]["logits"].to(device)
        logits_samp_inst = instruct_cache["sampled"][idx]["logits"].to(device)
        logits_samp_ref = sampling_cache["sampled"][idx]["logits"].to(device)
        labels_samp_base = base_cache["sampled"][idx]["labels"].to(device)
        labels_samp_inst = instruct_cache["sampled"][idx]["labels"].to(device)

        with torch.no_grad():
            if method == 'base':
                original_score = compute_base_method(logits_orig_base, labels_orig_base)
                sampled_score = compute_base_method(logits_samp_base, labels_samp_base)
            elif method == 'base-inst':
                original_score = compute_base_inst_method(
                    logits_orig_base, logits_orig_inst, labels_orig_base, labels_orig_inst
                )
                sampled_score = compute_base_inst_method(
                    logits_samp_base, logits_samp_inst, labels_samp_base, labels_samp_inst
                )
            elif method == 'd_base-inst':
                original_score = compute_d_base_inst_method(
                    args, logits_orig_ref, logits_orig_base, logits_orig_inst,
                    labels_orig_base, labels_orig_inst
                )
                sampled_score = compute_d_base_inst_method(
                    args, logits_samp_ref, logits_samp_base, logits_samp_inst,
                    labels_samp_base, labels_samp_inst
                )
            elif method == 'inst_mul_base-inst':
                original_score = compute_inst_mul_base_inst_method(
                    args, logits_orig_base, logits_orig_inst,
                    labels_orig_base, labels_orig_inst
                )
                sampled_score = compute_inst_mul_base_inst_method(
                    args, logits_samp_base, logits_samp_inst,
                    labels_samp_base, labels_samp_inst
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        original_scores.append(original_score)
        sampled_scores.append(sampled_score)

    return original_scores, sampled_scores