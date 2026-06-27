from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import os
import gc
import time
import torch

def get_samples(logits, labels, nsamples=10000):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    lprobs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    """Compute mean log-likelihood of a sequence."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels  # [1, seq_len, 1]
    lprobs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    log_likelihood = lprobs.gather(dim=-1, index=labels)  # [1, seq_len]
    return log_likelihood.mean(dim=1)  # [1]

def get_likelihood_no_mean(logits, labels):
    """Compute per-token log-likelihood without averaging."""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels  # [1, seq_len, 1]s
    lprobs = torch.log_softmax(logits, dim=-1)  # [1, seq_len, vocab_size]
    log_likelihood = lprobs.gather(dim=-1, index=labels)  # [1, seq_len]
    return log_likelihood  # [1, seq_len]

def get_sampling_discrepancy(args, logits_ref, logits_score1, logits_score2, labels1, labels2, lambda_, agg_strategy):
    """
    1 -> base model
    2 -> instruct model
    calculate our "align imprint" metric
    """
    assert logits_ref.shape[0] == 1
    assert logits_score1.shape[0] == 1
    assert logits_score2.shape[0] == 1
    assert labels1.shape[0] == 1
    assert labels2.shape[0] == 1

    # Check vocab size compatibility with WARNING (not ERROR)
    if logits_ref.size(-1) != logits_score1.size(-1) or logits_ref.size(-1) != logits_score2.size(-1) or logits_score1.size(-1) != logits_score2.size(-1):
        vocab_sizes = {
            'sampling (ref)': logits_ref.size(-1),
            'base (score1)': logits_score1.size(-1),
            'instruct (score2)': logits_score2.size(-1)
        }
        print(f"[WARNING] Vocab size mismatch detected: {vocab_sizes}")
        print(f"[WARNING] Results may be unreliable if tokenizers differ significantly")

        # Truncate to minimum vocab size (conservative approach)
        # Useless in fact, because perturbation-based methods require the same tokenizer
        vocab_size = min(logits_ref.size(-1), logits_score1.size(-1), logits_score2.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score1 = logits_score1[:, :, :vocab_size]
        logits_score2 = logits_score2[:, :, :vocab_size]
        print(f"[WARNING] Truncated all logits to vocab_size={vocab_size}")

    samples1 = get_samples(logits_ref, labels1)  # get_samples doesn't use labels in fact
    samples2 = samples1

    get_ll_fn = get_likelihood_no_mean
    log_likelihood_x1 = get_ll_fn(logits_score1, labels1)
    log_likelihood_x2 = get_ll_fn(logits_score2, labels2)

    log_likelihood_x_tilde1 = get_ll_fn(logits_score1, samples1)
    log_likelihood_x_tilde2 = get_ll_fn(logits_score2, samples2)

    log_likelihood_x = agg_strategy(log_likelihood_x1, log_likelihood_x2, lambda_)
    log_likelihood_x_tilde = agg_strategy(log_likelihood_x_tilde1, log_likelihood_x_tilde2, lambda_)

    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)

    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde  # normalization

    return discrepancy.item()

def compute_logits_cache(args, data, model_name, model_type="scoring", shift=True):
    from method.model import load_tokenizer, load_model
    import tqdm

    print(f"Computing logits cache for {model_name} ({model_type})...")

    tokenizer = load_tokenizer(model_name, args.cache_dir)
    model = load_model(model_name, args.device, args.cache_dir)
    model.eval()

    n_samples = len(data["sampled"])
    cache = {"original": [], "sampled": []}
    max_length = 1024

    infer_time_start = time.time()
    with torch.no_grad():
        for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {model_type} logits"):
            # original text
            original_text = data["original"][idx]
            tokenized = tokenizer(original_text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length+1,  # +1 because we'll remove first token later
                                return_token_type_ids=False).to(args.device)
            if shift:
                labels = tokenized.input_ids[:, 1:].cpu()
                logits = model(**tokenized).logits[:, :-1].cpu()
            else:
                labels = tokenized.input_ids.cpu()
                logits = model(**tokenized).logits.cpu()

            cache["original"].append({"logits": logits, "labels": labels})

            # samplled text
            sampled_text = data["sampled"][idx]
            tokenized = tokenizer(sampled_text, return_tensors="pt", padding=True,
                                truncation=True, max_length=max_length+1,  # +1 because we'll remove first token later
                                return_token_type_ids=False).to(args.device)
            if shift:
                labels = tokenized.input_ids[:, 1:].cpu()
                logits = model(**tokenized).logits[:, :-1].cpu()
            else:
                labels = tokenized.input_ids.cpu()
                logits = model(**tokenized).logits.cpu()

            cache["sampled"].append({"logits": logits, "labels": labels})
    infer_time = time.time() - infer_time_start
    del model
    del tokenizer
    from method.utils import clear_gpu_memory
    clear_gpu_memory()
    return cache, infer_time

def stages_logits_cache(args, data):
    print("[Stage 1] Computing instruct model logits...")
    instruct_cache, inst_infer_time = compute_logits_cache(args, data, args.instruct_model, "instruct")
    
    print("[Stage 2] Computing base model logits...")
    base_cache, base_infer_time = compute_logits_cache(args, data, args.base_model, "base")
    
    sampling_infer_time = 0
    print("[Stage 3] Computing sampling model logits...")
    if args.sampling_model == args.base_model:
        sampling_cache = base_cache
        print("Sampling model == base model, reusing base model cache.")
    elif args.sampling_model == args.instruct_model:
        sampling_cache = instruct_cache
        print("Sampling model == instruct model, reusing instruct model cache.")
    else:
        sampling_cache, sampling_infer_time = compute_logits_cache(args, data, args.sampling_model, "sampling")
    infer_time = inst_infer_time + base_infer_time + sampling_infer_time
    return instruct_cache, base_cache, sampling_cache, infer_time