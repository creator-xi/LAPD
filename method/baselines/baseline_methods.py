"""
Simple Baseline Methods: likelihood, rank, logrank, entropy

These methods use a single scoring model and compute simple statistical measures
of how well the model fits the text.
"""

import torch
import torch.nn.functional as F
import tqdm
import time
from typing import Dict, List, Tuple
from method.model import load_tokenizer, load_model
from method.metrics import get_roc_metrics, get_precision_recall_metrics

def get_likelihood(logits, labels):
    """Compute average log-likelihood of the sequence"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    logits = logits.view(-1, logits.shape[-1])
    labels = labels.view(-1)
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    log_likelihood = log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)
    return log_likelihood.mean().item()

def get_rank(logits, labels):
    """Compute average rank of the true tokens in the model's ordering"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # Get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # Make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), \
        "Expected one match per timestep"

    ranks = ranks.float() + 1  # Convert to 1-indexed rank
    return -ranks.mean().item()  # Negative because higher rank is better

def get_logrank(logits, labels):
    """Compute average log-rank of the true tokens"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    # Get rank of each label token in the model's likelihood ordering
    matches = (logits.argsort(-1, descending=True) == labels.unsqueeze(-1)).nonzero()
    assert matches.shape[1] == 3, f"Expected 3 dimensions in matches tensor, got {matches.shape}"

    ranks, timesteps = matches[:, -1], matches[:, -2]

    # Make sure we got exactly one match for each timestep in the sequence
    assert (timesteps == torch.arange(len(timesteps)).to(timesteps.device)).all(), \
        "Expected one match per timestep"

    ranks = ranks.float() + 1  # Convert to 1-indexed rank
    ranks = torch.log(ranks)
    return -ranks.mean().item()  # Negative because lower log-rank is better

def get_entropy(logits, labels):
    """Compute average entropy of the model's predictions"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1

    entropy = F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)
    entropy = -entropy.sum(-1)
    return entropy.mean().item()

def evaluate_method(args, criterion_fn, data, n_samples):
    """Evaluate a single criterion function"""
    # Load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()

    # Evaluate
    results = []
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {args.method} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        max_length = 1024

        # Original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt",
                                    padding=True, truncation=True, max_length=max_length+1,
                                    return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = scoring_model(**tokenized).logits[:, :-1]
            original_crit = criterion_fn(logits, labels)

        # Sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt",
                                    padding=True, truncation=True, max_length=max_length+1,
                                    return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        with torch.no_grad():
            logits = scoring_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits, labels)

        # Result
        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })
    total_time = time.time() - total_time_start
    time_per_sample = total_time / n_samples
    print(f"[INFO] total time: {total_time:.4f}")
    print(f"[INFO] time per sample of {args.method} method: {time_per_sample:.4f}")
    # Compute prediction scores
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    print(f"Criterion {args.method}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results
    results_dict = {
        'name': f'{args.method}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
        },
        'pr_metrics': {
            'pr_auc': pr_auc,
            'precision': p,
            'recall': r,
        },
        'loss': 1 - pr_auc,
        'time': {
            'total_time': total_time, 
            'n_samples': n_samples, 
            'time_per_sample': time_per_sample
        }
    }

    return results_dict

def handle_simple_baselines(args, data, n_samples):
    """Handle simple baseline methods (likelihood, rank, logrank, entropy)"""
    # Map method names to criterion functions
    criterion_fns = {
        'likelihood': get_likelihood,
        'rank': get_rank,
        'logrank': get_logrank,
        'entropy': get_entropy
    }

    # Get the appropriate criterion function
    criterion_fn = criterion_fns[args.method]

    # Evaluate
    return evaluate_method(args, criterion_fn, data, n_samples)
