"""
Fast-DetectGPT Sampling Discrepancy Method

This method computes the sampling discrepancy between a scoring model and a sampling model.
It can use either Monte Carlo sampling or an analytic approximation.
"""
import torch
import tqdm
import time
import numpy as np
from method.utils import setup_seed
from method.model import load_tokenizer, load_model
from method.metrics import get_roc_metrics, get_precision_recall_metrics

def get_samples(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    nsamples = 10000
    lprobs = torch.log_softmax(logits, dim=-1)
    distrib = torch.distributions.categorical.Categorical(logits=lprobs)
    samples = distrib.sample([nsamples]).permute([1, 2, 0])
    return samples

def get_likelihood(logits, labels):
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels)
    return log_likelihood.mean(dim=1)

def get_sampling_discrepancy(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        print(f"[WARNING] vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    samples = get_samples(logits_ref, labels)
    log_likelihood_x = get_likelihood(logits_score, labels)
    log_likelihood_x_tilde = get_likelihood(logits_score, samples)
    miu_tilde = log_likelihood_x_tilde.mean(dim=-1)
    sigma_tilde = log_likelihood_x_tilde.std(dim=-1)
    discrepancy = (log_likelihood_x.squeeze(-1) - miu_tilde) / sigma_tilde
    return discrepancy.item()

def get_sampling_discrepancy_analytic(logits_ref, logits_score, labels):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
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
    return discrepancy.item()

def handle_fast_origin(args, data, n_samples):
    """
    Handle sampling discrepancy detection

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
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, DEVICE_2, args.cache_dir)
        sampling_model.eval()
    
    # criterion function
    if args.discrepancy_analytic:
        name = "sampling_discrepancy_analytic"
        criterion_fn = get_sampling_discrepancy_analytic
    else:
        name = "sampling_discrepancy"
        criterion_fn = get_sampling_discrepancy
    
    setup_seed(args.seed)
    results = []
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # Original text
        tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True,
                                    truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)
        labels = tokenized.input_ids[:, 1:].to(DEVICE_2)
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1].to(DEVICE_1)
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(original_text, return_tensors="pt", padding=True, 
                    truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_2)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1].to(DEVICE_1)
            labels = labels.to(DEVICE_1)
            original_crit = criterion_fn(logits_ref, logits_score, labels)

        # Sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                             truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)
        labels = tokenized.input_ids[:, 1:].to(DEVICE_2)
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1].to(DEVICE_1)
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                # Move to sampling model device for inference
                tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, 
                    truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_2)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1].to(DEVICE_1)
            labels = labels.to(DEVICE_1)
            sampled_crit = criterion_fn(logits_ref, logits_score, labels)

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
            'total_time': total_time, 
            'n_samples': n_samples, 
            'time_per_sample': time_per_sample
        }
    }

    return results_dict