"""
RAI (Implicit Reward Model) - Double GPU Version

This version loads instruct and base models on separate GPUs for faster parallel inference.
Field mapping for RAI:
    - scoring_model -> instruct_model (phi) -> GPU 1
    - sampling_model -> base_model (theta) -> GPU 2

RAI Score = ll_base - ll_instruct
"""

import torch
import tqdm
import time
import numpy as np
from method.utils import setup_seed
from method.model import load_tokenizer, load_model
from method.metrics import get_roc_metrics, get_precision_recall_metrics


def get_likelihood(logits, labels):
    """Compute average log-likelihood of the sequence"""
    assert logits.shape[0] == 1
    assert labels.shape[0] == 1
    labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    log_likelihood = lprobs.gather(dim=-1, index=labels).squeeze(-1)
    return log_likelihood.mean().item()


def handle_rai_double_gpu(args, data, n_samples):
    """
    Handle RAI detection with double GPU setup

    Device mapping:
        - Instruct model (scoring_model) -> GPU 1
        - Base model (sampling_model) -> GPU 2

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

    print("=" * 50)
    print("RAI Detection Method (Double GPU Version)")
    print("=" * 50)
    print(f"Device mapping:")
    print(f"  Instruct Model (phi) -> {DEVICE_1}")
    print(f"  Base Model (theta) -> {DEVICE_2}")
    print(f"Field mapping: scoring_model->instruct, sampling_model->base")
    print("=" * 50)

    # Load instruct model on GPU 1
    print(f"\nLoading instruct model: {args.scoring_model_name} on {DEVICE_1}...")
    instruct_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    instruct_model = load_model(args.scoring_model_name, DEVICE_1, args.cache_dir)
    instruct_model.eval()

    # Load base model on GPU 2
    print(f"Loading base model: {args.sampling_model_name} on {DEVICE_2}...")
    if args.sampling_model_name != args.scoring_model_name:
        base_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        base_model = load_model(args.sampling_model_name, DEVICE_2, args.cache_dir)
        base_model.eval()
    else:
        print("Base model = instruct model, reusing tokenizer and model")
        base_tokenizer = instruct_tokenizer
        base_model = instruct_model

    # Compute RAI scores
    print("\n[Stage 3] Computing RAI scores...")
    print("RAI formula: score = ll_base - ll_instruct")

    setup_seed(args.seed)
    results = []
    total_time_start = time.time()

    for idx in tqdm.tqdm(range(n_samples), desc="Computing RAI scores"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # ===== Original text (human) =====
        # Tokenize for instruct model
        tokenized_inst = instruct_tokenizer(
            original_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length + 1,
            return_token_type_ids=False
        ).to(DEVICE_1)
        labels_inst = tokenized_inst.input_ids[:, 1:]

        # Tokenize for base model
        if args.sampling_model_name == args.scoring_model_name:
            tokenized_base = tokenized_inst
            labels_base = labels_inst
        else:
            tokenized_base = base_tokenizer(
                original_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length + 1,
                return_token_type_ids=False
            ).to(DEVICE_2)
            labels_base = tokenized_base.input_ids[:, 1:]
            # Verify tokenizers match
            assert torch.all(labels_base.cpu() == labels_inst.cpu()), "Tokenizer mismatch!"

        # Compute logits and likelihoods
        with torch.no_grad():
            # Instruct model on GPU 1
            logits_inst = instruct_model(**tokenized_inst).logits[:, :-1]

            # Base model on GPU 2
            if args.sampling_model_name == args.scoring_model_name:
                logits_base = logits_inst
            else:
                logits_base = base_model(**tokenized_base).logits[:, :-1].to(DEVICE_1)

            # Compute RAI score: ll_base - ll_instruct
            ll_human_inst = get_likelihood(logits_inst, labels_inst)
            ll_human_base = get_likelihood(logits_base, labels_base.to(DEVICE_1))
            original_crit = ll_human_base - ll_human_inst

        # ===== Sampled text (AI) =====
        # Tokenize for instruct model
        tokenized_inst = instruct_tokenizer(
            sampled_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length + 1,
            return_token_type_ids=False
        ).to(DEVICE_1)
        labels_inst = tokenized_inst.input_ids[:, 1:]

        # Tokenize for base model
        if args.sampling_model_name == args.scoring_model_name:
            tokenized_base = tokenized_inst
            labels_base = labels_inst
        else:
            tokenized_base = base_tokenizer(
                sampled_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_length + 1,
                return_token_type_ids=False
            ).to(DEVICE_2)
            labels_base = tokenized_base.input_ids[:, 1:]
            # Verify tokenizers match
            assert torch.all(labels_base.cpu() == labels_inst.cpu()), "Tokenizer mismatch!"

        # Compute logits and likelihoods
        with torch.no_grad():
            # Instruct model on GPU 1
            logits_inst = instruct_model(**tokenized_inst).logits[:, :-1]

            # Base model on GPU 2
            if args.sampling_model_name == args.scoring_model_name:
                logits_base = logits_inst
            else:
                logits_base = base_model(**tokenized_base).logits[:, :-1].to(DEVICE_1)

            # Compute RAI score: ll_base - ll_instruct
            ll_ai_inst = get_likelihood(logits_inst, labels_inst)
            ll_ai_base = get_likelihood(logits_base, labels_base.to(DEVICE_1))
            sampled_crit = ll_ai_base - ll_ai_inst

        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })

    total_time = time.time() - total_time_start
    time_per_sample = total_time / n_samples

    print(f"\n[INFO] total time: {total_time:.4f}")
    print(f"[INFO] time per sample: {time_per_sample:.4f}")

    # Compute metrics
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }

    print(f"\nReal mean/std: {np.mean(predictions['real']):.4f}/{np.std(predictions['real']):.4f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.4f}/{np.std(predictions['samples']):.4f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])

    print(f"\nRAI ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Return results in standard format
    results_dict = {
        'name': 'rai_double_gpu',
        'info': {
            'n_samples': n_samples,
            'instruct_model': args.scoring_model_name,
            'base_model': args.sampling_model_name,
            'device_1': DEVICE_1,
            'device_2': DEVICE_2
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
        }
    }

    return results_dict
