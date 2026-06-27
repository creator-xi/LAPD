"""
Double GPUs loading without compute-then-cache.
For time efficiency estimation only.
"""
import torch
import tqdm
import time
import numpy as np

from method.model import load_tokenizer, load_model, check_tokenizer_compatibility
from method.metrics import get_roc_metrics, get_precision_recall_metrics, evaluate
from method.utils import exp_root, setup_seed, save_results
from method.dataloader import load_data
from method.args import create_mul_method_parser
from method.core import token_level_sample_ll_mul, get_sampling_discrepancy, stages_logits_cache

def handle_ours_multi_gpu(args, data, n_samples):
    """
    Handle multi-model sampling discrepancy detection with 2GPU loading

    Args:
        args: Command-line arguments
        data: Dict with 'original' and 'sampled' keys
        n_samples: Number of samples

    Returns:
        Dict with results
    """
    if n_samples == 0:
        print(f"[INFO] No data found for data_source={args.data_source}. skip")
        return
    DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
    max_length = 1024

    print(f"Using devices: base_model -> {DEVICE_1}, instruct_model -> {DEVICE_2}, sampling_model -> {DEVICE_1}")

    # Base model (for scoring)
    base_tokenizer = load_tokenizer(args.base_model, args.cache_dir)
    base_model = load_model(args.base_model, DEVICE_1, args.cache_dir)
    base_model.eval()

    # Instruct model (for scoring)
    instruct_tokenizer = load_tokenizer(args.instruct_model, args.cache_dir)
    instruct_model = load_model(args.instruct_model, DEVICE_2, args.cache_dir)
    instruct_model.eval()

    # Sampling model
    if args.sampling_model == args.base_model:
        sampling_tokenizer = base_tokenizer
        sampling_model = base_model
    elif args.sampling_model == args.instruct_model:
        sampling_tokenizer = instruct_tokenizer
        sampling_model = instruct_model
    else:
        sampling_tokenizer = load_tokenizer(args.sampling_model, args.cache_dir)
        sampling_model = load_model(args.sampling_model, DEVICE_1, args.cache_dir)
        sampling_model.eval()

    # Aggregation strategy
    agg_strategy = token_level_sample_ll_mul
    setup_seed(args.seed)
    results = []
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing mul criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        tokenized_base = base_tokenizer(original_text, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)
        tokenized_instruct = instruct_tokenizer(original_text, return_tensors="pt", padding=True,
                                             truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_2)
        tokenized_sampling = sampling_tokenizer(original_text, return_tensors="pt", padding=True,
                                              truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)

        # Move labels between devices as needed
        labels_base = tokenized_base.input_ids[:, 1:].to(DEVICE_1)
        labels_instruct = tokenized_instruct.input_ids[:, 1:].to(DEVICE_2)

        with torch.no_grad():
            # Get logits from all models
            logits_base = base_model(**tokenized_base).logits[:, :-1].to(DEVICE_1)
            logits_instruct = instruct_model(**tokenized_instruct).logits[:, :-1].to(DEVICE_2)
            logits_ref = sampling_model(**tokenized_sampling).logits[:, :-1].to(DEVICE_1)

            # Move everything to DEVICE_1 for computation
            logits_instruct = logits_instruct.to(DEVICE_1)
            labels_instruct = labels_instruct.to(DEVICE_1)

            # Compute multi-model sampling discrepancy
            original_crit = get_sampling_discrepancy(
                args=args,
                logits_ref=logits_ref,
                logits_score1=logits_base,
                logits_score2=logits_instruct,
                labels1=labels_base,
                labels2=labels_instruct,
                lambda_=args.lambda_,
                agg_strategy=agg_strategy
            )

        tokenized_base = base_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                     truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)
        tokenized_instruct = instruct_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                             truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_2)
        tokenized_sampling = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True,
                                              truncation=True, max_length=max_length+1, return_token_type_ids=False).to(DEVICE_1)

        labels_base = tokenized_base.input_ids[:, 1:].to(DEVICE_1)
        labels_instruct = tokenized_instruct.input_ids[:, 1:].to(DEVICE_2)

        with torch.no_grad():
            logits_base = base_model(**tokenized_base).logits[:, :-1].to(DEVICE_1)
            logits_instruct = instruct_model(**tokenized_instruct).logits[:, :-1].to(DEVICE_2)
            logits_ref = sampling_model(**tokenized_sampling).logits[:, :-1].to(DEVICE_1)

            # Move everything to DEVICE_1 for computation
            logits_instruct = logits_instruct.to(DEVICE_1)
            labels_instruct = labels_instruct.to(DEVICE_1)

            sampled_crit = get_sampling_discrepancy(
                args=args,
                logits_ref=logits_ref,
                logits_score1=logits_base,
                logits_score2=logits_instruct,
                labels1=labels_base,
                labels2=labels_instruct,
                lambda_=args.lambda_,
                agg_strategy=agg_strategy
            )

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
    _, best_f1, best_f1_threshold = evaluate(predictions['real'], predictions['samples'])
    print(f"Criterion mul_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, Best F1: {best_f1:.4f}")

    # Results
    results_dict = {
        'name': 'mul_threshold',
        'info': {'n_samples': n_samples},
        'best_f1': {'best_f1': best_f1, 'best_f1_threshold': best_f1_threshold},
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
    save_results(args, results_dict, args.method)
    return results_dict

if __name__ == '__main__':
    parser = create_mul_method_parser()
    args = parser.parse_args()
    method = 'lapd_multi_gpu'
    args.method = method
    args.output_dir = exp_root(args, method)
    data, n_samples = load_data(args)
    handle_ours_multi_gpu(args, data, n_samples)