import os
import numpy as np
import torch
import tqdm
import time

from method.model import check_tokenizer_compatibility
from method.metrics import get_roc_metrics, get_precision_recall_metrics, evaluate
from method.utils import exp_root, setup_seed, save_results, choose_agg_strategy
from method.core import get_sampling_discrepancy, stages_logits_cache
from method.dataloader import load_data
from method.args import create_mul_method_parser

def experiment(args):
    # check_tokenizer_compatibility(args.base_model, args.instruct_model, args.cache_dir)
    
    # load data
    data, n_samples = load_data(args)
    if n_samples == 0:
        print(f"[INFO] No data found for data_source={args.data_source}. skip")
        return

    setup_seed(args.seed)
    agg_strategy = choose_agg_strategy(args)
    
    instruct_cache, base_cache, sampling_cache, infer_time = stages_logits_cache(args, data)
    print("[Stage 4] Computing discrepancy scores...")
    results = []
    crit_compute_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing mul criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        
        logits_orig_inst = instruct_cache["original"][idx]["logits"].to(args.device)
        logits_orig_base = base_cache["original"][idx]["logits"].to(args.device)
        logits_orig_ref = sampling_cache["original"][idx]["logits"].to(args.device)

        labels_orig_inst = instruct_cache["original"][idx]["labels"].to(args.device)
        labels_orig_base = base_cache["original"][idx]["labels"].to(args.device)
        
        
        logits_samp_inst = instruct_cache["sampled"][idx]["logits"].to(args.device)
        logits_samp_base = base_cache["sampled"][idx]["logits"].to(args.device)
        logits_samp_ref = sampling_cache["sampled"][idx]["logits"].to(args.device)
        
        labels_samp_inst = instruct_cache["sampled"][idx]["labels"].to(args.device)
        labels_samp_base = base_cache["sampled"][idx]["labels"].to(args.device)
        

        with torch.no_grad():
            original_crit = get_sampling_discrepancy(
                args=args,
                logits_ref=logits_orig_ref,
                logits_score1=logits_orig_base,
                logits_score2=logits_orig_inst,
                labels1=labels_orig_base,
                labels2=labels_orig_inst,
                lambda_=args.lambda_,
                agg_strategy=agg_strategy
            )
            sampled_crit = get_sampling_discrepancy(
                args=args,
                logits_ref=logits_samp_ref,
                logits_score1=logits_samp_base,
                logits_score2=logits_samp_inst,
                labels1=labels_samp_base,
                labels2=labels_samp_inst,
                lambda_=args.lambda_,
                agg_strategy=agg_strategy
            )

        results.append({
            "original": original_text,
            "original_crit": original_crit,
            "sampled": sampled_text,
            "sampled_crit": sampled_crit
        })

    crit_compute_time = time.time() - crit_compute_time_start
    total_time = infer_time + crit_compute_time
    time_per_sample = total_time / n_samples
    print(f"[INFO] infer time: {infer_time:.4f}, crit time: {crit_compute_time:.4f}, total time: {total_time:.4f}")
    print(f"[INFO] time per sample of {args.method} method: {time_per_sample:.4f}")

    # compute prediction scores for real/sampled passages
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }
    
    human_mean = np.mean(predictions['real'])
    human_std = np.std(predictions['real'])
    ai_mean = np.mean(predictions['samples'])
    ai_std = np.std(predictions['samples'])
    print(f"Real mean/std: {human_mean:.2f}/{human_std:.2f}; Samples mean/std: {ai_mean:.2f}/{ai_std:.2f}")
    
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    _, best_f1, best_f1_threshold = evaluate(predictions['real'], predictions['samples'])
    print(f"Criterion mul threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, Best F1: {best_f1:.4f}")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {
        'name': f'mul_threshold',
        'info': {'n_samples': n_samples},
        'best_f1': {'best_f1': best_f1, 'best_f1_threshold': best_f1_threshold},
        'human mean/std': {'mean': human_mean, 'std': human_std},
        'ai mean/std': {'mean': ai_mean, 'std': ai_std},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'time': {
            'infer_time': infer_time,
            'crit_compute_time': crit_compute_time,
            'total_time': total_time, 
            'n_samples': n_samples, 
            'time_per_sample': time_per_sample
        },
    }
    save_results(args, results_dict, args.method)
    return results_dict

if __name__ == '__main__':
    parser = create_mul_method_parser()
    args = parser.parse_args()

    # Set method-specific parameters
    method = "lapd"
    args.method = 'lapd'
    args.output_dir = exp_root(args, method)

    experiment(args)