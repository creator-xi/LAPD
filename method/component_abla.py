import os
import numpy as np

from method.model import check_tokenizer_compatibility
from method.metrics import get_roc_metrics, get_precision_recall_metrics, evaluate
from method.dataloader import load_data
from method.utils import setup_seed, save_results
from method.core import stages_logits_cache, process_samples_with_method
from method.args import create_contribution_analysis_parser

def experiment(args):
    # check_tokenizer_compatibility(args.base_model, args.instruct_model, args.cache_dir)
    
    # load data
    data, n_samples = load_data(args)
    if n_samples == 0:
        print(f"[INFO] No data found for data_source={args.data_source}. skip")
        return
        
    setup_seed(args.seed)
    
    instruct_cache, base_cache, sampling_cache, _ = stages_logits_cache(args, data)
    
    print(f"[Stage 4] Computing {args.comparison_method} criterion...")
    original_scores, sampled_scores = process_samples_with_method(
        args.comparison_method, args, data, base_cache, instruct_cache, sampling_cache, n_samples, args.device
    )

    # Create results with text data
    results = []
    for idx in range(n_samples):
        results.append({
            "original": data["original"][idx],
            "original_crit": original_scores[idx],
            "sampled": data["sampled"][idx],
            "sampled_crit": sampled_scores[idx]
        })

    predictions = {
        'real': original_scores,
        'samples': sampled_scores
    }
    
    human_mean = np.mean(predictions['real'])
    human_std = np.std(predictions['real'])
    ai_mean = np.mean(predictions['samples'])
    ai_std = np.std(predictions['samples'])
    print(f"Real mean/std: {human_mean:.2f}/{human_std:.2f}; Samples mean/std: {ai_mean:.2f}/{ai_std:.2f}")
    
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    _, best_f1, best_f1_threshold = evaluate(predictions['real'], predictions['samples'])
    print(f"Criterion {args.comparison_method} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}, Best F1: {best_f1:.4f}")
    
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    results_dict = {
        'name': f'{args.comparison_method}_threshold',
        'info': {'n_samples': n_samples},
        'best_f1': {'best_f1': best_f1, 'best_f1_threshold': best_f1_threshold},
        'human mean/std': {'mean': human_mean, 'std': human_std},
        'ai mean/std': {'mean': ai_mean, 'std': ai_std},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
    }
    save_results(args, results_dict, args.comparison_method)

if __name__ == '__main__':
    parser = create_contribution_analysis_parser()
    args = parser.parse_args()

    FORMAL_EXP_ROOT = os.environ.get("FORMAL_EXP_ROOT", "./formal_exp")

    # Set output directory based on parameters
    base_dir = os.path.join(FORMAL_EXP_ROOT, "compare", args.comparison_method, args.data_source)
    if args.data_source == 'main':
        output_dir = os.path.join(base_dir, f"{args.dataset}_{args.source_model}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'raid_attack':
        output_dir = os.path.join(base_dir, args.raid_attack_type, f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'cred':
        output_dir = os.path.join(base_dir, f"{args.cred_domain}_{args.cred_model}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    elif args.data_source == 'text_attack':
        output_dir = os.path.join(base_dir, f"{args.source_model}_{args.attack_type}", f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")
    else:
        output_dir = os.path.join(base_dir, f"{args.sampling_model}_{args.base_model}_{args.instruct_model}")

    args.output_dir = output_dir

    experiment(args)