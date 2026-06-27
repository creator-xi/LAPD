import torch
import tqdm
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from method.utils import setup_seed
from method.metrics import get_roc_metrics, get_precision_recall_metrics
from method.model import from_pretrained

def handle_roberta(args, data, n_samples):
    if args.method == 'roberta_base':
        detector = from_pretrained(AutoModelForSequenceClassification, 'roberta-base-openai-detector', {}, args.cache_dir).to(args.device)
        tokenizer = from_pretrained(AutoTokenizer, 'roberta-base-openai-detector', {}, args.cache_dir)
    elif args.method == 'roberta_large':
        detector = from_pretrained(AutoModelForSequenceClassification, 'roberta-large-openai-detector', {}, args.cache_dir).to(args.device)
        tokenizer = from_pretrained(AutoTokenizer, 'roberta-large-openai-detector', {}, args.cache_dir)
    else:
        raise ValueError(f"Invalid method: {args.method} not in 'roberta_base' or 'roberta_large'")

    detector.eval()
    setup_seed(args.seed)
    results = []
    for i in tqdm.tqdm(range(n_samples), desc=f"Computing {args.method} criterion"):
        original_text = data['original'][i]
        sampled_text = data['sampled'][i]
        # original text
        tokenized = tokenizer(original_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
        with torch.no_grad():
            original_crit = detector(**tokenized).logits.softmax(-1)[0, 0].item()
        # sampled text
        tokenized = tokenizer(sampled_text, padding=True, truncation=True, max_length=512, return_tensors="pt").to(args.device)
        with torch.no_grad():
            sampled_crit = detector(**tokenized).logits.softmax(-1)[0, 0].item()
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
        
    # Compute metrics
    predictions = {
        'real': [x["original_crit"] for x in results],
        'samples': [x["sampled_crit"] for x in results]
    }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {args.method}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results
    results_dict = {
        'name': f'{args.method}_threshold',
        'info': {'n_samples': n_samples},
        'predictions': predictions,
        'raw_results': results,
        'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
        'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
        'loss': 1 - pr_auc
    } 
    return results_dict