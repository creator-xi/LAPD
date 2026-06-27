from typing import Union

import sys
import os
import tqdm
import time
import torch
import numpy as np
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
# from .utils import assert_tokenizer_consistency
from .metrics import perplexity, entropy
from method.baselines.fast_detect_gpt_double_gpu import get_sampling_discrepancy_analytic
from method.utils import setup_seed
from method.model import get_model_fullname, load_model, load_tokenizer
from method.metrics import get_roc_metrics, get_precision_recall_metrics

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
BINOCULARS_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
BINOCULARS_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class Binoculars(object):
    def __init__(self,
                 observer_name_or_path: str = "falcon-7b",
                 performer_name_or_path: str = "falcon-7b-instruct",
                 use_bfloat16: bool = True,
                 max_token_observed: int = 1024,
                 mode: str = "low-fpr",
                 cache_dir: str = "./cache",
                 device: str = "cuda"
                 ) -> None:
        self.change_mode(mode)
        self.cache_dir = cache_dir
        self.device = device

        # Load models using Fast-DetectGPT's loading utilities
        self.observer_model = load_model(observer_name_or_path, DEVICE_1, cache_dir)

        self.performer_model = load_model(performer_name_or_path, DEVICE_2, cache_dir)

        self.observer_model.eval()
        self.performer_model.eval()

        # Load tokenizer using Fast-DetectGPT's loading utilities
        self.tokenizer = load_tokenizer(observer_name_or_path, cache_dir)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = BINOCULARS_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = BINOCULARS_ACCURACY_THRESHOLD
        else:
            raise ValueError(f"Invalid mode: {mode}")

    def _tokenize(self, batch: list[str]) -> transformers.BatchEncoding:
        batch_size = len(batch)
        encodings = self.tokenizer(
            batch,
            return_tensors="pt",
            padding="longest" if batch_size > 1 else False,
            truncation=True,
            max_length=self.max_token_observed,
            return_token_type_ids=False).to(self.observer_model.device)
        return encodings

    @torch.inference_mode()
    def _get_logits(self, encodings: transformers.BatchEncoding) -> torch.Tensor:
        observer_logits = self.observer_model(**encodings.to(DEVICE_1)).logits
        performer_logits = self.performer_model(**encodings.to(DEVICE_2)).logits
        if DEVICE_1 != "cpu":
            torch.cuda.synchronize()
        return observer_logits, performer_logits

    def compute_score(self, input_text: Union[list[str], str], zscore: bool = False) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        if zscore:
            return self._compute_score_zscore(encodings, observer_logits, performer_logits)

        ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        binoculars_scores = ppl / x_ppl
        binoculars_scores = binoculars_scores.tolist()
        return binoculars_scores[0] if isinstance(input_text, str) else binoculars_scores

    def _compute_score_zscore(self, encodings, observer_logits, performer_logits):
        """Compute z-score standardized Binoculars score using analytic formula.

        Key insight: x_ppl (cross-model entropy) is constant under perturbation
        because it depends only on the two models' logits, not the actual tokens.
        So z(ppl/x_ppl) = z(ppl) / x_ppl, and z(ppl) = -get_sampling_discrepancy_analytic().
        """
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)
        discrepancy = get_sampling_discrepancy_analytic(
            observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
            encodings.input_ids[:, 1:].to(DEVICE_1))
        # discrepancy is z-score of log-likelihood (positive for AI text).
        # Negate to get z-score of perplexity (negative for AI text, matching Binoculars convention).
        # Then divide by x_ppl to preserve the Binoculars structure.
        scores = np.array([-discrepancy]) / x_ppl
        return scores.item()

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        binoculars_scores = np.array(self.compute_score(input_text))
        pred = np.where(binoculars_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred


def handle_binoculars(args, data, n_samples):
    """
    Handle Binoculars detection method for Fast-DetectGPT framework
    Args:
        args: Command line arguments
        data: Dictionary with 'original' and 'sampled' keys
        n_samples: Number of samples to process
    Returns:
        Dict with results matching fast_detect_gpt.py format
    """
    DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
    DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1
    max_length = 1024
    zscore = (args.method == 'binoculars_zscore')
    method_name = 'binoculars_zscore' if zscore else 'binoculars'

    print(f"Using devices: observer_model -> {DEVICE_1}, performer_model -> {DEVICE_2}")
    print(f"Z-score standardization: {'enabled' if zscore else 'disabled'}")
    setup_seed(args.seed)
    # Initialize Binoculars detector
    detector = Binoculars(
        observer_name_or_path=args.sampling_model_name,  # observer -> base
        performer_name_or_path=args.scoring_model_name,  # performer -> inst
        use_bfloat16=True,
        max_token_observed=max_length,
        mode="accuracy",
        cache_dir=args.cache_dir,
        device=args.device
    )

    results = []
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {method_name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # Original text - lower scores indicate AI-generated for Binoculars
        original_crit = detector.compute_score(original_text, zscore=zscore)

        # Sampled text
        sampled_crit = detector.compute_score(sampled_text, zscore=zscore)

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
    # Compute metrics - Binoculars uses lower scores for AI text
    # So we negate scores to match framework's convention (higher = more AI-like)
    predictions = {
        'real': [-x["original_crit"] for x in results],
        'samples': [-x["sampled_crit"] for x in results]
    }
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, "
          f"Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")

    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {method_name} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # Results matching fast_detect_gpt.py format
    results_dict = {
        'name': method_name,
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
        },
    }

    return results_dict
