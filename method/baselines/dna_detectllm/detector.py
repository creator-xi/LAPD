import os
import sys
import json
import tqdm
import time
import torch
import numpy as np
from typing import Union
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc
from .metrics import perplexity, entropy, min_perplexity, entropy_pro, calculate_window_ppl, calculate_window_entropy, joint_score_ratio, w_entropy, auc_perplexity, relative_entropy
from .metrics import sum_perplexity
from method.model import load_model, load_tokenizer    
from method.utils import setup_seed

torch.set_grad_enabled(False)

huggingface_config = {
    # Only required for private models from Huggingface (e.g. LLaMA models)
    "TOKEN": os.environ.get("HF_TOKEN", None)
}

# selected using Falcon-7B and Falcon-7B-Instruct at bfloat16
detectllm_ACCURACY_THRESHOLD = 0.9015310749276843  # optimized for f1-score
detectllm_FPR_THRESHOLD = 0.8536432310785527  # optimized for low-fpr [chosen at 0.01%]

DEVICE_1 = "cuda:0" if torch.cuda.is_available() else "cpu"
DEVICE_2 = "cuda:1" if torch.cuda.device_count() > 1 else DEVICE_1

class DetectLLM(object):
    def __init__(self,
                 observer_model_name: str = "falcon-7b",
                 performer_model_name: str = "falcon-7b-instruct",
                 use_bfloat16: bool = False,
                 max_token_observed: int = 1024,
                 mode: str = "low-fpr",
                 cache_dir: str = "./cache",
                ) -> None:
        self.mode = mode
        self.change_mode(mode)
        self.observer_model_name = observer_model_name
        self.performer_model_name = performer_model_name
        self.cache_dir = cache_dir

        # Load models using the centralized loading function
        self.observer_model = load_model(observer_model_name, DEVICE_1, cache_dir)
        self.performer_model = load_model(performer_model_name, DEVICE_2, cache_dir)
        self.observer_model.eval()
        self.performer_model.eval()

        # Load tokenizer using the centralized loading function
        self.tokenizer = load_tokenizer(observer_model_name, cache_dir)
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.max_token_observed = max_token_observed

    def change_mode(self, mode: str) -> None:
        if mode == "low-fpr":
            self.threshold = detectllm_FPR_THRESHOLD
        elif mode == "accuracy":
            self.threshold = detectllm_ACCURACY_THRESHOLD
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

    def cleanup(self):
        if self.observer_model is not None:
            del self.observer_model
            del self.performer_model
            self.observer_model = None
            self.performer_model = None
        torch.cuda.empty_cache()

    def compute_score(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl = sum_perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        detectllm_scores = ppl / (2*x_ppl)
        detectllm_scores = detectllm_scores.tolist()

        if isinstance(input_text, str):
            return detectllm_scores[0] if len(detectllm_scores) > 0 else 0.0
        else:
            return detectllm_scores

    def compute_anomaly_level(self, input_text: Union[list[str], str]) -> Union[float, list[float]]:
        batch = [input_text] if isinstance(input_text, str) else input_text
        encodings = self._tokenize(batch)
        observer_logits, performer_logits = self._get_logits(encodings)

        ppl_auc = auc_perplexity(encodings, performer_logits, repair_order="h2l")

        # ppl = perplexity(encodings, performer_logits)
        x_ppl = entropy(observer_logits.to(DEVICE_1), performer_logits.to(DEVICE_1),
                        encodings.to(DEVICE_1), self.tokenizer.pad_token_id)

        detectllm_scores = ppl_auc / x_ppl
        detectllm_scores = detectllm_scores.tolist()

        print("scores:", ppl_auc, x_ppl, detectllm_scores)

        if isinstance(input_text, str):
            return detectllm_scores[0] if len(detectllm_scores) > 0 else 0.0
        else:
            return detectllm_scores

    def predict(self, input_text: Union[list[str], str]) -> Union[list[str], str]:
        detectllm_scores = self.compute_score(input_text)
        pred = np.where(detectllm_scores < self.threshold,
                        "Most likely AI-generated",
                        "Most likely human-generated"
                        ).tolist()
        return pred

    def evaluate_and_save_results(self, human_scores: list[float], ai_scores: list[float],
                                 human_texts: list[str] = None, ai_texts: list[str] = None,
                                 observer_model_name: str = None, performer_model_name: str = None,
                                 mode: str = None, save_path: str = None) -> dict:
        
        labels = np.array([1] * len(human_scores) + [0] * len(ai_scores))
        scores = np.array(human_scores + ai_scores)

        n_samples = len(human_scores)

        roc_auc = roc_auc_score(labels, scores)
        fpr, tpr, _ = roc_curve(labels, scores)

        precision, recall, _ = precision_recall_curve(labels, scores)
        pr_auc = auc(recall, precision)

        human_mean = np.mean(human_scores)
        human_std = np.std(human_scores)
        ai_mean = np.mean(ai_scores)
        ai_std = np.std(ai_scores)

        predictions = {
            'real': human_scores,
            'samples': ai_scores
        }

        results = []
        min_length = min(len(human_scores), len(ai_scores))
        for i in range(min_length):
            results.append({
                "original": human_texts[i],
                "original_crit": human_scores[i],
                "sampled": ai_texts[i],
                "sampled_crit": ai_scores[i]
            })

        results_dict = {
            'name': 'dna_detectllm',
            'info': {
                'n_samples': n_samples,
                'observer_model': observer_model_name or self.observer_model_name,
                'performer_model': performer_model_name or self.performer_model_name,
                'mode': mode or self.mode,
            },
            'human mean/std': {'mean': float(human_mean), 'std': float(human_std)},
            'ai mean/std': {'mean': float(ai_mean), 'std': float(ai_std)},
            'predictions': predictions,
            'raw_results': results,
            'metrics': {
                'roc_auc': float(roc_auc),
                'fpr': fpr.tolist(),
                'tpr': tpr.tolist()
            },
            'pr_metrics': {
                'pr_auc': float(pr_auc),
                'precision': precision.tolist(),
                'recall': recall.tolist()
            },
            'loss': 1 - float(pr_auc),
            'threshold': float(self.threshold)
        }
        return results_dict


def handle_dna_detectllm(args, data, n_samples):
    """
    Handle DNA-DetectLLM detection method integration with unified baseline framework

    Args:
        args: Command line arguments
        data: Dictionary containing 'original' and 'sampled' texts
        n_samples: Number of samples to process

    Returns:
        Dictionary containing detection results in standard format
    """
    # Add parent directory to path to import Fast-DetectGPT utilities
    sys.path.append(os.path.join(os.path.dirname(__file__), '../../../'))
    setup_seed(args.seed)

    # Initialize detector
    detector = DetectLLM(
        observer_model_name=args.sampling_model_name,
        performer_model_name=args.scoring_model_name,
        use_bfloat16=True,
        max_token_observed=1024,
        mode="accuracy",  # Use accuracy mode for framework consistency
        cache_dir=args.cache_dir
    )

    # Compute scores for all samples
    human_scores = []
    ai_scores = []
    human_texts = []
    ai_texts = []
    total_time_start = time.time()
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing dna-detectllm criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]

        # Compute DNA-DetectLLM scores
        original_score = detector.compute_score(original_text)
        sampled_score = detector.compute_score(sampled_text)

        # For DNA-DetectLLM, lower scores indicate AI-generated text (similar to Binoculars)
        # So we negate the scores to match framework convention where higher = more AI-like
        human_scores.append(original_score)
        ai_scores.append(sampled_score)
        human_texts.append(original_text)
        ai_texts.append(sampled_text)
    total_time = time.time() - total_time_start
    time_per_sample = total_time / n_samples
    print(f"[INFO] total time: {total_time}")
    print(f"[INFO] time per sample: {time_per_sample}")
    # Use the existing evaluate_and_save_results function for consistent formatting
    results = detector.evaluate_and_save_results(
        human_scores=human_scores,
        ai_scores=ai_scores,
        human_texts=human_texts,
        ai_texts=ai_texts,
        observer_model_name=args.sampling_model_name,
        performer_model_name=args.scoring_model_name,
        mode="accuracy",
        save_path=None  # Don't save, just return
    )
    results['time'] = {
        'total_time': total_time, 
        'n_samples': n_samples, 
        'time_per_sample': time_per_sample
    }

    # Cleanup models to free memory
    detector.cleanup()

    return results
