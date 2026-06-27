import os
import json
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve
from .detector import DetectLLM
from method.utils import exp_root, setup_seed, save_results
from method.dataloader import load_data

def compute_scores(bino, texts, label):
    scores = []
    for text in tqdm(texts, desc=f"Scoring {label} texts"):
        try:
            score = bino.compute_score(text)
            if isinstance(score, (list, tuple, np.ndarray)):
                if len(score) > 0:
                    score = float(score[0])
                else:
                    score = 0.0
            else:
                score = float(score)
        except Exception as e:
            print(f"Error scoring text: {e}")
            score = 0.0
        scores.append(score)
    return scores

def evaluate(human_scores, ai_scores):
    scores = human_scores + ai_scores
    labels = [1] * len(human_scores) + [0] * len(ai_scores)

    auroc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
    best_f1 = np.max(f1_scores)

    return auroc, best_f1

def main():
    # Updated to use the existing data loading framework
    parser = argparse.ArgumentParser(description="Evaluate DNA-DetectLLM AUROC and F1.")

    # Data source arguments (following lapd.py)
    parser.add_argument('--data_source', type=str, choices=['main', 'm4', 'detectrl_multidomain', 'detectrl_multillm', 'realdet', 'text_attack'],
                       default='main', help='which data source/benchmark to use')
    parser.add_argument('--dataset', type=str, default="xsum", choices=["xsum", "wp", "arxiv"])
    parser.add_argument('--source_model', type=str, default="gpt4o", choices=["claude3.7", "gemini2.0", "gpt4o"],
                       help="model to generate machine text from original human text")
    parser.add_argument('--attack_type', type=str, default="delete", choices=["delete", "dipper", "insert", "replace"],
                       help="type of text attack for text_attack data source")

    # DNA-DetectLLM specific arguments
    parser.add_argument('--observer_model', type=str, default="falcon-7b", help="Observer model name")
    parser.add_argument('--performer_model', type=str, default="falcon-7b-instruct", help="Performer model name")
    parser.add_argument('--mode', type=str, default="low-fpr", choices=["low-fpr", "accuracy"], help="Detection mode")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cache_dir', type=str, default="./cache")
    parser.add_argument('--max_samples', type=int, default=150, help='Maximum number of samples to process')
    parser.add_argument('--device', type=str, default="cuda")

    args = parser.parse_args()
    method = 'dna_detectllm'
    args.method = method
    args.base_model = args.observer_model
    args.instruct_model = args.performer_model
    args.output_file = exp_root(args, method)

    # Set seed for reproducibility
    setup_seed(args.seed)

    # Load data using the existing framework
    data, n_samples = load_data(args)
    human_texts = data["original"]
    ai_texts = data["sampled"]

    bino = DetectLLM(
        observer_model_name=args.observer_model,
        performer_model_name=args.performer_model,
        cache_dir=args.cache_dir,
        mode=args.mode,
    )
    human_scores = compute_scores(bino, human_texts, "human")
    ai_scores = compute_scores(bino, ai_texts, "AI")

    results_dict = bino.evaluate_and_save_results(
        human_scores=human_scores,
        ai_scores=ai_scores,
        human_texts=human_texts,
        ai_texts=ai_texts,
        observer_model_name=args.observer_model,
        performer_model_name=args.performer_model,
        mode="low-fpr",
        save_path="./results/dna_detectllm_results.json"
    )
    auroc, best_f1 = evaluate(human_scores, ai_scores)

    print(f"AUROC: {auroc:.4f}")
    print(f"Best F1: {best_f1:.4f}")

    save_results(args, results_dict, method)

if __name__ == '__main__':
    main()
