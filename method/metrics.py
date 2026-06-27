from sklearn.metrics import roc_curve, precision_recall_curve, auc, roc_auc_score
import numpy as np

COLORS = ["#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442",
            "#56B4E9", "#E69F00", "#000000", "#0072B2", "#009E73",
            "#D55E00", "#CC79A7", "#F0E442", "#56B4E9", "#E69F00"]

def evaluate(human_scores, ai_scores):
    scores = human_scores + ai_scores
    # Use consistent labeling: human=0, ai=1 to match get_roc_metrics and get_precision_recall_metrics
    labels = [0] * len(human_scores) + [1] * len(ai_scores)

    auroc = roc_auc_score(labels, scores)
    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
    best_f1_index = np.argmax(f1_scores)
    best_f1 = f1_scores[best_f1_index]
    best_f1_threshold = thresholds[best_f1_index]
    return auroc, best_f1, best_f1_threshold

def get_roc_metrics(real_preds, sample_preds):
    # y_true: 0 for real (human) text, 1 for sampled (AI) text
    # y_score: prediction scores for each sample
    print('real_preds:', len(real_preds), 'sample_preds:', len(sample_preds))
    print('y_true num of 1:', np.sum([0]*len(real_preds) + [1]*len(sample_preds)))
    y_true = [0] * len(real_preds) + [1] * len(sample_preds)
    y_score = real_preds + sample_preds
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)

def get_precision_recall_metrics(real_preds, sample_preds):
    # y_true: 0 for real (human) text, 1 for sampled (AI) text
    # y_score: prediction scores for each sample
    y_true = [0] * len(real_preds) + [1] * len(sample_preds)
    y_score = real_preds + sample_preds
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    pr_auc = auc(recall, precision)
    return precision.tolist(), recall.tolist(), float(pr_auc)
