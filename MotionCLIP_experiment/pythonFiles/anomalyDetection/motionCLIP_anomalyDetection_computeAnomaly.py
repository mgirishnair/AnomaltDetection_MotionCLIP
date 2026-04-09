#!/usr/bin/env python
# coding: utf-8
import sys
from datetime import datetime
import numpy as np
import argparse
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# --------------------------------------------------
# 1. Fit Gaussian on normal training embeddings
# --------------------------------------------------
def fit_full_gaussian(Z_train_normal, eps=1e-6):
    mu = np.mean(Z_train_normal, axis=0)
    cov = np.cov(Z_train_normal, rowvar=False)

    # numerical regularization
    cov = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)

    inv_cov = np.linalg.pinv(cov)
    return mu, cov, inv_cov


# --------------------------------------------------
# 2. Mahalanobis-style anomaly score with normalization
# --------------------------------------------------
def ood_score_paper(Z, mu, inv_cov):
    diff = Z - mu
    return np.sqrt(np.einsum("bi,ij,bj->b", diff, inv_cov, diff))
    #score = w1 * np.power(md, 1.0 / w2)
    #return np.minimum(1.0, score)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="motionclip_embeddings.npz")
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--split", type=str, required=True)
    args = parser.parse_args()

    log_file = f"results/anomaly_results_{args.split}.txt"
    sys.stdout = open(log_file, "w")
    print(f"Logging output to: {log_file}")

    data = np.load(args.embeddings_path, allow_pickle=True)

    Z_train_normal = data["Z_train_normal"]
    Z_test_normal = data["Z_test_normal"]
    Z_test_abnormal = data["Z_test_abnormal"]

    print("Loaded embeddings:")
    print("Z_train_normal:", Z_train_normal.shape)
    print("Z_test_normal:", Z_test_normal.shape)
    print("Z_test_abnormal:", Z_test_abnormal.shape)

    mu, cov, inv_cov = fit_full_gaussian(Z_train_normal)

    train_scores = ood_score_paper(Z_train_normal, mu, inv_cov)
    scores_normal = ood_score_paper(Z_test_normal, mu, inv_cov)
    scores_abnormal = ood_score_paper(Z_test_abnormal, mu, inv_cov)

    y_true = np.concatenate([
        np.zeros(len(scores_normal), dtype=int),
        np.ones(len(scores_abnormal), dtype=int),
    ])

    y_scores = np.concatenate([scores_normal, scores_abnormal])

    auroc = roc_auc_score(y_true, y_scores)
    prauc = average_precision_score(y_true, y_scores)

    print(f"\nAUROC : {auroc:.4f}")
    print(f"PR-AUC: {prauc:.4f}")

    threshold = np.percentile(train_scores, args.threshold_percentile)
    print(f"Threshold ({args.threshold_percentile}th percentile of train-normal scores): {threshold:.4f}")

    y_pred = (y_scores > threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    print("\nConfusion Matrix:")
    print(cm)

    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=["normal", "abnormal"]))

    print("Train normal scores: ", train_scores)
    print("Test normal scores:  ", scores_normal)
    print("Test abnormal scores:", scores_abnormal)


if __name__ == "__main__":
    main()
