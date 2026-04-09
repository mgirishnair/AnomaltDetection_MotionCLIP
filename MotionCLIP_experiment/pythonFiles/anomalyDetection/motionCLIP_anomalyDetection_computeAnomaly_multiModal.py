#!/usr/bin/env python
# coding: utf-8
import sys
import os
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

# --------------------------------------------------
# Cosine distance
# --------------------------------------------------
def ood_score_cosine(Z, mu, eps=1e-8):
    Z_norm = np.linalg.norm(Z, axis=1, keepdims=True)
    mu_norm = np.linalg.norm(mu)
    Z_unit = Z / np.clip(Z_norm, eps, None)
    mu_unit = mu / max(mu_norm, eps)
    cosine_sim = Z_unit @ mu_unit
    return 1.0 - cosine_sim


# --------------------------------------------------
# 3. Fit multimodal gaussian to the data with shared covariance
# --------------------------------------------------
def fit_classwise_means_shared_cov(Z_train_normal, y_train_normal, eps=1e-6):
    shared_cov = np.cov(Z_train_normal, rowvar=False)
    shared_cov = shared_cov + eps * np.eye(shared_cov.shape[0], dtype=shared_cov.dtype)
    shared_inv_cov = np.linalg.pinv(shared_cov)
    models = {}
    for cls in np.unique(y_train_normal):
        Z_cls = Z_train_normal[y_train_normal == cls]
        mu = np.mean(Z_cls, axis=0)
        models[int(cls)] = {
            "mu": mu,
            "inv_cov": shared_inv_cov,
            "n": len(Z_cls),
        }
    return models

# -------------------------------------------------
# 3.5 Fit multimodal gaussian to the data with per class covariance
# -------------------------------------------------
def fit_classwise_gaussians(Z_train_normal, y_train_normal, eps=1e-6):
    models = {}
    for cls in np.unique(y_train_normal):
        Z_cls = Z_train_normal[y_train_normal == cls]
        mu = np.mean(Z_cls, axis=0)
        cov = np.cov(Z_cls, rowvar=False)
        cov = cov + eps * np.eye(cov.shape[0], dtype=cov.dtype)
        inv_cov = np.linalg.pinv(cov)
        models[int(cls)] = {
            "mu": mu,
            "cov": cov,
            "inv_cov": inv_cov,
            "n": len(Z_cls),
        }
    return models

# ---------------------------------------------------
# Classwise Gaussian without covariance, when using with cosine distance
# ---------------------------------------------------
def fit_classwise_means(Z_train_normal, y_train_normal):
    models = {}
    for cls in np.unique(y_train_normal):
        Z_cls = Z_train_normal[y_train_normal == cls]
        mu = np.mean(Z_cls, axis=0)
        models[int(cls)] = {
            "mu": mu,
            "n": len(Z_cls),
        }
    return models

# -------------------------------------------------
# 4. Multimodal scoring function
# -------------------------------------------------
def multimodal_ood_score(Z, models):
    all_scores = []
    for cls, model in models.items():
        scores_cls = ood_score_paper(Z, model["mu"], model["inv_cov"])
#        scores_cls = ood_score_cosine(Z, model["mu"])
        all_scores.append(scores_cls)
    all_scores = np.stack(all_scores, axis=1)   # [N, num_classes]
    min_scores = np.min(all_scores, axis=1)
    return min_scores, all_scores

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings_path", type=str, default="motionclip_embeddings.npz")
    parser.add_argument("--threshold_percentile", type=float, default=95.0)
    parser.add_argument("--split", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()

#    os.makedirs("results/multimodal/classwiseCosine", exist_ok=True)
#    log_file = f"results/multimodal/classwiseCosine/anomaly_results_multimodalClasswiseCosine_{args.split}.txt"
#    sys.stdout = open(log_file, "w")
#    print(f"Logging output to: {log_file}")

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    sys.stdout = open(args.output_path, "w")
    print(f"Logging output to: {args.output_path}") 

    data = np.load(args.embeddings_path, allow_pickle=True)

    Z_train_normal = data["Z_train_normal"]
    Z_test_normal = data["Z_test_normal"]
    Z_test_abnormal = data["Z_test_abnormal"]
    y_train_normal = data["y_train_normal"]

    print("Loaded embeddings:")
    print("Z_train_normal:", Z_train_normal.shape)
    print("Z_test_normal:", Z_test_normal.shape)
    print("Z_test_abnormal:", Z_test_abnormal.shape)

    models = fit_classwise_gaussians(Z_train_normal, y_train_normal)   #change to fit_classwise_gaussians to have per class covariance instead of shared

    print("Normal classes in training:", np.unique(y_train_normal))

    for cls, model in models.items():
       print(f"class {cls}: n={model['n']}")

    train_scores,_ = multimodal_ood_score(Z_train_normal,models)
    scores_normal,_ = multimodal_ood_score(Z_test_normal, models)
    scores_abnormal,_ = multimodal_ood_score(Z_test_abnormal, models)

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
