#!/usr/bin/env python
# coding: utf-8

# ## Anomaly Detection using MotionCLIP - Probability based

# ### Input Data:
# Would be NTU-RGB+D dataset converted into clips of [60, 25, 6] shape, which are 60 frames, 25 SMPL points (24 + translation) and rot6d representation. 
# 
# Would be of the format: 
# ```
# sample = {
#     "pose":  np.ndarray of shape [60, 25, 6],
#     "label": int,   # NTU class label 1..60 depending on your indexing
#     "id":    str,   # e.g. S001C001P001R001A001
# }
# ```
# Stack the whole dataset and it would be of the format: 
# ```
# X.shape == [N, 60, 25, 6]
# y.shape == [N]
# ```

# ### Steps: 
# 1) Split the input dataset into normal/abnormal motion (train (only anomaly), test (anomaly and normal clips))
# 2) Extract motionCLIP embeddings from the input data (NTU-RGB+D)
# 3) Fit Gaussian model
# 4) Score the test CLIPS (using Mahalanobis distance)
# 5) Pick threshold 
# 6) Make anomaly decision
# 7) Evaluate

# In[21]:


import numpy as np
import torch
import os
import sys
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)


# #### Step 1: Split the input dataset

# In[3]:


def make_markovitz_split(X, y, normal_classes=(28, 29, 30, 33), label_base="auto"):
    """
    Markovitz-style Few-vs-Many split:
      - train: only normal classes
      - test normal: normal classes
      - test abnormal: all non-normal classes

    Parameters
    ----------
    X : np.ndarray
        Shape [N, 60, 25, 6]
    y : np.ndarray
        Shape [N]
    normal_classes : iterable of int
        NTU class ids from the paper, e.g. Office split: (28, 29, 30, 33)
    label_base : {"auto", 0, 1}
        Whether y is 0-based [0..59], 1-based [1..60], or infer automatically.

    Returns
    -------
    dict with:
        X_train_normal, y_train_normal
        X_test_normal,  y_test_normal
        X_test_abnormal, y_test_abnormal
        normal_mask
    """
    X = np.asarray(X)
    y = np.asarray(y)
    normal_classes = np.asarray(list(normal_classes))

    if label_base == "auto":
        if y.min() == 0:
            normal_cmp = normal_classes - 1
        else:
            normal_cmp = normal_classes
    elif label_base == 0:
        normal_cmp = normal_classes - 1
    elif label_base == 1:
        normal_cmp = normal_classes
    else:
        raise ValueError("label_base must be 'auto', 0, or 1")

    normal_mask = np.isin(y, normal_cmp)
    abnormal_mask = ~normal_mask

    X_train_normal = X[normal_mask]
    y_train_normal = y[normal_mask]

    X_test_normal = X[normal_mask] #TODO: same normal samples are used for training and testing, but maybe later try to keep one class out of normal and use that as normal for testing only
    y_test_normal = y[normal_mask]

    X_test_abnormal = X[abnormal_mask]
    y_test_abnormal = y[abnormal_mask]

    return {
        "X_train_normal": X_train_normal,
        "y_train_normal": y_train_normal,
        "X_test_normal": X_test_normal,
        "y_test_normal": y_test_normal,
        "X_test_abnormal": X_test_abnormal,
        "y_test_abnormal": y_test_abnormal,
        "normal_mask": normal_mask,
    }


# In[4]:


# based on the office split they used in the paper, which is classes 28, 29, 30, and 33 (1-based ids)
# OFFICE_NORMAL_CLASSES = [28, 29, 30, 33]
# TRIAL_NORMAL_CLASSES = [""drink water", "eat meal/snack", "reading", "writing"]
TRIAL_NORMAL_CLASSES = [1, 2, 11, 12]
#TRIAL_NORMAL_CLASSES = [1,2]
X = np.load("/scratch/mgirishnair/MotionCLIP_ready_datasetTest/X.npy")
y = np.load("/scratch/mgirishnair/MotionCLIP_ready_datasetTest/y.npy")
splits = make_markovitz_split(
    X, y,
    normal_classes=TRIAL_NORMAL_CLASSES,
    label_base="auto"
)


# In[5]:


X_train_normal = splits["X_train_normal"]
X_test_normal = splits["X_test_normal"]
X_test_abnormal = splits["X_test_abnormal"]

y_train_normal = splits["y_train_normal"]
y_test_normal = splits["y_test_normal"]
y_test_abnormal = splits["y_test_abnormal"]


# In[7]:


print(len(X_train_normal), len(X_test_normal), len(X_test_abnormal))
print(len(y_train_normal), len(y_test_normal), len(y_test_abnormal))
print(y_train_normal, y_test_normal, y_test_abnormal)
print(X_train_normal.shape, X_test_normal.shape, X_test_abnormal.shape)


# #### Step 2: Extract MotionCLIP embeddings of the dataset

# In[10]:


from MotionCLIP.src.models.architectures.transformer import Encoder_TRANSFORMER


# In[11]:


MOTIONCLIP_REPO = r"MotionCLIP" 
CHECKPOINT_PATH = os.path.join(
    MOTIONCLIP_REPO,
    "exps",
    "paper-model",
    "checkpoint_0100.pth.tar"
)

sys.path.append(MOTIONCLIP_REPO)


# In[13]:


class NTURot6dDataset(Dataset):
    def __init__(self, X):
        assert isinstance(X, np.ndarray), "X must be a numpy array"
        assert X.ndim == 4, f"Expected 4D array, got shape {X.shape}"
        assert X.shape[1:] == (60, 25, 6), f"Expected [N, 60, 25, 6], got {X.shape}"

        self.X = X.astype(np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        pose = self.X[idx]                    # [60, 25, 6]
        pose = np.transpose(pose, (1, 2, 0)) # -> [25, 6, 60]

        return {
            "x": torch.from_numpy(pose),      # [25, 6, 60]
            "y": torch.tensor(0, dtype=torch.long),   # dummy class
            "lengths": torch.tensor(60, dtype=torch.long),
        }


# In[14]:


def collate_motionclip(batch):
    x = torch.stack([b["x"] for b in batch], dim=0)          # [B, 25, 6, 60]
    y = torch.stack([b["y"] for b in batch], dim=0)          # [B]
    lengths = torch.stack([b["lengths"] for b in batch], 0)  # [B]

    T = x.shape[-1]
    mask = torch.arange(T).unsqueeze(0) < lengths.unsqueeze(1)   # [B, 60]

    return {
        "x": x,
        "y": y,
        "lengths": lengths,
        "mask": mask,
    }


# In[15]:


def build_motionclip_encoder(checkpoint_path, device):
    encoder = Encoder_TRANSFORMER(
        modeltype="motionclip",
        njoints=25,
        nfeats=6,
        num_frames=60,
        num_classes=1,                # only need dummy class 0 for inference
        translation=True,
        pose_rep="rot6d",
        glob=True,
        glob_rot=[math.pi, 0.0, 0.0],
        latent_dim=512,
        ff_size=1024,
        num_layers=8,
        num_heads=4,
        dropout=0.1,
        ablation=None,
        activation="gelu",
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    # The saved checkpoint contains full model weights. We only need encoder.*
    encoder_state = {}
    for k, v in ckpt.items():
        if k.startswith("encoder."):
            encoder_state[k[len("encoder."):]] = v

    missing, unexpected = encoder.load_state_dict(encoder_state, strict=False)

    if unexpected:
        raise RuntimeError(f"Unexpected encoder keys: {unexpected}")
    if missing:
        print("Warning: missing encoder keys:", missing)

    encoder = encoder.to(device)
    encoder.eval()
    print("MotionCLIP encoder built")
    return encoder


# In[16]:


@torch.no_grad()
def extract_motionclip_embeddings(encoder, X, batch_size=32, device="cuda"):
    dataset = NTURot6dDataset(X)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_motionclip,
    )

    all_embeddings = []

    for batch in loader:
        batch["x"] = batch["x"].to(device).float()
        batch["y"] = batch["y"].to(device)
        batch["lengths"] = batch["lengths"].to(device)
        batch["mask"] = batch["mask"].to(device)

        out = encoder(batch)
        z = out["mu"]                  # [B, 512]

        all_embeddings.append(z.cpu())
    print("Embeddings generated")
    return torch.cat(all_embeddings, dim=0).numpy()


# In[17]:


device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device used: " + device)
encoder = build_motionclip_encoder(CHECKPOINT_PATH, device=device)

Z_train_normal = extract_motionclip_embeddings(
    encoder,
    X_train_normal,
    batch_size=32,
    device=device,
)

Z_test_normal = extract_motionclip_embeddings(
    encoder,
    X_test_normal,
    batch_size=32,
    device=device,
)

Z_test_abnormal = extract_motionclip_embeddings(
    encoder,
    X_test_abnormal,
    batch_size=32,
    device=device,
)

print("Z_train_normal:", Z_train_normal.shape)
print("Z_test_normal:", Z_test_normal.shape)
print("Z_test_abnormal:", Z_test_abnormal.shape)


# #### Step 3: Fit Gaussian Model

# In[18]:


mu = np.mean(Z_train_normal, axis=0)
cov = np.cov(Z_train_normal, rowvar=False)
cov_inv = np.linalg.pinv(cov)


# #### Step 4: Score the test samples

# In[19]:


def mahalanobis(x, mu, cov_inv):
    d = x - mu
    return np.sqrt(d @ cov_inv @ d.T)


# In[20]:


scores_normal = np.array([mahalanobis(z, mu, cov_inv) for z in Z_test_normal])
scores_abnormal = np.array([mahalanobis(z, mu, cov_inv) for z in Z_test_abnormal])


# ### Temporary: Diagonal gaussian since there are only 4 datapoints for normal

# In[22]:


# --------------------------------------------------
# 1. Fit diagonal Gaussian on normal training embeddings
# --------------------------------------------------
def fit_diagonal_gaussian(Z_train_normal, eps=1e-6):
    mu = np.mean(Z_train_normal, axis=0)              # [D]
    var = np.var(Z_train_normal, axis=0) + eps        # [D]
    return mu, var


# In[23]:


# --------------------------------------------------
# 2. Diagonal Mahalanobis-style anomaly score
# --------------------------------------------------
def diagonal_gaussian_score(Z, mu, var):
    # Z: [N, D]
    # returns: [N]
    return np.sqrt(np.sum(((Z - mu) ** 2) / var, axis=1))


# In[24]:


# --------------------------------------------------
# 3. Fit model
# --------------------------------------------------
mu, var = fit_diagonal_gaussian(Z_train_normal)

train_scores = diagonal_gaussian_score(Z_train_normal, mu, var)
scores_normal = diagonal_gaussian_score(Z_test_normal, mu, var)
scores_abnormal = diagonal_gaussian_score(Z_test_abnormal, mu, var)


# In[25]:


# --------------------------------------------------
# 4. Combine test scores for ranking metrics
#    label 0 = normal, 1 = abnormal
# --------------------------------------------------
y_true = np.concatenate([
    np.zeros(len(scores_normal), dtype=int),
    np.ones(len(scores_abnormal), dtype=int),
])

y_scores = np.concatenate([scores_normal, scores_abnormal])


# In[26]:


# --------------------------------------------------
# 5. Ranking-based metrics
# --------------------------------------------------
auroc = roc_auc_score(y_true, y_scores)
prauc = average_precision_score(y_true, y_scores)

print(f"AUROC : {auroc:.4f}")
print(f"PR-AUC: {prauc:.4f}")


# In[27]:


# --------------------------------------------------
# 6. Threshold selection
#    Use only train-normal scores
#    Example: 95th percentile
# --------------------------------------------------
threshold = np.percentile(train_scores, 95)

print(f"Threshold (95th percentile of train-normal scores): {threshold:.4f}")


# In[28]:


# --------------------------------------------------
# 7. Convert scores to predictions
#    score > threshold => abnormal
# --------------------------------------------------
y_pred = (y_scores > threshold).astype(int)

cm = confusion_matrix(y_true, y_pred)

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["normal", "abnormal"]))


# In[29]:


print("Train normal scores: ", train_scores)
print("Test normal scores:  ", scores_normal)
print("Test abnormal scores:", scores_abnormal)

