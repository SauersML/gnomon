"""Training-defined audit regions and paired held-out probability losses."""
from __future__ import annotations

import numpy as np
import pandas as pd


def audit_groups(train, test):
    groups = [("overall", np.ones(len(test), dtype=bool))]
    groups += [(f"ancestry:{a}", test.ancestry.eq(a).to_numpy()) for a in sorted(test.ancestry.unique())]
    groups += [(f"sex:{s}", test.sex.eq(s).to_numpy()) for s in (0, 1)]
    groups += [(f"age:{lo}-{hi}", ((test.age0 >= lo) & (test.age0 < hi)).to_numpy())
               for lo, hi in ((18, 40), (40, 60), (60, 1000))]
    pcs = [name for name in train if name.startswith("PC") and name[2:].isdigit()]
    x, y = train[pcs].to_numpy(float), test[pcs].to_numpy(float)
    center, scale = x.mean(axis=0), x.std(axis=0)
    if not len(pcs) or not np.isfinite(x).all() or not np.isfinite(y).all() or (scale <= 0).any():
        raise ValueError("PC audit geometry needs finite, varying training PCs")
    x, y = (x - center) / scale, (y - center) / scale
    # Greedy space-filling anchors learned only from training covariates.
    anchors = [int(np.argmin(np.sum(x*x, axis=1)))]
    nearest = np.sum((x - x[anchors[0]])**2, axis=1)
    for _ in range(min(6, len(x)) - 1):
        idx = int(np.argmax(nearest))
        if nearest[idx] <= 0:
            break
        anchors.append(idx)
        nearest = np.minimum(nearest, np.sum((x - x[idx])**2, axis=1))
    train_distance = np.sum((x[:, None, :] - x[anchors][None, :, :])**2, axis=2)
    test_distance = np.sum((y[:, None, :] - x[anchors][None, :, :])**2, axis=2)
    train_labels, test_labels = train_distance.argmin(axis=1), test_distance.argmin(axis=1)
    supported = np.zeros(len(test), dtype=bool)
    for label in range(len(anchors)):
        radius = np.quantile(train_distance[train_labels == label, label], .95)
        mask = (test_labels == label) & (test_distance[:, label] <= radius)
        groups.append((f"pc_neighborhood:{label}", mask))
        supported |= mask
    groups.append(("pc_outside_training_support", ~supported))
    return groups


def paired_loss_summary(reference_loss, model_loss, mask, group_ids):
    """Cluster-robust SE, conditional on fitted models and censoring weights."""
    delta = (np.asarray(reference_loss) - np.asarray(model_loss))[mask]
    groups = np.asarray(group_ids)[mask]
    mean = float(delta.mean())
    cluster_sums = pd.Series(delta - mean).groupby(groups, sort=False).sum().to_numpy()
    count = len(cluster_sums)
    if count < 2:
        raise ValueError("paired uncertainty needs at least two held-out groups")
    se = float(np.sqrt(count / (count - 1) * np.sum(cluster_sums**2) / len(delta)**2))
    return {"brier_improvement": mean, "standard_error": se,
            "conditional_95_interval": [mean - 1.959963984540054*se, mean + 1.959963984540054*se]}
