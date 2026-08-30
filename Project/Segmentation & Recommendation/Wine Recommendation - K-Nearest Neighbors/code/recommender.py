"""
recommender.py
Content-based K-Nearest Neighbors wine recommender: standardize the 13
lab-measured chemical attributes from the UCI Wine dataset, then recommend
the k wines closest by cosine similarity in that standardized feature space
-- "if you like this wine's chemical profile, here are its k nearest
neighbors", as opposed to a user-user collaborative-filtering approach.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

FEATURE_COLS = [
    "alcohol", "malic_acid", "ash", "alcalinity_of_ash", "magnesium",
    "total_phenols", "flavanoids", "nonflavanoid_phenols", "proanthocyanins",
    "color_intensity", "hue", "od280/od315_of_diluted_wines", "proline",
]


def standardize_features(df, feature_cols=FEATURE_COLS):
    """Z-score every chemical attribute so no single feature (e.g. proline,
    scale ~1000) dominates the similarity purely due to units."""
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols])
    return X, scaler


def build_similarity_matrix(X):
    """Cosine similarity between every pair of wines in standardized feature space."""
    return cosine_similarity(X)


def recommend(sim_matrix, df, wine_id, top_n=5):
    """Top-N wines most similar to `wine_id` (a row label in df), excluding itself."""
    scores = pd.Series(sim_matrix[df.index.get_loc(wine_id)], index=df.index)
    scores = scores.drop(index=wine_id).sort_values(ascending=False)
    top_ids = scores.index[:top_n]
    result = df.loc[top_ids].copy()
    result.insert(0, "similarity", scores.loc[top_ids].values)
    return result


def precision_at_k(sim_matrix, labels, k=5):
    """
    For every wine, check whether its top-k nearest neighbors share its
    cultivar label. Returns (mean precision@k, per-wine precision array) --
    this validates that "chemically similar" actually tracks "same style",
    using the dataset's real cultivar labels as ground truth.
    """
    labels = np.asarray(labels)
    n = len(labels)
    precisions = np.zeros(n)
    for i in range(n):
        order = np.argsort(-sim_matrix[i])
        order = order[order != i][:k]
        precisions[i] = np.mean(labels[order] == labels[i])
    return precisions.mean(), precisions


def precision_at_k_curve(sim_matrix, labels, k_values=range(1, 11)):
    """precision@k averaged across all wines, for a range of k -- shows how
    neighborhood purity degrades as the recommendation list grows."""
    rows = [{"k": k, "precision_at_k": precision_at_k(sim_matrix, labels, k=k)[0]} for k in k_values]
    return pd.DataFrame(rows)
