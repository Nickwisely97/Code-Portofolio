# K-Means Clustering — Complaint Topic Segmentation (concept demo)

Split into two notebooks by concern, run in order:

1. **`code/text_feature_engineering.ipynb`** — turns real, unlabeled text (20 Newsgroups, reframed as 7 complaint/support-ticket topics: PC Hardware, Mac Hardware, Automotive, Motorcycle/Vehicle, Health/Medical Product, Electronics Product, Billing/Marketplace) into a numeric feature matrix via TF-IDF -> Truncated SVD (LSA). Saves the feature matrix and fitted vectorizer/SVD to `data/` for reuse.
2. **`code/kmeans_algorithms.ipynb`** — the actual K-Means demo: what the algorithm does, choosing `k` (elbow + silhouette, without peeking at ground truth), fitting, and evaluating the result against the true labels (ARI = 0.29, NMI = 0.39), interpreted through top-terms-per-cluster and a cluster-vs-category heatmap.

Kept as two files rather than one so the K-Means notebook is actually about K-Means — text vectorization is a separate skill with its own reasoning (stopword tuning, TF-IDF weighting, LSA) that would otherwise crowd out the clustering method itself.

Previously this was a toy 22-row Age/Income table (classic tutorial data) with no real validation. Rebuilt against real text data with a proper unsupervised-evaluation methodology, at the user's request, to demonstrate segmentation that's actually usable operationally rather than a syntax demo.

**Status:** flagged for future integration into a full applied "Segmentation" project (RFM + PCA + K-Means, on a player/customer behavior dataset) — see the repo root README. Kept standalone here for now rather than moved, since that merge is a separate, not-yet-started piece of work.

## How to run
Run `code/text_feature_engineering.ipynb` first, then `code/kmeans_algorithms.ipynb`. First run of the feature-engineering notebook downloads and caches the 20 Newsgroups corpus via scikit-learn (outside this repo, under `~/scikit_learn_data`); its own outputs — a lightweight snapshot of the exact text used, the LSA feature matrix, and the fitted vectorizer/SVD — are all saved to `data/` so the K-Means notebook never needs to touch the raw corpus. Plots saved to `result/`.
