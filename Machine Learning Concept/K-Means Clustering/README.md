# K-Means Clustering — Complaint Topic Segmentation (concept demo)

Unsupervised clustering of **real, unlabeled text** (20 Newsgroups, reframed as 7 complaint/support-ticket topics) via TF-IDF -> Truncated SVD (LSA) -> K-Means, validated against the true category labels (ARI = 0.29, NMI = 0.39) and interpreted through top-terms-per-cluster and a cluster-vs-category heatmap. Full write-up and reasoning are in the notebook itself.

Previously this was a toy 22-row Age/Income table (classic tutorial data) with no real validation. Rebuilt entirely against real text data with a proper unsupervised-evaluation methodology, at the user's request, to demonstrate segmentation that's actually usable operationally rather than a syntax demo.

**Status:** flagged for future integration into a full applied "Segmentation" project (RFM + PCA + K-Means, on a player/customer behavior dataset) — see the repo root README. Kept standalone here for now rather than moved, since that merge is a separate, not-yet-started piece of work.

## How to run
Open `code/kmeans_algorithms.ipynb` and run top to bottom. First run downloads and caches the 20 Newsgroups corpus via scikit-learn (outside this repo, under `~/scikit_learn_data`); the exact filtered subset actually used is then snapshotted to `data/complaint_topics_subset.parquet` for reproducibility. Plots saved to `result/`.
