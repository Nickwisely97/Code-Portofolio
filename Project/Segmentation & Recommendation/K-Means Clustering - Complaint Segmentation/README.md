# K-Means Clustering — Customer Complaint Segmentation

**CV skill represented:** unsupervised segmentation (K-Means), applied to a customer-complaint-triage use case.

Split into two notebooks by concern, run in order:

1. **`code/text_feature_engineering.ipynb`** — data preparation: turns real, unlabeled text (20 Newsgroups, reframed as 7 complaint/support-ticket topics: PC Hardware, Mac Hardware, Automotive, Motorcycle/Vehicle, Health/Medical Product, Electronics Product, Billing/Marketplace) into a numeric feature matrix. Saves `data/complaint_features.parquet` for the K-Means notebook to load.
2. **`code/kmeans_algorithms.ipynb`** — the actual K-Means demo, and the point of this project: what the algorithm does, choosing `k` with the Elbow Method (without peeking at ground truth), fitting, and evaluating the result against the true labels (ARI = 0.29, NMI = 0.39), interpreted through a cluster-vs-category heatmap and by reading real example complaints per cluster.

Kept as two files rather than one so the K-Means notebook stays entirely about K-Means — how the input features were engineered is a separate, earlier step with its own reasoning, deliberately kept out of this notebook.

Previously this was a toy 22-row Age/Income table (classic tutorial data) with no real validation. Rebuilt against real text data with a proper unsupervised-evaluation methodology, at the user's request, to demonstrate segmentation that's actually usable operationally rather than a syntax demo — then moved here from `Machine Learning Concept/` once it became a real applied project rather than a bare technique demo.

## How to run
Run `code/text_feature_engineering.ipynb` first, then `code/kmeans_algorithms.ipynb`. First run of the feature-engineering notebook downloads and caches the 20 Newsgroups corpus via scikit-learn (outside this repo, under `~/scikit_learn_data`); its outputs — a snapshot of the exact text used and the numeric feature matrix — are saved to `data/` so the K-Means notebook never needs to touch the raw corpus. Plots saved to `result/`.
