# K-Means Clustering — Customer Complaint Segmentation

**CV skill represented:** unsupervised segmentation (K-Means), applied to a customer-complaint-triage use case.

## Structure

```
code/
  kmeans_algorithms.ipynb        <- the project: K-Means method, Elbow, fitting, evaluation
data/
  text_feature_engineering.ipynb <- data prep only: text -> numeric feature matrix
  complaint_features.parquet     <- output of the notebook above
  complaint_topics_subset.parquet
result/
```

**`code/kmeans_algorithms.ipynb`** is the actual point of this project and the only notebook meant to be read as "the analysis": what K-Means does, choosing `k` with the Elbow Method (without peeking at ground truth), fitting, and evaluating the result against the true labels (ARI = 0.29, NMI = 0.39), interpreted through a cluster-vs-category heatmap and by reading real example complaints per cluster. It never mentions how its input features were built — that's a deliberately separate concern.

**`data/text_feature_engineering.ipynb`** is the notebook that builds `complaint_features.parquet` and `complaint_topics_subset.parquet` (real, unlabeled text — 20 Newsgroups, reframed as 7 complaint/support-ticket topics — turned into a numeric feature matrix). It lives in `data/` rather than `code/` on purpose: it isn't part of the K-Means analysis, it's the reproducible record of how the data sitting next to it came to exist. Putting it in `code/` would make it look like a second, equally-important piece of the analysis and dilute what this project is actually demonstrating; keeping it in `data/` signals "read this only if you need to know where the numbers came from or want to regenerate them," while `code/` stays exclusively about K-Means.

Previously this was a toy 22-row Age/Income table (classic tutorial data) with no real validation. Rebuilt against real text data with a proper unsupervised-evaluation methodology, then moved here from `Machine Learning Concept/` once it became a real applied project rather than a bare technique demo.

## How to run
Run `data/text_feature_engineering.ipynb` first, then `code/kmeans_algorithms.ipynb`. The first run of the feature-engineering notebook downloads and caches the 20 Newsgroups corpus via scikit-learn (outside this repo, under `~/scikit_learn_data`); its outputs are the two files listed above. Plots from the K-Means notebook are saved to `result/`.
