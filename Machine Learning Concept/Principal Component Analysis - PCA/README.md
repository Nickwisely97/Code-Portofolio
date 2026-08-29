# Principal Component Analysis — PCA (concept demo)

Dimensionality reduction on real pixel data (`sklearn.datasets.load_digits`, 1,797 8x8 handwritten-digit images = 64 pixel-features each), with a clear stated objective and every claim backed by a measured number rather than assumed: baseline vs. PCA-reduced classifier accuracy, a full accuracy-vs-training-speed sweep across every component count from 1 to 64, a scree plot, and a 2D projection visualizing how much class separation survives an extreme 64-to-2 compression. Full reasoning is in the notebook itself.

**Status:** flagged for future integration into a full applied "Segmentation" project (RFM + PCA + K-Means) — see the repo root README. Kept standalone here for now rather than moved, since that merge is a separate, not-yet-started piece of work.

## How to run
Open `code/pca_analysis.ipynb` and run top to bottom. Plots saved to `result/`.
