# Regularization (concept demo)

Standalone comparison of OLS vs. Ridge/Lasso/ElasticNet (all cross-validated) on the sklearn diabetes dataset — R² and coefficient shrinkage side by side.

**Status:** doesn't have a natural standalone "applied" home — flagged to fold into the Temperature Forecast project as a baseline-comparison appendix (regularized linear regression vs. LightGBM), showing the reasoning for reaching for a more complex model rather than jumping straight to it. Kept standalone here for now.

## How to run
Open `code/regularization_analysis.ipynb` and run top to bottom — uses `sklearn.datasets.load_diabetes()`, no external data file.
