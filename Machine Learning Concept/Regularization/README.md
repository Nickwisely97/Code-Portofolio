# Regularization — OLS vs. Ridge vs. Lasso vs. ElasticNet (concept demo)

**Skills:** Linear regression, regularization (L1 / L2 / ElasticNet), cross-validated hyperparameter tuning.

Compares plain linear regression with three regularized variants on the scikit-learn diabetes dataset (442 patients, 10 features), each tuned with 5-fold cross-validation.

## Key results
| Model | Test R² |
|---|---|
| OLS | 0.453 |
| Ridge | 0.454 |
| ElasticNet | 0.461 |
| **Lasso** | **0.471** |

- With only 10 features, OLS barely overfits, so regularization gives a small gain.
- **Lasso performs feature selection:** it sets two blood-serum features (`s2`, `s4`) exactly to zero. Ridge keeps every feature and only moderately shrinks the correlated ones.

## How to run
Open `code/regularization_analysis.ipynb` and run it top to bottom. It uses `sklearn.datasets.load_diabetes()`, so no data file is needed.
