# Wine Recommendation — K-Nearest Neighbors

**CV skill represented:** unsupervised/similarity-based modeling (general ML breadth).

## Problem
Given a wine a taster likes, recommend chemically similar wines from the catalog — no other tasters or ratings required, unlike collaborative filtering.

## Data
The UCI Machine Learning Repository's Wine recognition dataset (Forina et al., 1991) — the complete 178-wine catalog, 3 cultivars grown in the same region of Italy, 13 lab-measured chemical attributes each (alcohol, phenols, color intensity, proline, etc.). Bundled in scikit-learn via `sklearn.datasets.load_wine()`; exported to `data/wine.csv` for a visible, portable copy.

This project originally targeted movie ratings via user-user collaborative filtering (see git history), but the standard real-data source for that (MovieLens, via files.grouplens.org) turned out to be unreachable from this environment — that server's TLS certificate is expired. Rather than force a workaround, the project pivoted to a genuinely real, reliably-available dataset and a complementary technique: content-based filtering instead of collaborative filtering.

## Method
Standardize the 13 chemical attributes (z-score, so no single attribute dominates purely due to scale), then use **K-Nearest Neighbors** — cosine similarity in that standardized space — to rank the catalog. Validated with **precision@k**: for every wine, check whether its k nearest neighbors actually share its cultivar (its real, known style) — using the dataset's ground-truth labels as an honest correctness check, not just an anecdotal example.

**Result:** precision@5 averages 93.0% across all 178 wines (vs. 33.3% for a random guess among 3 cultivars), and stays flat out to k=10.

## Visuals
- `eda_boxplots.png` — cultivar separability on a few key attributes.
- `precision_at_k.png` — recommendation quality vs. neighborhood size, against a random-guess baseline.
- `radar_chart.png` — the genre/preference-fit visual: a reference wine's standardized chemical profile overlaid against its top-3 recommendations, so the match is visible attribute-by-attribute, not just asserted.
- `pca_landscape.png` — all 178 wines in 2D (PCA), colored by cultivar, with the reference wine and its recommendations highlighted.

## Output layout
```
result/
  figures/   -- all chart PNGs (eda, precision@k, radar, PCA)
  slides/    -- Executive_Wine_Recommendation_Report_<date>.pptx
```
Same `figures/` + `slides/` convention as this portfolio's other executive reports (Headcount Attrition, Temperature Forecast), built by `code/report_builder.py`.

## Code structure
- `code/wine_recommendation.ipynb` — orchestrates the analysis, STEP by STEP.
- `code/recommender.py` — feature standardization, similarity matrix, `recommend()`, `precision_at_k()` / `precision_at_k_curve()`.
- `code/report_builder.py` — 4-slide executive PPTX (summary, recommendation quality, sample walkthrough, similarity landscape).

## How to run
Open `code/wine_recommendation.ipynb` and run top to bottom — data loads directly from scikit-learn, no download needed.
