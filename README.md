# Code Portfolio — Nick Wisely

Applied data science / ML projects, organized by technique rather than by industry — mirroring the **Statistical & ML Modeling** skill categories on my CV, so each folder maps directly to a claim there.

## Project/ — applied case studies

### Forecasting & Predictive Modeling
| Project | Technique | CV skill |
|---|---|---|
| [Headcount Attrition - Survival Analysis](<Project/Forecasting & Predictive Modeling/Headcount Attrition - Survival Analysis>) | Cox Proportional Hazards (forecasting + inference) | Survival & Reliability Analysis |
| [Temperature Forecast — LightGBM Multi-Horizon](<Project/Forecasting & Predictive Modeling/Temperature Forecast - LightGBM Multi-Horizon>) | LightGBM, direct multi-horizon time-series | Forecasting & Time-Series |

### Optimization & Operations Research
| Project | Technique | CV skill |
|---|---|---|
| [Staffing Optimization — Linear Programming](<Project/Optimization & Operations Research/Staffing Optimization - Linear Programming>) | Integer LP (PuLP) | Queueing/Staffing Models |

### Segmentation & Recommendation
| Project | Technique | CV skill |
|---|---|---|
| [K-Means Clustering — Customer Complaint Segmentation](<Project/Segmentation & Recommendation/K-Means Clustering - Complaint Segmentation>) | K-Means Clustering, Elbow Method | Unsupervised segmentation |
| [Wine Recommendation — K-Nearest Neighbors](<Project/Segmentation & Recommendation/Wine Recommendation - K-Nearest Neighbors>) | K-Nearest Neighbors (cosine similarity on standardized chemical attributes), validated with precision@k | general ML |

### Game Analysis
Board games used as a **public, non-confidential stand-in** for the kind of structured decision-modeling and stochastic-process work I do professionally — same techniques, no proprietary data.

| Project | Technique | Represents |
|---|---|---|
| [Congklak — Minimax & Alpha-Beta Search](<Project/Game Analysis/Congklak - Minimax & Alpha-Beta Search>) | Adversarial search, tournament simulation | Algorithmic decision modeling |
| [Snake and Ladder — Markov Chain Analysis](<Project/Game Analysis/Snake and Ladder - Markov Chain Analysis>) | Absorbing Markov chains | Stochastic-process modeling |
| [Tournament Bracket — Score Tracker](<Project/Game Analysis/Tournament Bracket - Score Tracker>) | HTML/JS utility | (planned upgrade → Swiss-pairing LP optimizer) |

## Machine Learning Concept/ — standalone technique demos

Kept separate from `Project/` for now; each is flagged for future integration into an applied project rather than staying a bare demo (see each folder's README):

- [Principal Component Analysis — PCA](<Machine Learning Concept/Principal Component Analysis - PCA>) — planned merge → applied Segmentation project
- [Regularization](<Machine Learning Concept/Regularization>) — planned merge → Temperature Forecast baseline-comparison appendix

(K-Means Clustering has already made this move — see Segmentation & Recommendation above.)

## Gaps vs. CV (tracked, not yet built)
A few CV skills aren't represented in code yet: Erlang C, Discrete Event Simulation, Weibull/reliability analysis, Trigram TF-IDF duplicate detection, RFM segmentation, formal A/B testing. Planned as new Game Analysis or Optimization entries.

## Conventions
- Every project folder: `code/`, optionally `data/` and `result/`, a `README.md`, and a `requirements.txt` where it has external dependencies.
- Notebook/script filenames are snake_case and match the project's subject, no prefixes.
