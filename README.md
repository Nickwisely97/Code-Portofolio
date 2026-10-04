# Code Portfolio — Nick Wisely

Applied data science projects, organized by technique. Each folder matches a skill category on my CV, so every claim there has working code behind it.

## Featured projects

| | |
|---|---|
| **[Self-Order Terminal Staffing](<Project/Optimization & Operations Research/Self-Order Terminal Staffing - Erlang C Queueing Model>)**<br>Should a restaurant replace cashiers with kiosks? A two-stage queue simulation shows the kitchen, not the front end, caps capacity at ~111 orders/hour, and that kiosk-only is the most labor-efficient setup today. | ![](<Project/Optimization & Operations Research/Self-Order Terminal Staffing - Erlang C Queueing Model/result/figures/p99_vs_arrival_rate.png>) |
| **[Headcount Attrition — Survival Analysis](<Project/Forecasting & Predictive Modeling/Headcount Attrition - Survival Analysis>)**<br>One Cox model ranks who is likely to resign next (C-index 0.85) and explains why. Employees working overtime resign at 2.2× the rate of those who don't. | ![](<Project/Forecasting & Predictive Modeling/Headcount Attrition - Survival Analysis/result/figures/hazard_ratio_barplot.png>) |
| **[Temperature Forecast — LightGBM](<Project/Forecasting & Predictive Modeling/Temperature Forecast - LightGBM Multi-Horizon>)**<br>Hourly 14-day forecasts for five Indonesian cities from live BMKG and Open-Meteo data, validated with a leak-free walk-forward backtest (MAE 0.6–1.8 °C). | ![](<Project/Forecasting & Predictive Modeling/Temperature Forecast - LightGBM Multi-Horizon/result/figures/jakarta/forecast_fan.png>) |
| **[Battleship — Monte Carlo Strategy Simulation](<Project/Game Analysis/Battleship - Monte Carlo Strategy Simulation>)**<br>A probability-density attacker sinks the fleet in 45 shots vs 95 for random firing, and any predictable ship placement collapses once the opponent learns it. | ![](<Project/Game Analysis/Battleship - Monte Carlo Strategy Simulation/result/figures/shots_cdf.png>) |

## All projects

### Forecasting & Predictive Modeling
| Project | Technique | Key result |
|---|---|---|
| [Headcount Attrition — Survival Analysis](<Project/Forecasting & Predictive Modeling/Headcount Attrition - Survival Analysis>) | Cox Proportional Hazards, Kaplan-Meier | C-index 0.85; overtime raises resignation risk 2.2× |
| [Temperature Forecast — LightGBM Multi-Horizon](<Project/Forecasting & Predictive Modeling/Temperature Forecast - LightGBM Multi-Horizon>) | LightGBM, direct multi-horizon time series, walk-forward backtest | 14-day hourly MAE 0.6–1.8 °C across 5 cities |

### Optimization & Operations Research
| Project | Technique | Key result |
|---|---|---|
| [Self-Order Terminal Staffing — Queueing Simulation](<Project/Optimization & Operations Research/Self-Order Terminal Staffing - Erlang C Queueing Model>) | Discrete-event simulation (SimPy), tandem queues, P99 SLA | Kitchen caps capacity at ~111/hr; kiosk-only is most labor-efficient |
| [Staffing Optimization — Linear Programming](<Project/Optimization & Operations Research/Staffing Optimization - Linear Programming>) | Integer linear programming (PuLP) | Minimum 44 staff to cover 24-hour demand |

### Exploratory Data Analysis & Reporting
| Project | Technique | Key result |
|---|---|---|
| [BukaToko Funnel Conversion — Clickstream Analysis](<Project/Exploratory Data Analysis & Reporting/BukaToko Funnel Conversion - Clickstream Analysis>) | EDA, data cleaning, funnel analysis, executive reporting | 75% of users who add to cart never buy |

### Segmentation & Recommendation
| Project | Technique | Key result |
|---|---|---|
| [Complaint Segmentation — K-Means Clustering](<Project/Segmentation & Recommendation/Complaint Segmentation - K-Means Clustering>) | TF-IDF + LSA, K-Means, Elbow Method | 5 of 7 topics separate cleanly (ARI 0.29, NMI 0.39) |

### Game Analysis
Board games used as a **public, non-confidential stand-in** for the decision-modeling and stochastic-process work I do professionally: same techniques, no proprietary data.

| Project | Technique | Key result |
|---|---|---|
| [Battleship — Monte Carlo Strategy Simulation](<Project/Game Analysis/Battleship - Monte Carlo Strategy Simulation>) | Monte Carlo simulation, probability modeling, paired hypothesis tests | Best strategy needs 45 shots vs 95 for random |
| [Snake and Ladder — Markov Chain Analysis](<Project/Game Analysis/Snake and Ladder - Markov Chain Analysis>) | Absorbing Markov chains | 3 dice per turn is fastest; more dice slows the game |
| [Congklak — Minimax & Alpha-Beta Search](<Project/Game Analysis/Congklak - Minimax & Alpha-Beta Search>) | Adversarial search, tournament simulation | 6-move search wins 83% of games overall |
| [Tournament Bracket — Score Tracker](<Project/Game Analysis/Tournament Bracket - Score Tracker>) | HTML/JS utility | Planned upgrade: Swiss-pairing optimizer |

## Machine Learning Concept — technique demos
Short, self-contained notebooks on a single technique:
- [Principal Component Analysis — PCA](<Machine Learning Concept/Principal Component Analysis - PCA>): how far 64 pixel features can be compressed before accuracy suffers.
- [Regularization](<Machine Learning Concept/Regularization>): OLS vs Ridge vs Lasso vs ElasticNet, including Lasso's built-in feature selection.

## Roadmap
- A/B testing and SQL analysis project
- Recommender system on real user ratings
- RFM customer segmentation (with PCA + K-Means)
- Weibull reliability analysis
- Duplicate-ticket detection with trigram TF-IDF

## Conventions
- Every project folder has `code/`, optionally `data/` and `result/`, a `README.md`, and a `requirements.txt`.
- Executive PowerPoint reports share one design system: [`Project/Executive_Report_Template`](Project/Executive_Report_Template).
