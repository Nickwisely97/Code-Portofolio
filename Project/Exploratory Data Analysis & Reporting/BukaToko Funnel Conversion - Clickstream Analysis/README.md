# BukaToko Funnel Conversion — Clickstream Analysis

**Skills:** Exploratory data analysis, data-quality auditing and cleaning, funnel and conversion analysis, executive reporting.

Case study completed for RevoU's Data Analytics mini course.

## Problem
BukaToko, a fictional e-commerce company, wants to understand user activity across countries, devices and channels, and find where its purchase funnel (Browse → Add to Cart → Checkout → Purchase) loses customers.

## Key results
- **Cart abandonment is the biggest leak:** 75% of users who added something to their cart never bought anything.
- **Browse → Add to Cart is the slowest step in the funnel.** Retargeting browsers who never add to cart is a better first investment than optimizing checkout or payment.
- **Indonesia** leads Q2 2025 active users, well ahead of the US and Vietnam, so it is the highest-leverage market for conversion tests.
- **Device doesn't gate any funnel step:** every event type splits about 49 / 40 / 10% across Android / iOS / Desktop, the same as overall usage.

## Data
`data/Dataset Case Study - dirty_dummy_events.csv`: ~10,000 clickstream events (event type, timestamp, user, session, product, country, device, channel), deliberately "dirty":
- inconsistent device casing (`IOS` / `android` / `desktop`)
- ~1% missing `channel`
- a partial first month

**Limitation found in the audit:** every `session_id` holds exactly one event, so sessions can't link a user's journey. Throughout the analysis, (user, calendar date) is used as a proxy for one visit.

## Method
1. Data-quality audit and cleaning (casing, missing values, data types)
2. Scope trim: drop the partial first month
3. Funnel construction: each user's furthest stage reached
4. Lead time between funnel stages
5. Best-sellers and purchase timing (day of week, hour)
6. Cart-abandonment breakdown and recommendations
7. 9-slide executive PowerPoint report

## Project structure
```
code/
  BukaToko_Funnel_Analysis.ipynb  <- the analysis, STEP 00–13
  build_notebook.py               <- generates the notebook's cells
  report_builder.py               <- recomputes the metrics from the raw CSV and builds the deck
result/slides/                    <- BukaToko_Report_-_Nick_Wisely.pptx (the submitted deck)
```

## How to run
```
pip install -r requirements.txt
```
Open `code/BukaToko_Funnel_Analysis.ipynb` and run it top to bottom, or run `python code/report_builder.py` to regenerate the executive deck.
