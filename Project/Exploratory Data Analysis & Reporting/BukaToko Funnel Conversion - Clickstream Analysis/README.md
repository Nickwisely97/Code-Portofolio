# BukaToko Funnel Conversion — Clickstream Analysis

**CV skill represented:** exploratory data analysis, data cleaning/auditing, and executive reporting.

Case study completed for RevoU's Data Analytics mini course.

## Problem
BukaToko (fictional e-commerce case study) wants to understand user activity across countries and channels, and find where the purchase funnel — Browse → Add to Cart → Checkout → Purchase — leaks conversion.

## Data
`data/Dataset Case Study - dirty_dummy_events.csv` — ~10,000 clickstream events (`event_type`, timestamp, `user_id`, `session_id`, `product_id`, `country`, `device`, `channel`), deliberately "dirty" for the exercise: inconsistent device casing (`IOS`/`android`/`desktop`), ~1% missing `channel`, and a partial first month (data starts 2025-03-27).

**Key structural limitation, found during the audit:** every `session_id` maps to exactly one event, so it can't link a user's multi-event journey — there's no way to group "one visit" from raw session id alone. Workaround used throughout: (`user_id`, calendar date) as a same-visit proxy wherever that matters (e.g. same-day search → add-to-cart).

## Method
Standard EDA pipeline, walked step-by-step in the notebook: data-quality audit → cleaning (device-casing fix, `channel` fill, dtype casting) → scope trim (drop the partial March data, keep partial September with a proportional forecast) → funnel construction (map raw events to Browse/Cart/Checkout/Purchase stages, take each user's highest stage reached) → lead-time analysis between stages → best-sellers and purchase-timing breakdowns → cart-abandonment analysis.

**Key findings:**
- **Indonesia** leads Q2 2025 active users, well ahead of the US and Vietnam — the highest-leverage market for conversion tests.
- **Android** is the most-used device by session, but device doesn't gate any specific funnel action — every event type splits ~49/40/10% Android/iOS/Desktop, matching the overall mix.
- **Cart abandonment is the single biggest leak:** 75% of users who ever added something to cart never completed a purchase.
- **Browse → Add to Cart is the slowest, best-supported stage transition** — worth prioritizing retargeting for browsers who never added to cart, rather than assuming the bottleneck is late-funnel (checkout/payment).

## Visuals
Charts are built inline in the notebook (Plotly for the country comparison, Matplotlib/Seaborn for everything else — MAU small multiples, device usage, funnel bars, lead-time boxplots, best-sellers, day/hour purchase patterns, cart-abandonment breakdown) rather than saved as standalone PNGs; the executive deck below carries native, editable versions of the same charts.

## Output layout
```
result/
  slides/   -- BukaToko_Report_-_Nick_Wisely.pptx  (submitted RevoU deliverable, kept as-is)
```
The deck was built with `code/report_builder.py` against the shared `Executive_Report_Template/` design system (same one used by this portfolio's other executive reports). Re-running `report_builder.py` regenerates a freshly-dated `Executive_BukaToko_Funnel_Report_<date>.pptx` from the same underlying data/logic; the committed file is the exact version submitted for the course, kept verbatim rather than overwritten.

## Code structure
- `code/build_notebook.py` — assembles `BukaToko_Funnel_Analysis.ipynb` cell-by-cell via `nbformat` (source of truth for the notebook's content).
- `code/BukaToko_Funnel_Analysis.ipynb` — the executed step-by-step walkthrough (STEP 00–13: audit, cleaning, scope, funnel, lead time, best sellers, abandonment, recommendations).
- `code/report_builder.py` — recomputes the same metrics directly from the raw CSV (never hardcodes a result) and renders the 9-slide executive PPTX.

## How to run
Open `code/BukaToko_Funnel_Analysis.ipynb` and run top to bottom, or run `python code/report_builder.py` to (re)generate the executive deck.
