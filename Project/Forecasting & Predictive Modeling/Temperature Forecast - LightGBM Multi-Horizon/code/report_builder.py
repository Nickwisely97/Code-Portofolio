"""
report_builder.py
Executive PowerPoint report builder for the Temperature Forecast pipeline.

Slide order is deliberately temperature-first: the executive summary and the
per-city forecast fans lead the deck since that's what a reader wants to know
first; backtest accuracy (MAE/RMSE/R² -- validation of the model itself, not
the forecast) is pushed to a single Model Performance slide at the very end.

Design system (navy/gold, header/footer band, stat cards, tables) comes from
Executive_Report_Template/report_template.py, shared across this portfolio's
executive reports -- this file only adds the content specific to this project.
"""

import os
import sys

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Executive_Report_Template")
sys.path.insert(0, _TEMPLATE_DIR)
from report_template import (
    NAVY, GOLD, LABEL_CLR, BODY_CLR, SLIDE_W, MARGIN,
    new_presentation, new_slide, add_text, add_stat_card, add_eyebrow, add_panel, add_table, save_report,
)
from pptx.util import Inches

KICKER = "EXECUTIVE REPORT   |   TEMPERATURE FORECAST"
FOOTER = "LightGBM Direct Multi-Horizon Forecast  |  14-Day Outlook"


def build_executive_report(df_summary, results, output_dir):
    """
    df_summary : DataFrame with columns Station, MAE (°C), RMSE (°C), MAPE (%), R².
    results    : dict {station: {"img_fan": path, "live": DataFrame, ...}} --
                 img_fan is the forecast-fan chart, live is the live 14-day
                 forecast frame (column "predicted_actual") used for the
                 outlook stats (avg/max/min).
    output_dir : the project's "result" directory -- the .pptx is saved to
                 output_dir/slides.

    Returns the saved report path.
    """
    stations = df_summary["Station"].tolist()
    total_pages = 1 + len(stations) + 1  # summary + one per city + performance

    best  = df_summary.loc[df_summary["MAE (°C)"].idxmin()]
    worst = df_summary.loc[df_summary["MAE (°C)"].idxmax()]
    avg_mae = df_summary["MAE (°C)"].mean()

    outlook = {}
    for station in stations:
        live = results.get(station, {}).get("live")
        if live is not None and not live.empty:
            outlook[station] = (
                live["predicted_actual"].mean(),
                live["predicted_actual"].max(),
                live["predicted_actual"].min(),
            )
        else:
            outlook[station] = (None, None, None)

    prs = new_presentation()

    # --- Slide 1: Executive Summary -- 14-day temperature outlook by city ---
    # Temperature first: this is the headline question, so it leads the deck.
    s1 = new_slide(prs, KICKER, FOOTER, "Executive Summary", 1, total_pages)
    cards = [
        (f"{len(stations)}", "Cities forecast"),
        ("14 days", "Forecast horizon"),
    ]
    card_w, gap = 2.12, 0.17
    for i, (value, label) in enumerate(cards):
        left = MARGIN + i * (card_w + gap)
        add_stat_card(s1, left, 1.75, value, label)

    add_eyebrow(s1, MARGIN, 3.44, 9.0, "14-DAY TEMPERATURE OUTLOOK BY CITY")
    outlook_rows = [
        (station,
         f"{avg_t:.1f}°C" if avg_t is not None else "n/a",
         f"{max_t:.1f}°C" if max_t is not None else "n/a",
         f"{min_t:.1f}°C" if min_t is not None else "n/a")
        for station, (avg_t, max_t, min_t) in outlook.items()
    ]
    table_bottom = add_table(
        s1, MARGIN, 3.78, 9.0,
        headers=["City", "Avg Forecast Temp", "Max Forecast Temp", "Min Forecast Temp"],
        rows=outlook_rows,
    )

    add_text(s1, MARGIN, table_bottom + 0.18, 9.0, 0.4, [
        ("Model performance (backtest accuracy) is on the last slide of this deck.", 13, False, LABEL_CLR, True),
    ])

    # --- Slides 2..N: one per city -- live forecast fan only ---
    img_w = 9.0
    fan_top = 2.35
    img_h = img_w * 6 / 14  # matches plot_forecast_fan's figsize=(14, 6)

    for i, (_, row) in enumerate(df_summary.iterrows(), start=2):
        station = row["Station"]
        res = results.get(station, {})
        s = new_slide(prs, KICKER, FOOTER, f"{station} -- 14-Day Forecast", i, total_pages)

        add_eyebrow(s, MARGIN, 1.7, 9.0, "LIVE 14-DAY FORECAST FAN")
        fan_path = res.get("img_fan", "")
        if fan_path and os.path.exists(fan_path):
            s.shapes.add_picture(fan_path, Inches(MARGIN), Inches(fan_top), width=Inches(img_w))

    # --- Final slide: Model Performance -- backtest accuracy by city ---
    # Pushed to the end on purpose: this validates the model, it isn't the
    # forecast itself, so it shouldn't compete with the temperature story.
    sN = new_slide(prs, KICKER, FOOTER, "Model Performance", total_pages, total_pages)
    add_eyebrow(sN, MARGIN, 1.75, 9.0, "BACKTEST ACCURACY BY CITY (WALK-FORWARD ROLLBACK TESTING)")
    perf_rows = [
        (r["Station"], f"{r['MAE (°C)']:.2f}°C", f"{r['RMSE (°C)']:.2f}°C", f"{r['R²']:.2f}")
        for _, r in df_summary.iterrows()
    ]
    perf_bottom = add_table(
        sN, MARGIN, 2.1, 9.0,
        headers=["City", "MAE", "RMSE", "R-squared"],
        rows=perf_rows,
    )

    add_eyebrow(sN, MARGIN, perf_bottom + 0.25, 9.0, "HEADLINE FINDINGS")
    add_panel(sN, MARGIN, perf_bottom + 0.57, SLIDE_W - 2 * MARGIN, 1.3, accent=GOLD)
    add_text(sN, MARGIN + 0.25, perf_bottom + 0.75, SLIDE_W - 2 * MARGIN - 0.5, 1.0, [
        (f"Most accurate: {best['Station']} ({best['MAE (°C)']:.2f}°C MAE). Least accurate: {worst['Station']} ({worst['MAE (°C)']:.2f}°C MAE).", 14, False, BODY_CLR, False),
        (f"Average backtest MAE across all cities: {avg_mae:.2f}°C, from many historical origins each forecast 14 days ahead and scored against what actually happened.", 14, False, BODY_CLR, False),
    ])

    return save_report(prs, output_dir, "Executive_Temperature_Report")
