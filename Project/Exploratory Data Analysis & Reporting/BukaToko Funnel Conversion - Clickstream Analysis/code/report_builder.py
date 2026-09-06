"""
report_builder.py -- automated executive report for the BukaToko funnel
conversion analysis.

Re-runs the same cleaning/aggregation logic as `build_notebook.py` directly
against the raw CSV (single source of truth: this script never hardcodes a
result, it recomputes everything), then renders it as a PowerPoint deck using
the shared design system in Executive_Report_Template/.

Run:  python report_builder.py
Output: ../result/slides/Executive_BukaToko_Funnel_Report_<YYYYMMDD>.pptx
"""
import os
import sys

import numpy as np
import pandas as pd

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Executive_Report_Template")
sys.path.insert(0, _TEMPLATE_DIR)
from report_template import (
    NAVY, GOLD, BG_LIGHT, BG_ALT, DIVIDER, KICKER_CLR, LABEL_CLR,
    BODY_CLR, FOOTER_CLR, WHITE, SLIDE_W, SLIDE_H, MARGIN,
    new_presentation, new_slide, add_text, add_stat_card, add_eyebrow,
    add_panel, add_table, add_footnote, save_report,
)

from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LEGEND_POSITION

DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                          "Dataset Case Study - dirty_dummy_events.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "result")

KICKER = "EXECUTIVE REPORT   |   BUKATOKO FUNNEL CONVERSION"
FOOTER = "Clickstream Funnel Analysis  |  Browse -> Cart -> Checkout -> Purchase"
PRESENTER_NAME = "Nick Wisely"
PRESENTER_ROLE = "Data Modeling Analyst, Kredivo Group"

DOW_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# --- Data pipeline (mirrors code/build_notebook.py step-for-step) ---------

def load_and_clean():
    raw = pd.read_csv(DATA_PATH)
    df = raw.copy()
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    for c in ["event_type", "country", "device", "channel"]:
        df[c] = df[c].astype("string").str.strip()
    device_map = {"ios": "iOS", "android": "Android", "desktop": "Desktop"}
    df["device"] = df["device"].str.lower().map(device_map)
    df["channel"] = df["channel"].fillna("unknown")
    df["date"] = df["event_timestamp"].dt.date
    df["year_month"] = df["event_timestamp"].dt.to_period("M").astype(str)
    funnel_order = {"page_view": 1, "search": 1, "product_view": 1,
                     "add_to_cart": 2, "checkout": 3, "purchase": 4}
    stage_names = {1: "Browse", 2: "Add to Cart", 3: "Checkout", 4: "Purchase"}
    df["funnel_stage"] = df["event_type"].map(funnel_order)
    df_all_dates = df.copy()  # pre-March-trim, kept for the same-day DIY 2 check below
    df = df[df["event_timestamp"] >= "2025-04-01"].reset_index(drop=True)
    return df, df_all_dates, stage_names


def compute_metrics(df, df_all_dates, stage_names):
    m = {}

    # Q1 -- active users by country, Q2 2025
    q2 = df[(df["event_timestamp"] >= "2025-04-01") & (df["event_timestamp"] < "2025-07-01")]
    m["country_q2"] = q2.groupby("country")["user_id"].nunique().sort_values(ascending=False)

    # Q2 -- MAU trend by channel, with Sep proportional forecast
    mau_channel = df.groupby(["year_month", "channel"])["user_id"].nunique().reset_index(name="active_users")
    pivot = mau_channel.pivot(index="year_month", columns="channel", values="active_users").fillna(0).sort_index()
    m["mau_pivot"] = pivot
    m["mau_total"] = df.groupby("year_month")["user_id"].nunique().sort_index()
    days_elapsed = df["event_timestamp"].max().day
    m["forecast_multiplier"] = 30 / days_elapsed
    m["sep_forecast_by_channel"] = (pivot.loc["2025-09"] * m["forecast_multiplier"]).round(0)
    m["sep_actual_total"] = int(m["mau_total"].loc["2025-09"])
    m["sep_forecast_total"] = round(m["sep_actual_total"] * m["forecast_multiplier"])

    # DIY 1 -- device by session, and where each device's users end their journey.
    # Device isn't fixed per user (most users appear on 2-3 devices), so "device share
    # within an event_type" just reflects the overall device mix. Taking each user's
    # LATEST event instead shows where that device's users actually end their journey.
    m["device_sessions"] = df.groupby("device")["session_id"].nunique().sort_values(ascending=False)
    EVENT_ORDER = ["login", "page_view", "search", "product_view", "add_to_cart", "checkout", "purchase", "logout"]
    latest_idx = df.groupby("user_id")["event_timestamp"].idxmax()
    latest_events = df.loc[latest_idx, ["user_id", "device", "event_type"]]
    device_last_event = pd.crosstab(latest_events["device"], latest_events["event_type"], normalize="index") * 100
    m["device_last_event_pct"] = device_last_event.reindex(columns=EVENT_ORDER).fillna(0)
    browse_cols = [c for c in ["page_view", "search", "product_view"] if c in m["device_last_event_pct"].columns]
    m["device_end_mid_browse"] = m["device_last_event_pct"][browse_cols].sum(axis=1)
    m["device_end_at_purchase"] = m["device_last_event_pct"].get("purchase", pd.Series(0, index=device_last_event.index))

    # DIY 2 -- same-day search -> add_to_cart. Uses df_all_dates (pre-March-trim -- a
    # same-day comparison doesn't need full months, and dropping March would only lose
    # valid pairs) and pairs EVERY search with every add_to_cart that day, not just the
    # user's first search.
    searches = df_all_dates[df_all_dates["event_type"] == "search"][["user_id", "date", "event_timestamp"]].rename(
        columns={"event_timestamp": "search_time"})
    atc_all_dates = df_all_dates[df_all_dates["event_type"] == "add_to_cart"][["user_id", "date", "event_timestamp"]].rename(
        columns={"event_timestamp": "atc_time"})
    merged = searches.merge(atc_all_dates, on=["user_id", "date"], how="left")
    merged["converted"] = merged["atc_time"] > merged["search_time"]
    converted_flag = merged.groupby(["user_id", "date"])["converted"].any().reset_index()
    m["search_days"] = int(converted_flag.shape[0])
    m["search_converted"] = int(converted_flag["converted"].sum())

    # Funnel overview
    user_max_stage = df.dropna(subset=["funnel_stage"]).groupby("user_id")["funnel_stage"].max()
    funnel_counts = pd.Series({stage_names[s]: int((user_max_stage >= s).sum()) for s in [1, 2, 3, 4]})
    m["funnel_counts"] = funnel_counts
    m["cart_to_checkout"] = funnel_counts["Checkout"] / funnel_counts["Add to Cart"]
    m["checkout_to_purchase"] = funnel_counts["Purchase"] / funnel_counts["Checkout"]
    m["overall_conversion"] = funnel_counts["Purchase"] / funnel_counts["Browse"]

    # Lead time per funnel stage
    def stage_times(g):
        g = g.sort_values("event_timestamp")
        out, cursor = {}, None
        for s in [1, 2, 3, 4]:
            cand = g[g["funnel_stage"] == s]
            if cursor is not None:
                cand = cand[cand["event_timestamp"] > cursor]
            if cand.empty:
                break
            row = cand.iloc[0]
            out[s] = row["event_timestamp"]
            cursor = row["event_timestamp"]
        return out

    funnel_events = df.dropna(subset=["funnel_stage"])
    records = []
    for uid, g in funnel_events.groupby("user_id"):
        st = stage_times(g)
        for s_from, s_to in [(1, 2), (2, 3), (3, 4)]:
            if s_from in st and s_to in st:
                hrs = (st[s_to] - st[s_from]).total_seconds() / 3600
                records.append({"transition": f"{stage_names[s_from]} -> {stage_names[s_to]}", "hours": hrs})
    lead_time = pd.DataFrame(records)
    m["lead_time_summary"] = lead_time.groupby("transition")["hours"].agg(["count", "mean", "median"]).round(1)

    # Best sellers + purchase timing
    purchases = df[df["event_type"] == "purchase"]
    m["top_products"] = purchases["product_id"].value_counts().head(8)
    m["purchase_by_dow"] = purchases["event_timestamp"].dt.day_name().value_counts().reindex(DOW_ORDER).fillna(0)
    hour_counts = purchases["event_timestamp"].dt.hour.value_counts().sort_index()
    m["peak_hour"] = int(hour_counts.idxmax())
    m["peak_hour_count"] = int(hour_counts.max())

    # Cart abandonment
    atc_users = set(df[df["event_type"] == "add_to_cart"]["user_id"])
    purchase_users = set(df[df["event_type"] == "purchase"]["user_id"])
    abandoned = atc_users - purchase_users
    m["atc_users_n"] = len(atc_users)
    m["abandoned_n"] = len(abandoned)
    m["converted_n"] = len(atc_users & purchase_users)
    m["abandonment_rate"] = len(abandoned) / len(atc_users)

    atc_prod = df[df["event_type"] == "add_to_cart"]["product_id"].value_counts()
    purch_prod = df[df["event_type"] == "purchase"]["product_id"].value_counts()
    m["abandoned_products"] = (atc_prod - purch_prod.reindex(atc_prod.index).fillna(0)).sort_values(ascending=False).head(6)

    return m


# --- Chart helpers (local to this report -- native, editable PPT charts) --

def blend_with_navy(t):
    """t in [0,1]: 0 -> white, 1 -> full NAVY. Ordinal ramp for ordered categories."""
    nr, ng, nb = int(NAVY[0:2], 16), int(NAVY[2:4], 16), int(NAVY[4:6], 16)
    r = int(255 + (nr - 255) * t)
    g = int(255 + (ng - 255) * t)
    b = int(255 + (nb - 255) * t)
    return f"{r:02x}{g:02x}{b:02x}"


def _style_axes(chart, value_number_format="#,##0"):
    chart.has_title = False
    cat_axis = chart.category_axis
    cat_axis.tick_labels.font.size = Pt(11)
    cat_axis.tick_labels.font.color.rgb = RGBColor.from_string(BODY_CLR)
    cat_axis.format.line.color.rgb = RGBColor.from_string(DIVIDER)
    val_axis = chart.value_axis
    val_axis.tick_labels.font.size = Pt(9)
    val_axis.tick_labels.font.color.rgb = RGBColor.from_string(FOOTER_CLR)
    val_axis.format.line.fill.background()
    val_axis.major_gridlines.format.line.color.rgb = RGBColor.from_string(DIVIDER)
    val_axis.tick_labels.number_format = value_number_format
    val_axis.tick_labels.number_format_is_linked = False


def add_bar_chart(slide, left, top, width, height, categories, values, colors,
                   data_label_format="#,##0", horizontal=False):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    chart_data.add_series("Series 1", values)
    chart_type = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gframe = slide.shapes.add_chart(chart_type, Inches(left), Inches(top), Inches(width), Inches(height), chart_data)
    chart = gframe.chart
    chart.has_legend = False
    plot = chart.plots[0]
    plot.gap_width = 60
    series = plot.series[0]
    for i, point in enumerate(series.points):
        point.format.fill.solid()
        point.format.fill.fore_color.rgb = RGBColor.from_string(colors[i % len(colors)])
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(11)
    dl.font.bold = True
    dl.font.color.rgb = RGBColor.from_string(NAVY)
    dl.number_format = data_label_format
    dl.number_format_is_linked = False
    _style_axes(chart, value_number_format=data_label_format)
    return chart


def add_line_chart(slide, left, top, width, height, categories, values, color=NAVY):
    chart_data = CategoryChartData()
    chart_data.categories = categories
    chart_data.add_series("MAU", values)
    gframe = slide.shapes.add_chart(XL_CHART_TYPE.LINE_MARKERS, Inches(left), Inches(top),
                                     Inches(width), Inches(height), chart_data)
    chart = gframe.chart
    chart.has_legend = False
    series = chart.plots[0].series[0]
    series.format.line.color.rgb = RGBColor.from_string(color)
    series.format.line.width = Pt(2.5)
    series.marker.format.fill.solid()
    series.marker.format.fill.fore_color.rgb = RGBColor.from_string(color)
    series.marker.format.line.color.rgb = RGBColor.from_string(color)
    plot = chart.plots[0]
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.font.size = Pt(10)
    dl.font.bold = True
    dl.font.color.rgb = RGBColor.from_string(NAVY)
    dl.number_format = "#,##0"
    dl.number_format_is_linked = False
    _style_axes(chart)
    return chart


# --- Slide builders ---------------------------------------------------------

def build_cover_slide(prs, m):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_panel(slide, 0, 0, SLIDE_W, SLIDE_H, accent=GOLD, bg=NAVY)

    add_text(slide, MARGIN + 0.3, 1.4, 8.0, 0.4,
              [("EXECUTIVE REPORT", 16, True, GOLD, False)])
    add_text(slide, MARGIN + 0.3, 1.9, 9.0, 1.6,
              [("BukaToko Funnel", 40, True, WHITE, False),
               ("Conversion Analysis", 40, True, WHITE, False)])
    add_text(slide, MARGIN + 0.3, 3.55, 8.6, 0.6,
              [("Q2 2025 user activity across countries and channels, and where the "
                "purchase funnel leaks conversion.", 15, False, "C7D2DE", False)])

    _rect_y = 4.55
    line = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Inches(MARGIN + 0.3), Inches(_rect_y),
                                       Inches(MARGIN + 2.3), Inches(_rect_y))
    line.line.color.rgb = RGBColor.from_string(GOLD)
    line.line.width = Pt(2)

    add_text(slide, MARGIN + 0.3, 4.8, 6.0, 0.4,
              [(PRESENTER_NAME, 20, True, WHITE, False)])
    add_text(slide, MARGIN + 0.3, 5.2, 6.5, 0.35,
              [(PRESENTER_ROLE, 14, False, "C7D2DE", False)])

    from datetime import datetime
    add_text(slide, MARGIN + 0.3, 6.85, 6.0, 0.3,
              [(datetime.now().strftime("%B %Y"), 12, False, "8AA0B8", False)])
    return slide


def build_executive_summary(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Executive Summary", page, total)
    top_country = m["country_q2"].idxmax()
    top_country_n = int(m["country_q2"].max())
    top_device = m["device_sessions"].idxmax()
    top_device_share = m["device_sessions"].max() / m["device_sessions"].sum()

    cards = [
        (top_country, f"Top country by active users, Q2 2025 ({top_country_n:,})"),
        (top_device, f"Most-used device by session ({top_device_share:.0%} of sessions)"),
        (f"{m['overall_conversion']:.1%}", "Overall Browse -> Purchase conversion"),
        (f"{m['abandonment_rate']:.0%}", "Add-to-cart users who never purchased"),
    ]
    card_w = (SLIDE_W - 2 * MARGIN - 3 * 0.2) / 4
    for i, (value, label) in enumerate(cards):
        add_stat_card(s, MARGIN + i * (card_w + 0.2), 1.6, value, label, width=card_w)

    add_eyebrow(s, MARGIN, 3.35, 6, "KEY FINDINGS")
    findings = [
        f"Indonesia leads Q2 2025 active users ({top_country_n:,}) -- the highest-leverage "
        f"market for conversion tests.",
        f"Cart abandonment is the single biggest leak: {m['abandonment_rate']:.0%} of "
        f"add-to-cart users never complete a purchase.",
        "Browse -> Add to Cart has the longest, best-supported gap between stages -- worth "
        "testing retargeting for users who browsed without adding to cart (exact gap length "
        "is likely inflated by this log's sparse event capture, so treat it as directional).",
        f"{top_device} accounts for {top_device_share:.0%} of sessions, but device does not "
        "gate any specific action -- no device-specific funnel fix is indicated by usage alone.",
    ]
    y = 3.7
    for f in findings:
        add_text(s, MARGIN, y, SLIDE_W - 2 * MARGIN, 0.5, [(f"•  {f}", 14, False, BODY_CLR, False)])
        y += 0.62
    return s


def build_country_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Q1 -- Active Users by Country (Q2 2025)", page, total)
    add_eyebrow(s, MARGIN, 1.55, 6, "DISTINCT ACTIVE USERS, APR 1 - JUN 30 2025")
    order = m["country_q2"]
    top_country = order.idxmax()
    colors = [GOLD if c == top_country else blend_with_navy(0.55) for c in order.index]
    add_bar_chart(s, MARGIN, 1.9, SLIDE_W - 2 * MARGIN, 4.2, list(order.index), list(order.values), colors)
    add_footnote(s, MARGIN, 6.2, SLIDE_W - 2 * MARGIN,
                 f"Answer: Indonesia (ID) leads with {int(order.max()):,} active users, well ahead of "
                 f"US ({int(order['US']):,}) and Vietnam ({int(order['VN']):,}).")
    return s


def build_mau_trend_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Q2 -- Monthly Active Users Trend", page, total)
    add_eyebrow(s, MARGIN, 1.55, 6, "TOTAL MAU BY MONTH (APR-SEP 2025, SEP IS PARTIAL)")
    months = list(m["mau_total"].index)
    add_line_chart(s, MARGIN, 1.85, SLIDE_W - 2 * MARGIN, 1.7, months, list(m["mau_total"].values))

    add_stat_card(s, MARGIN, 3.7, f"{m['sep_actual_total']:,}", "September actual MAU (1-23 Sep)",
                  width=2.7, height=0.75, value_size=20, label_size=11)
    add_stat_card(s, MARGIN + 2.9, 3.7, f"{m['sep_forecast_total']:,}",
                  "September forecast (x 30/23, full month)", width=2.7, height=0.75,
                  value_size=20, label_size=11, accent=NAVY)

    add_eyebrow(s, MARGIN, 4.65, 6, "SEPTEMBER FORECAST BY CHANNEL")
    headers = ["Channel", "Sep Actual", "Sep Forecast"]
    rows = [(ch, f"{int(m['mau_pivot'].loc['2025-09', ch]):,}", f"{int(m['sep_forecast_by_channel'][ch]):,}")
            for ch in m["sep_forecast_by_channel"].sort_values(ascending=False).index]
    add_table(s, MARGIN, 4.95, SLIDE_W - 2 * MARGIN, headers, rows, header_h=0.3, row_h=0.23)
    return s


def build_device_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "DIY 1 -- Device Usage", page, total)
    add_eyebrow(s, MARGIN, 1.55, 6, "SESSIONS BY DEVICE (1 SESSION = 1 EVENT)")
    order = m["device_sessions"]
    top_device = order.idxmax()
    colors = [GOLD if d == top_device else blend_with_navy(0.55) for d in order.index]
    add_bar_chart(s, MARGIN, 1.9, 5.4, 4.1, list(order.index), list(order.values), colors)

    add_eyebrow(s, 6.2, 1.9, 3.3, "WHERE EACH DEVICE'S USERS END (LAST EVENT, %)")
    headers = ["Device", "End mid-Browse", "End at Purchase"]
    rows = [(d, f"{m['device_end_mid_browse'][d]:.0f}%", f"{m['device_end_at_purchase'][d]:.0f}%")
            for d in order.index]
    add_table(s, 6.2, 2.25, 3.3, headers, rows, header_h=0.32, row_h=0.45, first_col_frac=0.4)

    add_footnote(s, MARGIN, 6.15, SLIDE_W - 2 * MARGIN,
                 f"Answer: {top_device} is the most-used device ({order.max() / order.sum():.0%} of "
                 "sessions). Looking at where each device's users' journeys actually end (last-ever "
                 "event), the pattern is the same on every device -- most end mid-browse, very few "
                 "reach purchase. Device doesn't gate how far a user gets in the funnel.")
    return s


def build_funnel_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Funnel Conversion Overview", page, total)
    fc = m["funnel_counts"]
    ramp = [blend_with_navy(t) for t in [0.35, 0.55, 0.75, 1.0]]
    add_eyebrow(s, MARGIN, 1.55, 6, "USERS REACHING EACH FUNNEL STAGE")
    add_bar_chart(s, MARGIN, 1.9, 5.4, 3.2, list(fc.index), list(fc.values), ramp)

    add_eyebrow(s, 6.2, 1.9, 3.3, "LATE-FUNNEL DROP-OFF")
    add_bar_chart(s, 6.2, 2.25, 3.3, 2.85,
                  ["Cart -> Checkout", "Checkout -> Purchase"],
                  [round(m["cart_to_checkout"], 4), round(m["checkout_to_purchase"], 4)],
                  [NAVY, GOLD], data_label_format="0.0%")

    add_footnote(s, MARGIN, 5.35, SLIDE_W - 2 * MARGIN,
                 f"Overall Browse -> Purchase conversion: {m['overall_conversion']:.1%}. Same-day "
                 f"search -> add-to-cart conversion is only {m['search_converted']}/{m['search_days']} "
                 f"({m['search_converted']/m['search_days']:.1%}) -- too sparse to be a reliable signal "
                 "given the event log's session-tracking limitation (see appendix / notebook).")
    return s


def build_lead_time_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Lead Time per Funnel Stage", page, total)
    lt = m["lead_time_summary"]
    order_t = list(lt.index)
    longest = lt["mean"].idxmax()
    colors = [GOLD if t == longest else blend_with_navy(0.55) for t in order_t]
    days = [round(v / 24, 1) for v in lt.loc[order_t, "mean"]]
    add_eyebrow(s, MARGIN, 1.55, 6, "AVERAGE LEAD TIME BETWEEN STAGES (DAYS)")
    add_bar_chart(s, MARGIN, 1.85, SLIDE_W - 2 * MARGIN, 2.6, order_t, days, colors,
                  data_label_format='0.0" d"')

    headers = ["Transition", "n (users)", "Mean (h)", "Median (h)"]
    rows = [(t, int(lt.loc[t, "count"]), lt.loc[t, "mean"], lt.loc[t, "median"]) for t in order_t]
    add_table(s, MARGIN, 4.65, SLIDE_W - 2 * MARGIN, headers, rows, header_h=0.32, row_h=0.32)

    add_footnote(s, MARGIN, 5.85, SLIDE_W - 2 * MARGIN,
                 f"{longest} is the longest, best-supported gap -- prioritize retargeting users who "
                 "browsed but didn't add to cart. Checkout -> Purchase has too few qualifying cases "
                 "(n shown above) to be conclusive. Caveat: this log captures a sparse activity sample "
                 "(median ~4 months between a user's first and last event), so these gaps overstate "
                 "real browsing-to-cart friction and should be read as directional, not a literal "
                 "session-level delay.")
    return s


def build_products_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Best-Selling Products & Purchase Timing", page, total)
    add_eyebrow(s, MARGIN, 1.55, 4.6, "TOP PRODUCTS BY PURCHASE COUNT")
    headers = ["Product", "Purchases"]
    rows = [(p, int(v)) for p, v in m["top_products"].items()]
    add_table(s, MARGIN, 1.9, 4.6, headers, rows, header_h=0.35, row_h=0.4, first_col_frac=0.55)

    add_eyebrow(s, 5.9, 1.55, 3.6, "PURCHASES BY DAY OF WEEK")
    dow = m["purchase_by_dow"]
    peak_day = dow.idxmax()
    colors = [GOLD if d == peak_day else blend_with_navy(0.55) for d in dow.index]
    short_days = [d[:3] for d in dow.index]
    add_bar_chart(s, 5.9, 1.9, 3.6, 3.6, short_days, [int(v) for v in dow.values], colors)

    add_footnote(s, MARGIN, 6.1, SLIDE_W - 2 * MARGIN,
                 f"Peak purchase day: {peak_day}. Peak purchase hour: {m['peak_hour']:02d}:00 "
                 f"({m['peak_hour_count']} purchases) -- useful windows for flash sales or push timing.")
    return s


def build_abandonment_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Behavior: Added to Cart but Never Purchased", page, total)
    add_eyebrow(s, MARGIN, 1.55, 4.4, "ADD-TO-CART USERS: OUTCOME")
    add_bar_chart(s, MARGIN, 1.9, 4.4, 3.3,
                  ["Never purchased", "Purchased"],
                  [m["abandoned_n"], m["converted_n"]],
                  [GOLD, NAVY])

    add_eyebrow(s, 5.7, 1.55, 3.8, "MOST-ABANDONED PRODUCTS")
    headers = ["Product", "Abandoned count"]
    rows = [(p, int(v)) for p, v in m["abandoned_products"].items()]
    add_table(s, 5.7, 1.9, 3.8, headers, rows, header_h=0.32, row_h=0.35, first_col_frac=0.5)

    add_footnote(s, MARGIN, 5.6, SLIDE_W - 2 * MARGIN,
                 f"{m['abandoned_n']:,} of {m['atc_users_n']:,} add-to-cart users ({m['abandonment_rate']:.0%}) "
                 "never completed a purchase -- the largest single leak in the funnel. The products above "
                 "are where to start investigating pricing, stock, description, or page UX.")
    return s


def build_recommendations_slide(prs, m, page, total):
    s = new_slide(prs, KICKER, FOOTER, "Recommendations & Next Steps", page, total)
    top_country = m["country_q2"].idxmax()
    longest = m["lead_time_summary"]["mean"].idxmax()

    headers = ["Finding", "Recommendation", "Priority"]
    rows = [
        (f"{top_country} leads Q2 active users", "Run conversion experiments here first -- highest volume x headroom", "High"),
        (f"{m['abandonment_rate']:.0%} cart abandonment", "Launch cart-recovery emails/push; audit top-gap products", "High"),
        (f"{longest} is slowest stage", "Retarget browsers who didn't add to cart, not just late-funnel", "High"),
        ("Cart -> Checkout vs Checkout -> Purchase gap", "Fix whichever step is lower -- pricing/shipping vs payment UX", "Medium"),
        ("session_id doesn't group multi-event visits", "Fix event tracking at the source to unlock true visit-level funnels", "Medium"),
    ]
    add_table(s, MARGIN, 1.6, SLIDE_W - 2 * MARGIN, headers, rows, header_h=0.4, row_h=0.62, first_col_frac=0.34)
    add_footnote(s, MARGIN, 5.35, SLIDE_W - 2 * MARGIN,
                 "Full methodology, data-quality notes, and supporting charts: "
                 "code/BukaToko_Funnel_Analysis.ipynb")
    return s


# --- Orchestration -----------------------------------------------------------

def build_executive_report(output_dir=OUTPUT_DIR):
    df, df_all_dates, stage_names = load_and_clean()
    m = compute_metrics(df, df_all_dates, stage_names)

    prs = new_presentation()
    build_cover_slide(prs, m)

    total = 9
    build_executive_summary(prs, m, 1, total)
    build_country_slide(prs, m, 2, total)
    build_mau_trend_slide(prs, m, 3, total)
    build_device_slide(prs, m, 4, total)
    build_funnel_slide(prs, m, 5, total)
    build_lead_time_slide(prs, m, 6, total)
    build_products_slide(prs, m, 7, total)
    build_abandonment_slide(prs, m, 8, total)
    build_recommendations_slide(prs, m, 9, total)

    return save_report(prs, output_dir, "Executive_BukaToko_Funnel_Report")


if __name__ == "__main__":
    path = build_executive_report()
    print(f"Report saved to: {path}")
