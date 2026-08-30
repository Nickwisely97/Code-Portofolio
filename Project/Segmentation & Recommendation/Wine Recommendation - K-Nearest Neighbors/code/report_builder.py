"""
report_builder.py
Executive PowerPoint report builder for the Wine Recommendation project --
4 slides: executive summary, recommendation quality (precision@k), a sample
recommendation walkthrough with a chemical-profile radar chart, and the
full similarity landscape (PCA).

Design system (navy/gold, header/footer band, stat cards, tables) comes from
Executive_Report_Template/report_template.py, shared across this portfolio's
executive reports -- this file only adds the content specific to this project.
"""

import os
import sys

_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Executive_Report_Template")
sys.path.insert(0, _TEMPLATE_DIR)
from report_template import (
    GOLD, BODY_CLR, SLIDE_W, MARGIN,
    new_presentation, new_slide, add_text, add_stat_card, add_eyebrow, add_panel, add_table, save_report,
)
from pptx.util import Inches

KICKER = "EXECUTIVE REPORT   |   WINE RECOMMENDATION"
FOOTER = "K-Nearest Neighbors  |  Cosine Similarity on Chemical Profile"


def build_executive_report(
    n_wines, n_features, n_cultivars, avg_precision_at5,
    precision_curve_img, radar_img, pca_img,
    reference_label, recommended_rows,
    output_dir,
):
    """
    n_wines, n_features, n_cultivars : dataset headline counts.
    avg_precision_at5                : mean precision@5 across the whole catalog.
    precision_curve_img              : path to the precision@k chart.
    radar_img                        : path to the reference-vs-recommended radar chart.
    pca_img                          : path to the PCA similarity-landscape scatter.
    reference_label                  : the sample wine used for the walkthrough (e.g. "Wine #12 (Cultivar 0)").
    recommended_rows                 : list of (label, cultivar, similarity_str) tuples for the table.
    output_dir                       : the project's "result" directory -- the .pptx is saved to
                                        output_dir/slides, figures already live in output_dir/figures.

    Returns the saved report path.
    """
    total_pages = 4
    prs = new_presentation()

    # --- Slide 1: Executive Summary ---
    s1 = new_slide(prs, KICKER, FOOTER, "Executive Summary", 1, total_pages)
    cards = [
        (f"{n_wines}", "Wines in catalog"),
        (f"{n_features}", "Chemical attributes"),
        (f"{n_cultivars}", "Cultivars"),
        (f"{avg_precision_at5:.0%}", "Avg precision@5"),
    ]
    card_w, gap = 2.12, 0.17
    for i, (value, label) in enumerate(cards):
        left = MARGIN + i * (card_w + gap)
        add_stat_card(s1, left, 1.75, value, label)

    add_eyebrow(s1, MARGIN, 3.6, 9.0, "METHOD")
    add_panel(s1, MARGIN, 3.94, SLIDE_W - 2 * MARGIN, 2.5, accent=GOLD)
    add_text(s1, MARGIN + 0.25, 4.16, SLIDE_W - 2 * MARGIN - 0.5, 2.1, [
        ("Every wine's 13 lab-measured chemical attributes (alcohol, phenols, color intensity, proline, etc.) are standardized, then compared pairwise with cosine similarity.", 14, False, BODY_CLR, False),
        ("Given a wine a taster likes, a K-Nearest Neighbors lookup finds the closest matches in that standardized space -- no other tasters or ratings required, unlike collaborative filtering.", 14, False, BODY_CLR, False),
        ("Precision@5 validates the approach: it checks whether a wine's nearest neighbors actually share its cultivar (its real, known style), using the dataset's ground-truth labels.", 14, False, BODY_CLR, False),
    ])

    # --- Slide 2: Recommendation Quality ---
    s2 = new_slide(prs, KICKER, FOOTER, "Recommendation Quality", 2, total_pages)
    add_eyebrow(s2, MARGIN, 1.6, 9.0, "PRECISION@K -- DOES 'CHEMICALLY SIMILAR' MEAN 'SAME STYLE'?")
    if precision_curve_img and os.path.exists(precision_curve_img):
        s2.shapes.add_picture(precision_curve_img, Inches(MARGIN), Inches(2.0), width=Inches(9.0))

    # --- Slide 3: Example Recommendation ---
    s3 = new_slide(prs, KICKER, FOOTER, "Sample Recommendation", 3, total_pages)
    add_eyebrow(s3, MARGIN, 1.55, 9.0, f"REFERENCE: {reference_label} -- CHEMICAL PROFILE MATCH")
    if radar_img and os.path.exists(radar_img):
        s3.shapes.add_picture(radar_img, Inches(MARGIN), Inches(1.9), width=Inches(5.3))

    table_left = MARGIN + 5.3 + 0.25
    table_w = SLIDE_W - MARGIN - table_left
    add_eyebrow(s3, table_left, 1.55, table_w, "TOP RECOMMENDATIONS")
    add_table(s3, table_left, 1.9, table_w,
              headers=["Wine", "Cultivar", "Similarity"],
              rows=recommended_rows)

    # --- Slide 4: Similarity Landscape ---
    s4 = new_slide(prs, KICKER, FOOTER, "Similarity Landscape", 4, total_pages)
    add_eyebrow(s4, MARGIN, 1.55, 9.0, "ALL WINES IN PCA SPACE, COLORED BY CULTIVAR")
    if pca_img and os.path.exists(pca_img):
        s4.shapes.add_picture(pca_img, Inches(MARGIN + 0.75), Inches(1.9), width=Inches(7.5))

    return save_report(prs, output_dir, "Executive_Wine_Recommendation_Report")
