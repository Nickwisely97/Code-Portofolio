# Executive Report Template

Shared design system for every project's executive PowerPoint report — colors, layout constants, and the building blocks (slide header/footer, stat cards, eyebrows, panels, tables, footnotes). Not a project on its own; every project that builds an executive report imports from here instead of redefining the same navy/gold helpers locally.

## Why this exists
Four projects (Headcount Attrition, Temperature Forecast, Wine Recommendation, Snake and Ladder) each had their own `report_builder.py` with a nearly-identical copy of the same ~150 lines of PowerPoint boilerplate (`_rect`, `_text`, `new_slide`, `add_stat_card`, `add_eyebrow`, `add_panel`, a table helper). Pulling that into one place means:
- A visual tweak (font size, color, spacing) made here propagates to every project's next report build, instead of needing to be hunted down and repeated four times.
- Every project's own `report_builder.py` shrinks down to just what's actually specific to it: which stats, which charts, how many slides, in what order.

## What's here
- `report_template.py` — the shared module. See its docstring for the exact API (`new_presentation`, `new_slide`, `add_stat_card`, `add_eyebrow`, `add_panel`, `add_table`, `add_footnote`, `save_report`, plus the color/layout constants `NAVY`, `GOLD`, `BODY_CLR`, `SLIDE_W`, `MARGIN`, etc.).

## How a project uses it
A project's `code/report_builder.py` adds this folder to `sys.path` (relative to its own file location, so it works regardless of the notebook's working directory), imports the shared primitives, defines its own `KICKER`/`FOOTER` strings and slide content, and calls `save_report(...)` at the end instead of hand-rolling the `result/slides/<name>_<date>.pptx` path itself:

```python
import os, sys
_TEMPLATE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "Executive_Report_Template")
sys.path.insert(0, _TEMPLATE_DIR)
from report_template import (
    NAVY, GOLD, BODY_CLR, LABEL_CLR, SLIDE_W, MARGIN,
    new_presentation, new_slide, add_stat_card, add_eyebrow, add_panel,
    add_table, save_report,
)

KICKER = "EXECUTIVE REPORT   |   <PROJECT NAME>"
FOOTER = "<Method>  |  <one-line tagline>"

def build_executive_report(..., output_dir):
    prs = new_presentation()
    s1 = new_slide(prs, KICKER, FOOTER, "Executive Summary", 1, total_pages)
    # ...project-specific slide content...
    return save_report(prs, output_dir, "Executive_<Project>_Report")
```

The `../../../Executive_Report_Template` path assumes the standard layout `Project/<Category>/<Project Name>/code/report_builder.py` — three levels up from `code/` reaches `Project/`, where this folder lives as a sibling of every category folder (`Forecasting & Predictive Modeling/`, `Segmentation & Recommendation/`, `Game Analysis/`, etc.).

## Requirements
`python-pptx` (already listed in every project's own `requirements.txt` that uses this).
