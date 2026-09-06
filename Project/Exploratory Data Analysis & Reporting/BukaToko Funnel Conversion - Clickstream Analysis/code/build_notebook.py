"""
Script that assembles the case-study notebook using nbformat.
Run once to (re)generate `BukaToko_Funnel_Analysis.ipynb`, then it is executed
with nbclient to make sure every cell runs cleanly.
"""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ---------------------------------------------------------------------------
md("""\
# BukaToko — Funnel Conversion Analysis

E-commerce clickstream case study. Goal: understand user activity across countries/channels
and improve funnel conversion (browse → add to cart → checkout → purchase).
""")

# ---------------------------------------------------------------------------
md("## STEP 00 — Setup")

code("""\
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from IPython.display import HTML

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 120)
sns.set_theme(style='whitegrid')
plt.rcParams['figure.dpi'] = 110

# Fixed categorical palette (colorblind-safe, fixed order -- never cycled/reassigned).
# Slot 4 (yellow) is tuned to RevoU's brand gold; blue/aqua already echo the logo's
# mint-to-periwinkle gradient, so they carry the data marks while yellow is reserved
# as the brand accent -- used only to highlight "the answer" bar in a chart, the same
# way the yellow circle pops against the gradient in the RevoU mark.
CAT = ['#2a78d6', '#eb6834', '#1baf7a', '#F5C518', '#e87ba4', '#008300', '#4a3aa7', '#e34948']
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = CAT
BRAND_YELLOW = YELLOW
SEQ_BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']

def blend_with_white(hex_color, t):
    \"\"\"t in [0,1]: 0 -> white, 1 -> full hex_color. Builds a one-hue sequential ramp on demand.\"\"\"
    r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
    r = int(255 + (r - 255) * t)
    g = int(255 + (g - 255) * t)
    b = int(255 + (b - 255) * t)
    return f'#{r:02x}{g:02x}{b:02x}'

DATA_PATH = '../data/Dataset Case Study - dirty_dummy_events.csv'
raw = pd.read_csv(DATA_PATH)
print('Raw shape:', raw.shape)
raw.head()
""")

# ---------------------------------------------------------------------------
md("## STEP 01 — Data Quality Audit")

code("""\
print(raw.dtypes)
print()
print('--- Missing values ---')
print(raw.isnull().sum())
print('--- Duplicate rows / event_id ---', raw.duplicated().sum(), '/', raw['event_id'].duplicated().sum())
""")

code("""\
events_per_session = raw.groupby('session_id').size()
print('Unique sessions:', raw['session_id'].nunique(), '/ total rows:', len(raw))
print('Avg events per session:', events_per_session.mean())
""")

md("""\
- `channel` missing ~1% (100 rows); `device` casing inconsistent (`IOS`/`android`/`desktop`); no duplicate rows.
- **Key structural issue:** every `session_id` = exactly one event -- it can't link
  multi-event journeys. Workaround used below: (`user_id`, calendar date) as a visit proxy
  wherever same-visit logic is needed (DIY 2).
""")

# ---------------------------------------------------------------------------
md("## STEP 02 — Data Cleaning")

code("""\
df = raw.copy()

df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

for c in ['event_type', 'country', 'device', 'channel']:
    df[c] = df[c].astype('string').str.strip()

device_map = {'ios': 'iOS', 'android': 'Android', 'desktop': 'Desktop'}
df['device'] = df['device'].str.lower().map(device_map)

df['channel'] = df['channel'].fillna('unknown')  # ~1% unattributed -- kept, not guessed

df['date'] = df['event_timestamp'].dt.date
df['year_month'] = df['event_timestamp'].dt.to_period('M').astype(str)

# Funnel stage: page_view/search/product_view are all pre-commitment discovery (stage 1);
# login/logout are session housekeeping, not funnel steps (left NaN, dropped later).
funnel_order = {
    'page_view': 1, 'search': 1, 'product_view': 1,
    'add_to_cart': 2, 'checkout': 3, 'purchase': 4,
}
stage_names = {1: 'Browse', 2: 'Add to Cart', 3: 'Checkout', 4: 'Purchase'}
df['funnel_stage'] = df['event_type'].map(funnel_order)

# Fixed color per category (alphabetical order -> CAT slots), reused in every chart
# below so e.g. "email" is always the same color wherever channel appears.
CHANNEL_COLORS = {c: CAT[i] for i, c in enumerate(sorted(df['channel'].unique()))}
DEVICE_COLORS = {d: CAT[i] for i, d in enumerate(sorted(df['device'].unique()))}
DOW_ORDER = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
DOW_COLORS = {d: CAT[i] for i, d in enumerate(DOW_ORDER)}
STAGE_RAMP = [SEQ_BLUE[1], SEQ_BLUE[3], SEQ_BLUE[4], SEQ_BLUE[6]]  # ordinal ramp, Browse -> Purchase

print('Clean shape:', df.shape)
df.head()
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 03 — Scope: Drop Partial March
Data starts 2025-03-27, so March is a partial month that would distort month-level
comparisons. Trim to full months, starting **2025-04-01**. September is also partial
(data ends 2025-09-23) -- kept, but see the forecast in STEP 05.
""")

code("""\
before_rows = len(df)
df = df[df['event_timestamp'] >= '2025-04-01'].reset_index(drop=True)
print(f'Rows: {before_rows} -> {len(df)} (dropped {before_rows - len(df)} March rows)')
print('New date range:', df['event_timestamp'].min(), '->', df['event_timestamp'].max())
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 04 — Q1: Which country has the most Monthly Active Users in Q2 2025 (Apr–Jun)?
Active user = distinct `user_id` with >=1 event. Q2 2025 = **[2025-04-01, 2025-07-01)**.
""")

code("""\
q2_2025 = df[(df['event_timestamp'] >= '2025-04-01') & (df['event_timestamp'] < '2025-07-01')]
active_users_by_country = (
    q2_2025.groupby('country')['user_id'].nunique().sort_values(ascending=False).rename('active_users')
)
active_users_by_country
""")

code("""\
def flag(code):
    return ''.join(chr(ord(c) + 127397) for c in code.upper())

order = active_users_by_country
top_country = order.idxmax()
labels = [f'{flag(c)}<br>{c}' for c in order.index]
colors = [BRAND_YELLOW if c == top_country else SEQ_BLUE[2] for c in order.index]

fig = go.Figure(go.Bar(x=labels, y=order.values, marker_color=colors,
                        text=order.values, textposition='outside'))
fig.update_layout(title='Q2 2025 Active Users by Country', yaxis_title='Active users',
                   showlegend=False, height=420, width=650,
                   plot_bgcolor='#fcfcfb', paper_bgcolor='#fcfcfb')
HTML(fig.to_html(full_html=False, include_plotlyjs='cdn'))
""")

md("**Answer:** Indonesia (**ID**) -- most active users in Q2 2025, well ahead of US and Vietnam.")

# ---------------------------------------------------------------------------
md("""\
## STEP 05 — Q2: Monthly Active Users Trend by Channel
Small multiples (one panel per channel) instead of one crowded overlapping-line chart.
September is partial (23 of 30 days) -- shown as-is, plus a proportional forecast
(`actual * 30/23`) to estimate the full month.
""")

code("""\
mau_channel = df.groupby(['year_month', 'channel'])['user_id'].nunique().reset_index(name='active_users')
pivot = mau_channel.pivot(index='year_month', columns='channel', values='active_users').fillna(0).sort_index()

days_elapsed = df['event_timestamp'].max().day  # 23
forecast_multiplier = 30 / days_elapsed
sep_forecast = (pivot.loc['2025-09'] * forecast_multiplier).round(0)

summary = pivot.copy()
summary.loc['2025-09 (forecast)'] = sep_forecast
summary
""")

code("""\
x_actual = pivot.index.tolist()
ncols = 4
nrows = int(np.ceil(len(pivot.columns) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), sharey=True)
axes = axes.flatten()
for ax, ch in zip(axes, pivot.columns):
    c = CHANNEL_COLORS[ch]
    y_actual = pivot[ch].values
    ax.plot(x_actual, y_actual, marker='o', color=c, linewidth=2)
    ax.plot([x_actual[-1], 'Sep (fcst)'], [y_actual[-1], sep_forecast[ch]],
            color=c, linewidth=2, linestyle='--')
    ax.plot(['Sep (fcst)'], [sep_forecast[ch]], marker='o', markerfacecolor='white',
             markeredgecolor=c, markersize=7, zorder=5)
    ax.set_title(ch, fontsize=11, fontweight='bold', color=c)
    ax.tick_params(axis='x', rotation=45)
for ax in axes[len(pivot.columns):]:
    ax.axis('off')
fig.suptitle('Monthly Active Users by Channel (dashed = Sep forecast)', fontweight='bold', y=1.02)
sns.despine()
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 06 — DIY 1: Most-used device (by session), and device preference by event_type
Each `session_id` = one event, so sessions-by-device = rows-by-device.
""")

code("""\
device_sessions = df.groupby('device')['session_id'].nunique().sort_values(ascending=False)
top_device = device_sessions.idxmax()
colors = [BRAND_YELLOW if d == top_device else DEVICE_COLORS[d] for d in device_sessions.index]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(device_sessions.index, device_sessions.values, color=colors)
ax.bar_label(bars, padding=3)
ax.set_title('Sessions by Device', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()

share = device_sessions.max() / device_sessions.sum()
print(f'Most used device: {top_device} ({device_sessions.max()} sessions, {share:.1%} of all sessions)')
""")

code("""\
device_event = pd.crosstab(df['event_type'], df['device'], normalize='index') * 100
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(device_event.round(1), annot=True, fmt='.1f', cmap='Blues', ax=ax, cbar_kws={'label': '% of event_type'})
ax.set_title('Device Share (%) within Each Event Type', fontweight='bold')
plt.tight_layout()
plt.show()
""")

md("""\
**Answer:** Android is the most-used device overall. No event_type shows a distinct
device preference -- every event type (browse, cart, checkout, purchase...) splits
roughly ~49% Android / 40% iOS / 10% Desktop, matching the overall mix. Device doesn't
appear to gate any specific action in this data.
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 07 — DIY 2: Search → Add-to-Cart same-day conversion
Same `user_id` + calendar date, add_to_cart timestamp strictly after search timestamp.
""")

code("""\
searches = df[df['event_type'] == 'search'][['user_id', 'date', 'event_timestamp']].rename(columns={'event_timestamp': 'search_time'})
atc_all = df[df['event_type'] == 'add_to_cart'][['user_id', 'date', 'event_timestamp']].rename(columns={'event_timestamp': 'atc_time'})

first_search = searches.groupby(['user_id', 'date'], as_index=False)['search_time'].min()
merged = first_search.merge(atc_all, on=['user_id', 'date'], how='left')
merged['converted'] = merged['atc_time'] > merged['search_time']
converted_flag = merged.groupby(['user_id', 'date'])['converted'].any().reset_index()

total_search_days = converted_flag.shape[0]
converted_days = converted_flag['converted'].sum()
print(f'User-days with search: {total_search_days} | converted (same-day ATC after search): {converted_days} '
      f'({converted_days/total_search_days:.2%})')
""")

md("""\
**Answer:** ~0.1% (1 of ~940 search occasions). Confirmed this is a sample-size artifact,
not a behavioral signal -- almost every `user_id`+`date` pair in this log has only one
event total (session_id can't link multiple events, STEP 01), leaving nothing to match a
search against on the same day.
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 08 — Funnel Conversion Overview
Funnel: **Browse -> Add to Cart -> Checkout -> Purchase**. Per user, take the highest
stage ever reached.
""")

code("""\
user_max_stage = df.dropna(subset=['funnel_stage']).groupby('user_id')['funnel_stage'].max()
funnel_counts = pd.Series({stage_names[s]: (user_max_stage >= s).sum() for s in [1, 2, 3, 4]})

fig, ax = plt.subplots(figsize=(7, 4.5))
bars = ax.bar(funnel_counts.index, funnel_counts.values, color=STAGE_RAMP)
ax.bar_label(bars, padding=3)
ax.set_title('Overall Funnel: Users Reaching Each Stage', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()

stages = list(funnel_counts.index)
for i in range(1, len(stages)):
    print(f'{stages[i-1]} -> {stages[i]}: {funnel_counts[stages[i]] / funnel_counts[stages[i-1]]:.1%}')
print(f\"Overall Browse -> Purchase: {funnel_counts['Purchase'] / funnel_counts['Browse']:.2%}\")
""")

code("""\
cart_to_checkout = funnel_counts['Checkout'] / funnel_counts['Add to Cart']
checkout_to_purchase = funnel_counts['Purchase'] / funnel_counts['Checkout']

fig, ax = plt.subplots(figsize=(6, 4.5))
labels = ['Add to Cart -> Checkout', 'Checkout -> Purchase']
vals = [cart_to_checkout, checkout_to_purchase]
bars = ax.bar(labels, vals, color=[BLUE, AQUA])
ax.set_ylim(0, 1)
for bar, v in zip(bars, vals):
    ax.text(bar.get_x() + bar.get_width()/2, v, f'{v:.1%}', ha='center', va='bottom')
ax.set_title('Late-Funnel Drop-off: Cart vs Checkout Step', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

md("Whichever step is lower is the higher-leverage fix -- cart/pricing/shipping friction vs payment-flow friction.")

# ---------------------------------------------------------------------------
md("""\
## STEP 09 — Lead Time per Funnel Stage
Per user: first Browse event, first Add-to-Cart *after* it, first Checkout after that,
first Purchase after that -- gaps between them show which stage is slowest to move
through.
""")

code("""\
def stage_times(g):
    g = g.sort_values('event_timestamp')
    out, cursor = {}, None
    for s in [1, 2, 3, 4]:
        cand = g[g['funnel_stage'] == s]
        if cursor is not None:
            cand = cand[cand['event_timestamp'] > cursor]
        if cand.empty:
            break
        row = cand.iloc[0]
        out[s] = row['event_timestamp']
        cursor = row['event_timestamp']
    return out

funnel_events = df.dropna(subset=['funnel_stage'])
records = []
for uid, g in funnel_events.groupby('user_id'):
    st = stage_times(g)
    for s_from, s_to in [(1, 2), (2, 3), (3, 4)]:
        if s_from in st and s_to in st:
            hrs = (st[s_to] - st[s_from]).total_seconds() / 3600
            records.append({'transition': f'{stage_names[s_from]} -> {stage_names[s_to]}', 'hours': hrs})
lead_time = pd.DataFrame(records)
lt_summary = lead_time.groupby('transition')['hours'].agg(['count', 'mean', 'median']).round(1)
lt_summary
""")

code("""\
order_t = [f'{stage_names[a]} -> {stage_names[b]}' for a, b in [(1, 2), (2, 3), (3, 4)]]
longest = lt_summary['mean'].idxmax()
box_palette = {t: (BRAND_YELLOW if t == longest else SEQ_BLUE[4]) for t in order_t}

fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=lead_time, x='transition', y='hours', order=order_t, ax=ax, palette=box_palette, hue='transition', legend=False, showfliers=False)
ax.set_title('Lead Time Between Funnel Stages (hours, outliers hidden)', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()

print(f\"Longest average lead time: {longest} ({lt_summary.loc[longest, 'mean']:.0f}h, n={lt_summary.loc[longest, 'count']:.0f})\")
""")

md("""\
**Answer:** **Browse -> Add to Cart** is the longest and best-supported gap (~65 days avg,
n=539) -- this is the stage to prioritize (e.g. retargeting browsers who didn't add to
cart). **Caution:** `Checkout -> Purchase` has only ~3 qualifying cases -- too few to draw
any conclusion from; don't read its number as representative.
""")

# ---------------------------------------------------------------------------
md("## STEP 10 — Best-Selling Products")

code("""\
purchases = df[df['event_type'] == 'purchase']
top_products = purchases['product_id'].value_counts().head(15)

norm = (top_products.values - top_products.values.min()) / (top_products.values.max() - top_products.values.min() + 1e-9)
colors = [blend_with_white(BLUE, 0.35 + 0.65 * n) for n in norm][::-1]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_products.index[::-1], top_products.values[::-1], color=colors)
ax.set_title('Top 15 Products by Purchase Count', fontweight='bold')
ax.set_xlabel('Purchases')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("## STEP 11 — Purchase Pattern by Day of Week & Hour")

code("""\
purchases_dow = purchases['event_timestamp'].dt.day_name()
dow_counts = purchases_dow.value_counts().reindex(DOW_ORDER)
dow_bar_colors = [DOW_COLORS[d] for d in DOW_ORDER]

hour_counts = purchases['event_timestamp'].dt.hour.value_counts().sort_index()
hour_norm = (hour_counts.values - hour_counts.values.min()) / (hour_counts.values.max() - hour_counts.values.min() + 1e-9)
hour_bar_colors = [blend_with_white(BLUE, 0.3 + 0.65 * n) for n in hour_norm]

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
dow_counts.plot(kind='bar', ax=axes[0], color=dow_bar_colors)
axes[0].set_title('Purchases by Day of Week', fontweight='bold')
hour_counts.plot(kind='bar', ax=axes[1], color=hour_bar_colors)
axes[1].set_title('Purchases by Hour of Day', fontweight='bold')
sns.despine()
plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------------------
md("""\
## STEP 12 — Behavior: Added to Cart but Never Purchased
Definition: a user who has >=1 `add_to_cart` event but no `purchase` event anywhere in
the log (no order/cart id exists to link a specific cart item to a specific purchase, so
this is measured at the user level).
""")

code("""\
atc_users = set(df[df['event_type'] == 'add_to_cart']['user_id'])
purchase_users = set(df[df['event_type'] == 'purchase']['user_id'])
abandoned_users = atc_users - purchase_users
converted_users = atc_users & purchase_users
rate = len(abandoned_users) / len(atc_users)

fig, ax = plt.subplots(figsize=(5, 5))
ax.bar(['Never purchased', 'Purchased\\n(at some point)'], [len(abandoned_users), len(converted_users)], color=[RED, BLUE])
ax.set_title('Add-to-Cart Users: Abandoned vs Converted', fontweight='bold')
for i, v in enumerate([len(abandoned_users), len(converted_users)]):
    ax.text(i, v, str(v), ha='center', va='bottom')
sns.despine()
plt.tight_layout()
plt.show()

print(f'{len(abandoned_users)} / {len(atc_users)} add-to-cart users never purchased ({rate:.1%})')
""")

code("""\
atc_prod = df[df['event_type'] == 'add_to_cart']['product_id'].value_counts()
purch_prod = df[df['event_type'] == 'purchase']['product_id'].value_counts()
gap = (atc_prod - purch_prod.reindex(atc_prod.index).fillna(0)).sort_values(ascending=False).head(10)

gap_norm = (gap.values - gap.values.min()) / (gap.values.max() - gap.values.min() + 1e-9)
gap_colors = [blend_with_white(RED, 0.35 + 0.65 * n) for n in gap_norm][::-1]

fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(gap.index[::-1], gap.values[::-1], color=gap_colors)
ax.set_title('Top 10 Products: Add-to-Cart Count minus Purchase Count', fontweight='bold')
ax.set_xlabel('Cart adds not converted to a purchase of that product')
sns.despine()
plt.tight_layout()
plt.show()
""")

md("**Answer:** 75% of users who ever added something to cart never completed a purchase -- the largest single leak in the funnel. The product list above is where to start investigating (pricing, stock, description, or page UX).")

# ---------------------------------------------------------------------------
md("""\
## STEP 13 — Summary & Recommendations

1. **Indonesia** leads in Q2 2025 active users (STEP 04) -- highest-leverage market for
   conversion tests.
2. **Android** dominates sessions (STEP 06), but device doesn't gate specific actions --
   no device-specific funnel fix is indicated by usage alone.
3. **Cart abandonment is the single biggest leak: 75% of add-to-cart users never
   purchase** (STEP 12). Prioritize cart-recovery: abandonment emails/push, and review
   the top-gap products for pricing/stock/description issues.
4. **Browse -> Add to Cart is the slowest stage** (STEP 09, ~65 days avg, largest sample)
   -- retarget users who browsed but didn't add to cart, rather than assuming the
   bottleneck is late-funnel (checkout/payment).
5. **Cart -> Checkout vs Checkout -> Purchase** (STEP 08) pinpoints whether the
   remaining friction is pre-payment (pricing/shipping surprise) or payment-flow itself.
""")

nb['cells'] = cells

with open('BukaToko_Funnel_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print('Notebook written.')
