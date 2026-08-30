# Snake and Ladder — Markov Chain Analysis

**Why a board game:** a public, non-confidential stand-in for stochastic-process modeling — see the repo root README for how each Game Analysis project maps to a real skill.
**CV skill represented:** Markov chain / absorbing-state modeling — general problem-solving depth beyond the explicit CV skill list.

## Problem
How many turns does Snake and Ladder actually take to finish, how much does each snake/ladder reshape the odds, and how does the number of dice used per turn change the game?

## Method
The board is modeled as an absorbing Markov chain (one absorbing state: the finish square). The transition matrix is built **directly in Python** (`SnL_markov_chain.py`) from the board layout and the dice-sum distribution — not read from a fixed spreadsheet — so the number of dice is just a parameter (`build_transition_matrix(n_dice=...)`), and the whole analysis re-runs for any dice count without needing a matching data file.

Board size, the ladder/snake layout, and the dice count are all set in the notebook's own **Parameters** cell (near the top), not buried as constants inside a module — change the board or the dice count there and re-run.

Rebuilding the matrix in code also surfaced a real bug in the project's old hand-built Excel data: the 2-dice sheet's finish square wasn't actually self-absorbing (it had outgoing "bounce-back" transitions as if the game were still in play), which would have silently corrupted any multi-step probability analysis run on it. The new matrix builder is correct by construction and was validated against the old 1-dice matrix's exact values before that file was retired.

## Deliverables
- Expected number of turns to finish, from every square (fundamental matrix).
- Fastest probabilistic route from square 1.
- Heatmap of expected steps to finish, per board square.
- Snake/ladder impact ranking — how many turns each element actually saves or costs, accounting for interactions with the rest of the board.
- Position distribution over time and the finish-time distribution (mean/std computed exactly from the fundamental matrix, not estimated from a truncated simulation).
- **Dice-count comparison** (1–4 dice) — the direct payoff of building the matrix in code instead of reading it from a spreadsheet.

Reference board layouts used while designing the board are in `docs/`.

## Output layout
```
result/
  figures/   -- all chart PNGs (expected steps, heatmap, impact, position/finish distributions, dice comparison)
  slides/    -- Executive_SnakeLadder_Report_<date>.pptx
```
Same `figures/` + `slides/` convention as this portfolio's other executive reports, built by `code/SnL_report_builder.py`.

## Code structure
- `code/snake_and_ladder_analysis.ipynb` — Parameters cell up top, then the analysis, STEP by STEP.
- `code/SnL_markov_chain.py` — dynamic transition-matrix builder, expected steps, fastest route, snake/ladder impact, position/finish-time distributions. `SnL_` prefix keeps it identifiable as project-specific, not a generic module name.
- `code/SnL_visualizations.py` — all chart-building, with a larger shared font baseline (the board heatmap especially was hard to read at the old default sizes).
- `code/SnL_report_builder.py` — 5-slide executive PPTX (summary, board heatmap, snake/ladder impact, finish-time distribution, dice-count comparison).

## How to run
Open `code/snake_and_ladder_analysis.ipynb` and run top to bottom — the board and transition matrix are both built in code, no external data file needed.
