# Congklak — Minimax & Alpha-Beta Search

**Why a board game:** used here as a public, non-confidential stand-in for structured decision-modeling and strategy-comparison work — see the repo root README for how each Game Analysis project maps to a real skill.
**CV skill represented:** algorithmic decision modeling / adversarial search — general problem-solving depth beyond the explicit CV skill list.

## Problem
Compare decision-making strategies for Congklak (traditional Indonesian mancala, 7 holes/7 seeds per side): does deeper search actually win, and does the opening move matter?

## Method
Four strategies of increasing sophistication — random, greedy, one-ply lookahead, and true minimax with alpha-beta pruning (correctly handling Congklak's "extra turn" rule, so those branches don't flip the maximizing player). Evidence comes from three experiments, not one anecdotal game:
1. Round-robin tournament (every strategy vs. every strategy, both seats) → win rates.
2. Opening-move study → is any starting hole a decisive advantage?
3. Score-progression analysis across many games → how reliably a lead holds, not just its average size.

## Result
Minimax-6 (6-ply search) dominates every shallower strategy; a shallow 3-ply search can still lose to the 1-ply lookahead in some matchups, showing search depth isn't free. Openings are close to symmetric.

## How to run
Open `code/congklak_analysis.ipynb` and run top to bottom.
