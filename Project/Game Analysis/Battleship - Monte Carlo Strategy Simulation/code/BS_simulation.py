"""
BS_simulation.py
Monte Carlo runner: plays many games per strategy and turns the results into
tables -- summary statistics, pairwise significance tests, and the
attacker-vs-placement-style matrix for the defense analysis.

Fairness: game g always uses the board seeded by (seed, g), so every strategy
is scored against exactly the same set of fleet placements. That turns the
comparison into a paired one and removes board-luck noise.
"""

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from BS_game import place_fleet


def play_game(board, strategy, rng, record=False):
    """
    Fire until every ship is sunk. Returns (n_shots, log); with record=True
    the log holds, per shot, the attacker's view *before* firing -- enough to
    redraw the probability map at any point of the game.
    """
    log = []
    while not board.all_sunk:
        remaining = board.remaining_lengths()
        shots_before = board.shots.copy() if record else None
        cell = strategy(board.shots, remaining, board.size, rng)
        result = board.fire(cell)
        if record:
            log.append({"shot": board.n_shots, "cell": cell, "result": result,
                        "shots_before": shots_before, "remaining_before": remaining})
    return board.n_shots, log


def run_simulation(strategies, n_games, size, fleet, seed, placement="random"):
    """Play n_games per strategy; one row per (strategy, game)."""
    rows = []
    for name, strategy in strategies.items():
        for g in range(n_games):
            board = place_fleet(size, fleet, np.random.default_rng([seed, g]), style=placement)
            n_shots, _ = play_game(board, strategy, np.random.default_rng([seed, g, 1]))
            rows.append({"strategy": name, "placement": placement, "game": g, "shots": n_shots})
    return pd.DataFrame(rows)


def summarize(results_df, by="strategy"):
    """Mean / median / spread of shots-to-win, plus the share of games won
    within 50 shots (half the 10x10 board)."""
    grouped = results_df.groupby(by, sort=False)["shots"]
    return pd.DataFrame({
        "mean": grouped.mean(),
        "median": grouped.median(),
        "std": grouped.std(),
        "p10": grouped.quantile(0.10),
        "p90": grouped.quantile(0.90),
        "min": grouped.min(),
        "max": grouped.max(),
        "win_within_50": grouped.apply(lambda s: (s <= 50).mean()),
    }).round(2)


def compare_consecutive(results_df):
    """
    Wilcoxon signed-rank test between each strategy and the next one up --
    paired by game, since both played the same board. Reports how many shots
    the better strategy saves per game and whether that is significant.
    """
    wide = results_df.pivot(index="game", columns="strategy", values="shots")
    names = list(results_df["strategy"].drop_duplicates())
    rows = []
    for weaker, stronger in zip(names[:-1], names[1:]):
        diff = wide[weaker] - wide[stronger]
        stat = wilcoxon(wide[weaker], wide[stronger])
        rows.append({
            "comparison": f"{stronger} vs {weaker}",
            "mean_shots_saved": round(diff.mean(), 2),
            "share_games_better": round((diff > 0).mean(), 3),
            "p_value": stat.pvalue,
        })
    return pd.DataFrame(rows)


def empirical_occupancy(size, fleet, style, n_boards, seed):
    """Share of boards (in a given placement style) on which each cell holds a
    ship -- what an attacker would learn by watching an opponent's habits."""
    total = np.zeros(size * size)
    for b in range(n_boards):
        total += place_fleet(size, fleet, np.random.default_rng([seed, b]), style=style).occupancy()
    return total / n_boards


def run_defense_matrix(attackers, styles, n_games, size, fleet, seed):
    """
    Every attacker against every placement style. `attackers` maps a name to
    a function style -> strategy, so an attacker can adapt to the opponent's
    placement habits (or ignore the style and always return the same one).
    """
    frames = []
    for name, make_attacker in attackers.items():
        for style in styles:
            frames.append(run_simulation({name: make_attacker(style)}, n_games, size, fleet, seed, placement=style))
    return pd.concat(frames, ignore_index=True)
