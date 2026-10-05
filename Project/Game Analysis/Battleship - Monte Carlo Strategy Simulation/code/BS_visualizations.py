"""
BS_visualizations.py
All chart-building for the Battleship analysis, pulled out of the notebook so
STEP cells stay short. Same larger shared font baseline as the Snake and
Ladder project, so charts stay readable once exported to slides.

Boards are drawn with rows A-J and columns 1-10, as on the physical game.
"""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import ListedColormap

from BS_game import HIT, MISS, SUNK
from BS_strategies import probability_map

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})

STRATEGY_COLORS = ["#9AA5B1", "#4C72B0", "#DD8452", "#55A868"]


def _board_axes(ax, size):
    ax.set_xticks(np.arange(size) + 0.5, [str(i + 1) for i in range(size)], rotation=0)
    ax.set_yticks(np.arange(size) + 0.5, [chr(ord("A") + i) for i in range(size)], rotation=0)
    ax.tick_params(length=0)


def plot_board(board, title, save_path):
    """One defender's fleet, each ship in its own color."""
    size = board.size
    grid = np.full((size, size), np.nan)
    for i, (_, cells) in enumerate(board.ships):
        grid.flat[cells] = i
    fig, ax = plt.subplots(figsize=(7, 7))
    sns.heatmap(grid, cmap="tab10", vmin=0, vmax=9, cbar=False, linewidths=1, linecolor="white",
                square=True, ax=ax, mask=np.isnan(grid))
    ax.set_facecolor("#DCEAF7")
    for name, cells in board.ships:
        r, c = np.divmod(cells, size)
        ax.text(c.mean() + 0.5, r.mean() + 0.5, name, ha="center", va="center", fontsize=10, color="white",
                fontweight="bold", rotation=90 if len(set(c)) == 1 and len(cells) > 1 else 0)
    _board_axes(ax, size)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_heatmaps(grids, size, save_path, fmt=".0f", cmap="rocket_r"):
    """Side-by-side board heatmaps -- grids: {title: flat array}."""
    fig, axes = plt.subplots(1, len(grids), figsize=(8 * len(grids), 7.5))
    axes = np.atleast_1d(axes)
    for ax, (title, values) in zip(axes, grids.items()):
        sns.heatmap(values.reshape(size, size), annot=True, fmt=fmt, cmap=cmap, square=True,
                    cbar=False, annot_kws={"fontsize": 10}, ax=ax)
        _board_axes(ax, size)
        ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


STRATEGY_RULES = {
    "Random": "fire anywhere not yet fired at",
    "Hunt & Target": "random, but after a hit\nfire next to it",
    "Hunt & Target + Parity": "hunt on a checkerboard only,\nafter a hit fire next to it",
    "Probability Density": "fire where the most possible\nship positions overlap",
}


def plot_strategy_decisions(scenarios, remaining_lengths, size, save_path):
    """
    Side-by-side illustration of how each strategy picks its next shot.
    scenarios : {row label: flat shots grid}. Shaded cells are the ones the
    strategy considers; for Probability Density darker = more likely, and the
    yellow box marks the cell it actually fires at.
    """
    from BS_strategies import STRATEGIES, decision_weights

    names = list(STRATEGIES)
    fig, axes = plt.subplots(len(scenarios), len(names), figsize=(5.2 * len(names), 6.2 * len(scenarios)))
    r, c = np.divmod(np.arange(size * size), size)
    for row, (label, shots) in enumerate(scenarios.items()):
        for col, name in enumerate(names):
            ax = axes[row, col]
            weights = decision_weights(name, shots, remaining_lengths, size)
            # Rule-based strategies: candidate or not (two flat colors). Probability Density: graded map.
            cmap = "mako_r" if name == "Probability Density" else ListedColormap(["#EAF2F8", "#3A7DC9"])
            sns.heatmap((weights / weights.max()).reshape(size, size), cmap=cmap, vmin=0, vmax=1.15 if name == "Probability Density" else 1,
                        square=True, cbar=False, linewidths=0.5, linecolor="white", ax=ax)
            for state, marker, color in [(MISS, "x", "#7F8C8D"), (HIT, "o", "#E74C3C")]:
                idx = shots == state
                ax.scatter(c[idx] + 0.5, r[idx] + 0.5, marker=marker, s=160, color=color, linewidths=3, zorder=3)
            n_candidates = int((weights > 0).sum())
            if name == "Probability Density":
                for cell in np.flatnonzero(weights == weights.max()):
                    tr, tc = divmod(cell, size)
                    ax.add_patch(plt.Rectangle((tc, tr), 1, 1, fill=False, edgecolor="#F1C40F", linewidth=4, zorder=4))
                caption = "fires at the darkest cell (yellow)"
            else:
                caption = f"picks 1 of {n_candidates} shaded cells at random"
            _board_axes(ax, size)
            ax.set_xlabel(caption, fontsize=13)
            if row == 0:
                ax.set_title(f"{name}\n{STRATEGY_RULES[name]}", fontsize=14, fontweight="bold")
            if col == 0:
                ax.set_ylabel(label, fontsize=15, fontweight="bold")
    fig.suptitle("How Each Strategy Chooses Its Next Shot (same board, same moment)   x = miss, o = hit",
                 fontsize=18, y=1.0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_shots_distribution(results_df, save_path):
    """Histogram of shots-to-win per strategy, with each median marked."""
    fig, ax = plt.subplots(figsize=(14, 6))
    names = list(results_df["strategy"].drop_duplicates())
    bins = np.arange(results_df["shots"].min() - 0.5, results_df["shots"].max() + 1.5)
    for name, color in zip(names, STRATEGY_COLORS):
        shots = results_df.loc[results_df["strategy"] == name, "shots"]
        ax.hist(shots, bins=bins, density=True, alpha=0.55, color=color, label=f"{name} (median {shots.median():.0f})")
        ax.axvline(shots.median(), color=color, linestyle="--", linewidth=2)
    ax.set_title("Shots Needed to Sink the Whole Fleet")
    ax.set_xlabel("Shots fired")
    ax.set_ylabel("Share of games")
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_shots_cdf(results_df, save_path, max_shots=100):
    """P(fleet sunk within n shots) -- the 'how fast do I usually win' view."""
    fig, ax = plt.subplots(figsize=(14, 6))
    names = list(results_df["strategy"].drop_duplicates())
    n = np.arange(0, max_shots + 1)
    for name, color in zip(names, STRATEGY_COLORS):
        shots = np.sort(results_df.loc[results_df["strategy"] == name, "shots"].to_numpy())
        ax.plot(n, np.searchsorted(shots, n, side="right") / len(shots), color=color, linewidth=3, label=name)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_title("Probability of Having Won Within N Shots")
    ax.set_xlabel("Shots fired (N)")
    ax.set_ylabel("P(all ships sunk)")
    ax.set_xlim(17, max_shots)
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def _draw_attacker_view(ax, entry, size, prior=None):
    """Probability map before this shot, with past misses/hits/sunk cells
    overlaid and the chosen cell outlined."""
    shots = entry["shots_before"]
    density = probability_map(shots, entry["remaining_before"], size, prior)
    ax.clear()
    sns.heatmap((density / density.max()).reshape(size, size), cmap="mako_r", vmin=0, vmax=1, square=True,
                cbar=False, linewidths=0.5, linecolor="white", ax=ax)
    r, c = np.divmod(np.arange(size * size), size)
    for state, marker, color in [(MISS, "x", "#7F8C8D"), (HIT, "o", "#E74C3C"), (SUNK, "s", "#2C3E50")]:
        idx = shots == state
        ax.scatter(c[idx] + 0.5, r[idx] + 0.5, marker=marker, s=140, color=color, linewidths=2.5)
    tr, tc = divmod(entry["cell"], size)
    ax.add_patch(plt.Rectangle((tc, tr), 1, 1, fill=False, edgecolor="#F1C40F", linewidth=4))
    _board_axes(ax, size)
    ax.set_title(f"Shot {entry['shot']} -> {entry['result']}")


def plot_game_snapshots(log, size, shot_numbers, save_path):
    """Small multiples of the attacker's probability map through one game."""
    fig, axes = plt.subplots(1, len(shot_numbers), figsize=(5.5 * len(shot_numbers), 6))
    for ax, n in zip(axes, shot_numbers):
        _draw_attacker_view(ax, log[n - 1], size)
    fig.suptitle("Probability Density Attacker -- what it 'sees' before each shot "
                 "(darker = more likely; x miss, o hit, ■ sunk, yellow = chosen cell)", fontsize=15)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def save_game_gif(log, size, save_path, fps=3):
    """The same view as plot_game_snapshots, for every shot, as a GIF."""
    fig, ax = plt.subplots(figsize=(6, 6.4))
    animation = FuncAnimation(fig, lambda i: _draw_attacker_view(ax, log[i], size), frames=len(log))
    animation.save(save_path, writer=PillowWriter(fps=fps), dpi=90)
    plt.close(fig)
    return save_path


def plot_defense_matrix(pivot, save_path):
    """Median shots-to-win, attacker (rows) x placement style (columns).
    Higher = better for the defender."""
    fig, ax = plt.subplots(figsize=(11, 4 + 0.6 * len(pivot)))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", cbar_kws={"label": "Mean shots to win"},
                linewidths=1, linecolor="white", annot_kws={"fontsize": 15}, ax=ax)
    ax.set_title("Defense: Mean Shots the Attacker Needs, by Placement Style\n(higher = better for the defender)")
    ax.set_xlabel("Defender placement style")
    ax.set_ylabel("Attacker")
    ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path
