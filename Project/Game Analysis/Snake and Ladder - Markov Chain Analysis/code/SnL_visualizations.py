"""
SnL_visualizations.py
All chart-building for the Snake & Ladder analysis. Pulled out of the
notebook so STEP cells stay short, and so every chart shares one larger,
more readable font baseline -- the board heatmap especially was hard to
read at the old default sizes with 100 squares crammed into one figure.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
})


def plot_expected_steps(expected_steps, save_path):
    plt.figure(figsize=(14, 6))
    sns.barplot(x=list(range(1, len(expected_steps) + 1)), y=expected_steps, color="#4C72B0")
    plt.title(f"Expected Steps from Each Position (From square 1: {expected_steps[0]:.2f} steps)")
    plt.xlabel("Position")
    plt.ylabel("Expected Steps")
    plt.xticks(range(0, len(expected_steps), 5), range(1, len(expected_steps) + 1, 5))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_board_heatmap(expected_steps, save_path):
    """
    Board laid out as a real 10x10 boustrophedon (snake-order) grid, colored
    by expected steps to finish from that square. Fonts are deliberately
    large -- with 100 squares on one figure, anything smaller becomes
    unreadable once exported to a slide.
    """
    board_data = np.zeros((10, 10))
    values_to_plot = np.append(expected_steps, np.nan)

    for pos in range(1, 101):
        row = (pos - 1) // 10
        col = (pos - 1) % 10 if row % 2 == 0 else 9 - ((pos - 1) % 10)
        board_data[9 - row, col] = values_to_plot[pos - 1]

    box_numbers = np.zeros((10, 10), dtype=int)
    for row in range(10):
        for col in range(10):
            pos = (9 - row) * 10 + (col if (9 - row) % 2 == 0 else 9 - col)
            box_numbers[row, col] = pos + 1

    plt.figure(figsize=(18, 16))
    mask = np.isnan(board_data)
    ax = sns.heatmap(
        board_data, cmap="YlOrRd", annot=False, mask=mask, linewidths=0.6,
        vmin=np.nanmin(board_data), vmax=np.nanmax(board_data),
        cbar_kws={"label": "Expected Steps", "shrink": 0.8},
    )
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)
    cbar.set_label("Expected Steps", fontsize=16)

    for i in range(10):
        for j in range(10):
            if box_numbers[i][j] == 100:
                ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=True, color="darkgreen"))

    for i in range(10):
        for j in range(10):
            box_num = box_numbers[i][j]
            if box_num == 1:
                ax.text(j + 0.08, i + 0.18, f"{box_num}", fontsize=12, fontweight="bold", color="black")
                ax.text(j + 0.5, i + 0.5, "START", fontsize=12, ha="center", color="black", fontweight="bold")
                ax.text(j + 0.5, i + 0.85, f"{expected_steps[box_num - 1]:.1f}", fontsize=17, ha="center", color="black", fontweight="bold")
            elif box_num == 100:
                ax.text(j + 0.08, i + 0.22, f"{box_num}", fontsize=12, fontweight="bold", color="white")
                ax.text(j + 0.5, i + 0.6, "FINISH", fontsize=15, ha="center", color="white", fontweight="bold")
            else:
                ax.text(j + 0.08, i + 0.18, f"{box_num}", fontsize=12, fontweight="bold", color="black")
                ax.text(j + 0.5, i + 0.72, f"{expected_steps[box_num - 1]:.1f}", fontsize=17, ha="center", color="black", fontweight="bold")

    ax.set_xticklabels([])
    ax.set_yticklabels([])
    plt.title("Heat Map: Expected Steps to Finish — Snake and Ladder", fontsize=22, pad=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    return save_path


def plot_snl_impact(impact_df, save_path):
    plt.figure(figsize=(16, 10))
    colors = {"Snake": "#C0392B", "Ladder": "#27AE60"}
    impact_df = impact_df.sort_values("Start")

    bars = plt.bar(impact_df["Start"].astype(str), impact_df["Impact"],
                    color=[colors[t] for t in impact_df["Type"]])
    plt.axhline(y=0, color="black", linestyle="-", alpha=0.3)

    for bar, start, end, impact in zip(bars, impact_df["Start"], impact_df["End"], impact_df["Impact"]):
        label = f"{start}->{end}\n({impact:.1f})"
        if abs(impact) >= 3:
            # Big enough bar: label fits inside it, centered.
            y_pos, va, color = impact / 2, "center", ("white" if abs(impact) > 5 else "black")
        else:
            # Bar too short for the label -- place it just outside instead of overflowing.
            y_pos = impact + (0.7 if impact >= 0 else -0.7)
            va, color = ("bottom" if impact >= 0 else "top"), "black"
        plt.text(bar.get_x() + bar.get_width() / 2, y_pos, label,
                  ha="center", va=va, color=color, fontweight="bold", fontsize=9)

    plt.title("Impact of Snakes and Ladders on Game Completion", fontsize=20)
    plt.xlabel("Starting Position")
    plt.ylabel("Change in Expected Steps (+ = fewer steps, - = more steps)")

    red_patch = plt.Rectangle((0, 0), 1, 1, fc=colors["Snake"])
    green_patch = plt.Rectangle((0, 0), 1, 1, fc=colors["Ladder"])
    plt.legend([green_patch, red_patch], ["Ladder (reduces steps)", "Snake (increases steps)"], loc="upper right")

    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_position_distribution(distributions, save_path):
    """distributions: dict {step_count: state_probability_vector}, from markov_chain.position_distribution_over_steps."""
    steps = sorted(distributions.keys())
    n = len(distributions[steps[0]])
    nrows = int(np.ceil(len(steps) / 3))
    ncols = min(len(steps), 3)

    plt.figure(figsize=(18, 5 * nrows))
    for i, step in enumerate(steps):
        state = distributions[step]
        plt.subplot(nrows, ncols, i + 1)
        plt.bar(range(1, n + 1), state, color="#5DADE2")
        plt.bar(n, state[-1], color="#27AE60")
        plt.title(f"After {step} Steps\nP(finish) = {state[-1]:.4f}", fontsize=15)
        plt.xlabel("Position")
        plt.ylabel("Probability")

        threshold = max(0.05, np.max(state) * 0.7)
        for pos, prob in enumerate(state):
            if prob > threshold:
                plt.annotate(f"{prob:.3f}", xy=(pos + 1, prob), ha="center", va="bottom", fontsize=10)

        plt.grid(alpha=0.3)
        plt.xticks(np.arange(0, n + 1, 10))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_finish_distribution(finish_df, save_path):
    max_steps = len(finish_df)
    plt.figure(figsize=(16, 11))

    plt.subplot(2, 1, 1)
    plt.bar(finish_df["step"], finish_df["density"], color="#5DADE2", alpha=0.85)
    plt.title("Probability Distribution of Finishing the Game at a Specific Step", fontsize=17)
    plt.xlabel("Number of Steps")
    plt.ylabel("Probability")
    plt.grid(alpha=0.3)

    max_row = finish_df.loc[finish_df["density"].idxmax()]
    plt.annotate(f"Most likely: Step {int(max_row['step'])}\nP = {max_row['density']:.4f}",
                 xy=(max_row["step"], max_row["density"]), fontsize=13,
                 bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.6))

    plt.subplot(2, 1, 2)
    plt.plot(finish_df["step"], finish_df["cumulative"], color="#C0392B", linewidth=2.5)
    plt.axhline(y=0.5, color="#27AE60", linestyle="--", alpha=0.8, label="50% probability")
    plt.axhline(y=0.9, color="#2E86C1", linestyle="--", alpha=0.8, label="90% probability")

    step_50 = finish_df.loc[finish_df["cumulative"] >= 0.5, "step"].min()
    step_90 = finish_df.loc[finish_df["cumulative"] >= 0.9, "step"].min()
    if pd.notna(step_50):
        plt.annotate(f"50%: Step {int(step_50)}", xy=(step_50, 0.5), xytext=(step_50 + 5, 0.55),
                     arrowprops=dict(facecolor="#27AE60", shrink=0.05, width=1), fontsize=13)
    if pd.notna(step_90):
        plt.annotate(f"90%: Step {int(step_90)}", xy=(step_90, 0.9), xytext=(step_90 + 5, 0.85),
                     arrowprops=dict(facecolor="#2E86C1", shrink=0.05, width=1), fontsize=13)

    plt.title("Cumulative Probability of Finishing Within N Steps", fontsize=17)
    plt.xlabel("Number of Steps")
    plt.ylabel("Cumulative Probability")
    plt.grid(alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path


def plot_dice_comparison(comparison_df, save_path):
    """comparison_df: columns n_dice, expected_steps -- expected turns from square 1, for each dice count tried."""
    plt.figure(figsize=(10, 6))
    bars = plt.bar(comparison_df["n_dice"].astype(str), comparison_df["expected_steps"], color="#4C72B0")
    for bar, val in zip(bars, comparison_df["expected_steps"]):
        plt.text(bar.get_x() + bar.get_width() / 2, val + 0.5, f"{val:.1f}", ha="center", fontsize=13, fontweight="bold")
    plt.title("Expected Turns to Finish vs. Number of Dice", fontsize=18)
    plt.xlabel("Number of Dice Rolled per Turn")
    plt.ylabel("Expected Turns from Square 1")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return save_path
