"""
SnL_markov_chain.py
Builds the Snake & Ladder transition matrix directly in Python -- no
pre-baked Excel matrix -- and runs the absorbing-Markov-chain analysis on
top of it (expected steps, fastest route, snake/ladder impact).

Building the matrix in code (rather than reading a fixed spreadsheet) is
what makes the dice count a parameter: `build_transition_matrix(n_dice=...)`
works for any number of dice, so the same analysis can be re-run and
compared across dice counts instead of being locked to whatever variants
happen to have a matching Excel file.

Board size, the ladder/snake layout, and the dice count are all inputs the
notebook owns and passes in explicitly (see its Parameters cell) rather
than constants hidden in this module -- so they're visible and easy to
change without touching this file.
"""

from itertools import product

import numpy as np
import pandas as pd


def build_snake_ladder_df(ladders, snakes):
    """Board layout as a DataFrame -- start, finish, type, distance.

    ladders, snakes : lists of (start, finish) tuples.
    """
    rows = [(s, f, "Ladder", f - s) for s, f in ladders]
    rows += [(s, f, "Snake", f - s) for s, f in snakes]
    return pd.DataFrame(rows, columns=["start", "finish", "type", "distance"]).sort_values("start").reset_index(drop=True)


def dice_sum_distribution(n_dice):
    """P(sum of n_dice standard 6-sided dice = s), for every reachable sum s."""
    total_outcomes = 6 ** n_dice
    counts = {}
    for combo in product(range(1, 7), repeat=n_dice):
        s = sum(combo)
        counts[s] = counts.get(s, 0) + 1
    return {s: c / total_outcomes for s, c in counts.items()}


def build_transition_matrix(n_dice, board_size, snake_ladder_df):
    """
    Build the board_size x board_size transition matrix for n_dice standard
    dice.

    Rules:
      - From square i, a roll of d moves to i+d; a roll that overshoots the
        board bounces back (need to land exactly on the last square to win).
      - Any square that is a snake/ladder START is a transient pass-through
        state: its own row redirects deterministically (100%) to its FINISH
        square, so landing there for one step always leads there next.
      - The last square is a true absorbing state (100% self-transition) --
        this also fixes a bug in this project's old hand-built Excel matrix,
        where the 2-dice sheet's last row wasn't actually self-absorbing.
    """
    redirect = dict(zip(snake_ladder_df["start"], snake_ladder_df["finish"]))
    dice_dist = dice_sum_distribution(n_dice)

    n = board_size
    M = np.zeros((n, n))
    M[n - 1, n - 1] = 1.0

    for pos in range(1, n):
        if pos in redirect:
            M[pos - 1, redirect[pos] - 1] = 1.0
            continue
        for roll, p in dice_dist.items():
            target = pos + roll
            if target > n:
                target = n - (target - n)
            M[pos - 1, target - 1] += p

    return M


def analyze_expected_steps(transition_matrix):
    """Expected number of turns to finish from every square (fundamental matrix N = (I - Q)^-1)."""
    n = transition_matrix.shape[0]
    Q = transition_matrix[:n - 1, :n - 1]
    I = np.identity(n - 1)
    N = np.linalg.inv(I - Q)
    return N @ np.ones(n - 1)


def variance_of_steps(transition_matrix):
    """
    Exact variance of the number of turns to finish from every square, via
    the standard absorbing-chain identity Var(T) = (2N - I)t - t^2, where
    N is the fundamental matrix and t = N @ 1 is analyze_expected_steps.

    This is exact (no truncation), unlike estimating mean/variance from a
    finish_time_distribution() run capped at some max_steps -- that
    distribution has a long right tail, so a step cap that's too low
    silently under-estimates both, sometimes by a lot.
    """
    n = transition_matrix.shape[0]
    Q = transition_matrix[:n - 1, :n - 1]
    I = np.identity(n - 1)
    N = np.linalg.inv(I - Q)
    t = N @ np.ones(n - 1)
    t2 = (2 * N - I) @ t
    return t2 - t ** 2


def find_fastest_route(transition_matrix, expected_steps, start_pos=1):
    """Greedy walk from start_pos always choosing the reachable square with the lowest expected remaining steps."""
    n = transition_matrix.shape[0]
    current_pos = start_pos
    route = [current_pos]

    while current_pos != n:
        transition_probs = transition_matrix[current_pos - 1]

        if 1.0 in transition_probs:
            current_pos = int(np.argmax(transition_probs)) + 1
            route.append(current_pos)
            continue

        candidates = [(pos + 1, expected_steps[pos] if pos < len(expected_steps) else 0)
                      for pos, prob in enumerate(transition_probs) if prob > 0]
        if not candidates:
            break
        candidates.sort(key=lambda x: x[1])
        current_pos = candidates[0][0]
        route.append(current_pos)

    return route


def snake_ladder_impact(transition_matrix, snake_ladder_df, n_dice, board_size):
    """
    For every snake/ladder, compare expected turns from square 1 with vs.
    without that single element. Impact is positive when the element
    *reduces* expected steps (ladders, typically) and negative when it
    *adds* steps (snakes, typically) -- i.e. impact = (expected steps
    without the element) - (expected steps with it).

    Removing an element has to rebuild the *whole* matrix, not just patch
    its start square's own row: other squares can transition onto a
    snake/ladder's start square as a real intermediate state (see
    build_transition_matrix), so removing the element changes which
    squares are reachable, not just what happens after landing on it.
    """
    original_expected = analyze_expected_steps(transition_matrix)[0]
    results = []

    for idx, row in snake_ladder_df.iterrows():
        start_pos, end_pos, element_type = int(row["start"]), int(row["finish"]), row["type"]

        reduced_df = snake_ladder_df.drop(idx)
        modified_matrix = build_transition_matrix(n_dice=n_dice, board_size=board_size, snake_ladder_df=reduced_df)

        new_expected = analyze_expected_steps(modified_matrix)[0]
        results.append((start_pos, end_pos, element_type, new_expected - original_expected, end_pos - start_pos))

    return pd.DataFrame(results, columns=["Start", "End", "Type", "Impact", "Distance"])


def position_distribution_over_steps(transition_matrix, steps):
    """State-probability vector after each step count in `steps` (list of ints)."""
    n = transition_matrix.shape[0]
    state = np.zeros(n)
    state[0] = 1.0
    results = {}
    last_step = 0
    for target_step in sorted(steps):
        for _ in range(target_step - last_step):
            state = state @ transition_matrix
        results[target_step] = state.copy()
        last_step = target_step
    return results


def finish_time_distribution(transition_matrix, max_steps=200):
    """
    Cumulative and per-step probability of having finished by step k, for
    k = 1..max_steps.
    """
    n = transition_matrix.shape[0]
    state = np.zeros(n)
    state[0] = 1.0
    cumulative = []
    for _ in range(max_steps):
        state = state @ transition_matrix
        cumulative.append(state[-1])
    density = [cumulative[0]] + [cumulative[i] - cumulative[i - 1] for i in range(1, len(cumulative))]
    return pd.DataFrame({"step": range(1, max_steps + 1), "density": density, "cumulative": cumulative})
