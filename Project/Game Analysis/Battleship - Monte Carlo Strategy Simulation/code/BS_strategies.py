"""
BS_strategies.py
Attacker strategies, from naive to near-optimal. Every strategy has the same
signature and is stateless -- it decides only from what the attacker can
legitimately see:

    strategy(shots, remaining_lengths, size, rng) -> flat cell index

  shots             : flat grid of UNKNOWN / MISS / HIT / SUNK (see BS_game)
  remaining_lengths : lengths of the ships still afloat
  rng               : numpy Generator, used to break ties randomly

Levels:
  1. random_shot          -- uniform over unfired cells (baseline)
  2. hunt_target          -- random until a hit, then fire around open hits
  3. hunt_target_parity   -- as 2, but hunt only on a checkerboard-style grid
  4. probability_density  -- fire where the most still-possible ship
                             positions overlap
"""

from collections import Counter

import numpy as np

from BS_game import HIT, MISS, SUNK, UNKNOWN, all_placements, neighbor_mask


def _pick(candidates, rng):
    return int(rng.choice(candidates))


def _argmax_random(scores, rng):
    return _pick(np.flatnonzero(scores == scores.max()), rng)


def random_shot(shots, remaining_lengths, size, rng):
    return _pick(np.flatnonzero(shots == UNKNOWN), rng)


def _target_cells(shots, size):
    """Unfired cells orthogonally next to a hit whose ship is still afloat."""
    open_hits = shots == HIT
    if not open_hits.any():
        return None
    candidates = np.flatnonzero(neighbor_mask(open_hits, size) & (shots == UNKNOWN))
    return candidates if candidates.size else None


def hunt_target(shots, remaining_lengths, size, rng):
    targets = _target_cells(shots, size)
    if targets is not None:
        return _pick(targets, rng)
    return random_shot(shots, remaining_lengths, size, rng)


def hunt_target_parity(shots, remaining_lengths, size, rng):
    """
    Hunt only on cells with (row + col) % k == 0, where k is the shortest ship
    still afloat. Any ship of length >= k must cover one of those cells, so
    nothing can hide between them -- with k = 2 this is the classic
    checkerboard, and it widens automatically once the destroyer is sunk.
    """
    targets = _target_cells(shots, size)
    if targets is not None:
        return _pick(targets, rng)
    k = min(remaining_lengths)
    row, col = np.divmod(np.arange(size * size), size)
    candidates = np.flatnonzero((shots == UNKNOWN) & ((row + col) % k == 0))
    if candidates.size == 0:
        candidates = np.flatnonzero(shots == UNKNOWN)
    return _pick(candidates, rng)


def probability_map(shots, remaining_lengths, size, prior=None):
    """
    For every ship still afloat, count every position it could still occupy
    (not overlapping a miss or a sunk ship) and add up, per cell, how many of
    those positions cover it.

    Target mode: while there are open hits, only positions that pass through
    a hit count, weighted by how many hits they pass through -- so a line of
    two hits pulls the next shot onto the same line.

    `prior` (optional, flat grid) multiplies the map in hunt mode only: an
    attacker's learned belief about where *this* opponent tends to put ships
    (see STEP 09 of the notebook).
    """
    blocked = ((shots == MISS) | (shots == SUNK)).astype(float)
    hits = (shots == HIT).astype(float)
    target_mode = hits.any()

    density = np.zeros(size * size)
    for length, count in Counter(remaining_lengths).items():
        positions = all_placements(size, length)
        weight = (positions @ blocked == 0).astype(float)
        if target_mode:
            weight *= positions @ hits
        density += count * (weight @ positions)

    density[shots != UNKNOWN] = 0
    if prior is not None and not target_mode:
        density *= prior
    if density.sum() == 0:
        density = (shots == UNKNOWN).astype(float)
    return density


def make_probability_density(prior=None):
    """Probability-density strategy, optionally with a learned placement prior."""
    def probability_density(shots, remaining_lengths, size, rng):
        return _argmax_random(probability_map(shots, remaining_lengths, size, prior), rng)
    return probability_density


probability_density = make_probability_density()

STRATEGIES = {
    "Random": random_shot,
    "Hunt & Target": hunt_target,
    "Hunt & Target + Parity": hunt_target_parity,
    "Probability Density": probability_density,
}
