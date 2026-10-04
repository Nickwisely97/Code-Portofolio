"""
BS_game.py
Battleship rules engine: every legal ship position, fleet placement (in
several placement styles, used by the defense analysis), and firing.

Board size and fleet are inputs the notebook owns and passes in explicitly
(see its Parameters cell) rather than constants hidden in this module.

Cells are addressed by flat index (row * size + col). The attacker only ever
sees the `shots` grid, with one of four states per cell:
  UNKNOWN -- not fired at yet
  MISS    -- fired at, water
  HIT     -- fired at, part of a ship that is still afloat
  SUNK    -- part of a ship that has been sunk
When a ship sinks its cells are revealed as SUNK -- the usual "you sank my
battleship" announcement, simplified so the attacker knows exactly which hits
belonged to it.
"""

from functools import lru_cache

import numpy as np

UNKNOWN, MISS, HIT, SUNK = 0, 1, 2, 3
PLACEMENT_STYLES = ("random", "edge", "cluster", "spread")


@lru_cache(maxsize=None)
def all_placements(size, length):
    """Every legal horizontal/vertical position for a ship of `length`, as an
    (n_placements, size*size) 0/1 float matrix -- float so the probability
    model can count overlaps with a single matrix product."""
    masks = []
    for r in range(size):
        for c in range(size - length + 1):
            m = np.zeros((size, size))
            m[r, c:c + length] = 1
            masks.append(m.ravel())
    if length > 1:
        for r in range(size - length + 1):
            for c in range(size):
                m = np.zeros((size, size))
                m[r:r + length, c] = 1
                masks.append(m.ravel())
    return np.array(masks)


def neighbor_mask(cells, size, diagonal=False):
    """Boolean flat mask of the cells touching `cells` (excluding `cells`)."""
    grid = cells.reshape(size, size)
    padded = np.pad(grid, 1)
    shifts = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        shifts += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    out = np.zeros_like(grid)
    for dr, dc in shifts:
        out |= padded[1 + dr:1 + dr + size, 1 + dc:1 + dc + size]
    return out.ravel() & ~cells


def border_mask(size):
    grid = np.zeros((size, size), bool)
    grid[0, :] = grid[-1, :] = grid[:, 0] = grid[:, -1] = True
    return grid.ravel()


class Board:
    """One defender's board: where the ships are, and what has been fired."""

    def __init__(self, size, ships):
        """ships : list of (name, array of flat cell indices)."""
        self.size = size
        self.ships = ships
        self.owner = np.full(size * size, -1)
        for i, (_, cells) in enumerate(ships):
            self.owner[cells] = i
        self.cells_left = [len(cells) for _, cells in ships]
        self.shots = np.full(size * size, UNKNOWN, dtype=np.int8)
        self.n_shots = 0

    def fire(self, cell):
        """Fire at flat index `cell`. Returns 'miss', 'hit' or 'sunk'."""
        if self.shots[cell] != UNKNOWN:
            raise ValueError(f"Cell {cell} has already been fired at")
        self.n_shots += 1
        ship = self.owner[cell]
        if ship < 0:
            self.shots[cell] = MISS
            return "miss"
        self.cells_left[ship] -= 1
        if self.cells_left[ship] == 0:
            self.shots[self.ships[ship][1]] = SUNK
            return "sunk"
        self.shots[cell] = HIT
        return "hit"

    @property
    def all_sunk(self):
        return all(n == 0 for n in self.cells_left)

    def remaining_lengths(self):
        """Lengths of the ships still afloat -- public information, since
        every sinking is announced."""
        return [len(cells) for (_, cells), left in zip(self.ships, self.cells_left) if left > 0]

    def occupancy(self):
        return self.owner >= 0


def place_fleet(size, fleet, rng, style="random", max_restarts=1000):
    """
    Place every ship in `fleet` ({name: length}, placed largest-first) on a
    fresh board, in one of four placement styles:
      random  -- any legal position, ships may touch
      edge    -- every ship lies entirely on the board's outer border
      cluster -- every ship after the first touches an already-placed ship
      spread  -- no two ships touch, not even diagonally
    Each ship is drawn uniformly from the positions that are legal given the
    ships already placed; if a style paints itself into a corner, restart.
    """
    if style not in PLACEMENT_STYLES:
        raise ValueError(f"Unknown placement style {style!r}; choose from {PLACEMENT_STYLES}")
    border = border_mask(size)
    order = sorted(fleet.items(), key=lambda kv: -kv[1])

    for _ in range(max_restarts):
        occupied = np.zeros(size * size, bool)
        ships = []
        for name, length in order:
            candidates = all_placements(size, length).astype(bool)
            ok = ~(candidates & occupied).any(axis=1)
            if style == "edge":
                ok &= ~(candidates & ~border).any(axis=1)
            elif style == "spread":
                ok &= ~(candidates & neighbor_mask(occupied, size, diagonal=True)).any(axis=1)
            elif style == "cluster" and ships:
                ok &= (candidates & neighbor_mask(occupied, size)).any(axis=1)
            options = np.flatnonzero(ok)
            if options.size == 0:
                break
            mask = candidates[rng.choice(options)]
            occupied |= mask
            ships.append((name, np.flatnonzero(mask)))
        else:
            return Board(size, ships)
    raise RuntimeError(f"Could not place fleet in style {style!r} after {max_restarts} restarts")
