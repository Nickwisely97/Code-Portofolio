# Staffing Optimization — Linear Programming

**CV skill represented:** Queueing/Staffing Models (LP Scheduling).

## Problem
Minimize total headcount across a 24-hour operation split into six 4-hour demand blocks, where each employee works one continuous 8-hour shift covering two consecutive blocks. Two variants:
- `single_line_scheduling.ipynb` — one combined staff pool.
- `multi_line_scheduling.ipynb` — two separate pools (kitchen vs. service) with independent demand curves.

## Method
Integer Linear Programming (PuLP): decision variables are shift start counts per block, constraints enforce that每 block's overlapping shift coverage meets demand, objective minimizes total staff.

## Result
Single-line: 44 staff (down from a naive one-shift-per-block estimate). Multi-line: 32 kitchen + 12 service.

## How to run
Open either notebook in `code/`; both read `data/demand.xlsx` and write charts to `result/single_line/` or `result/multi_line/`.
