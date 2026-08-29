# Snake and Ladder — Markov Chain Analysis

**Why a board game:** a public, non-confidential stand-in for stochastic-process modeling — see the repo root README for how each Game Analysis project maps to a real skill.
**CV skill represented:** Markov chain / absorbing-state modeling — general problem-solving depth beyond the explicit CV skill list.

## Problem
How many turns does Snake and Ladder actually take to finish, and how do snakes/ladders reshape the odds along the way?

## Method
The board is modeled as an absorbing Markov chain (one absorbing state: the finish square). Transition probabilities come from the die roll and are perturbed by every snake/ladder. Compares 1-dice vs. 2-dice variants.

## Deliverables
- Expected number of turns to finish (fundamental matrix).
- Heatmap of time spent per board position.
- Snake/ladder impact ranking (which ones most reshape the expected game length).
- Position distribution and finish-time density.

Reference board layouts used while building the transition matrix are in `docs/`.

## How to run
Open `code/snake_and_ladder_analysis.ipynb` and run top to bottom.
