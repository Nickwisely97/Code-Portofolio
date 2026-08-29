# Movie Recommendation — Collaborative Filtering

**CV skill represented:** unsupervised/similarity-based modeling (general ML breadth).

## Problem
Recommend movies to a user based on what similar users rated highly (user-user collaborative filtering).

## Method
Pivot a user × movie rating table, compute user-user cosine similarity (chosen over Pearson correlation because it handles the sparse "unwatched = 0" structure naturally), then recommend a similar user's highly-rated, not-yet-seen titles.

## Status — known limitation
This notebook currently runs on a small, hand-built illustrative rating table (6 users) to demonstrate the mechanism clearly. It's a deliberately simplified teaching example, not yet an applied project at the same scale as the others in this repo — planned upgrade: rebuild on a real dataset (e.g. MovieLens) with sparse-matrix handling and a train/test split for proper evaluation (precision@k).

## How to run
Open `code/movie_recommendation.ipynb` and run top to bottom — no external data file needed yet.
