# Self-Order Terminal Staffing — End-to-End Queueing Simulation

**CV skill represented:** Queueing/Staffing Models, Discrete Event Simulation.

## Problem
McDonald's Self-Order Terminals (SOT) are often assumed to be a straightforward efficiency upgrade over human cashiers. But per-transaction speed tells only half the story: a human cashier has a **faster** Average Handling Time (AHT) — no on-screen menu hesitation, faster order clarification — while a SOT has a **slower** AHT. What flips the outcome is capacity: a restaurant can typically staff only **2 cashier lanes** (hard to expand — hiring, counter space), but can install **4 SOT kiosks** in the same footprint.

A first version of this project modeled only the order-taking counter with the closed-form Erlang C formula, targeting an average-wait-based service level. That missed two things a real deployment decision needs: **the kitchen** (a long order clears the register fine but can still back up in food prep), and **the actual target metric** — an order-to-food promise is a tail (P99) guarantee, not an average. This version fixes both, and calibrates the kitchen and AHT figures against real QSR benchmarks rather than picking numbers arbitrarily.

## Method
Discrete-Event Simulation (SimPy) of a two-stage tandem queue: order-taking → kitchen. Order size (item count) is drawn once per customer and drives the service time at **both** stages, so a big order is correlated-slow at the register and in the kitchen — this correlation is exactly why a closed-form formula (which assumes independence) can't be extended cleanly to a percentile of the combined time. The kitchen is a **single shared, fixed resource** competed for by both scenarios, since the kitchen doesn't know which channel took the order.

Three complementary analyses:
1. **Constant-arrival-rate sweep** — the capacity-planning question: at a fixed, held-steady demand, at what arrival rate does each scenario's P99 breach the SLA, and at what point does the system fully collapse?
2. **One realistic operating day (07:00–23:00)** — the validation question: under a demand curve with a taller lunch peak and a shorter after-office/dinner peak (simulated via Poisson thinning, a non-homogeneous Poisson process), does a normal business day actually breach the SLA? Produces a full per-customer event log (arrival time, order size, time in each queue, completion time).
3. **A third front end: both channels open at once** — 2 cashiers and 4 kiosks running simultaneously, each customer joining whichever line is shorter (join-the-shorter-queue). Tests whether combining channels raises capacity, how many kitchen stations that would take to actually use, and which configuration needs the fewest total staff.

Code is modular: `queueing_model.py` (simulation mechanics, capacity math), `simulation_runner.py` (repeated-run orchestration — sweeps, stress tests, the operating day), `plotting.py` (all charts), `report_builder.py` (the executive PowerPoint deck, built on the portfolio's shared `Executive_Report_Template`). `order_to_food_simulation.ipynb` just wires these together and narrates the story, with every tunable assumption (SLA target, replication count, operating hours, demand-curve shape) collected in one `CONFIG` dict at the top.

Assumptions (illustrative, not measured restaurant data): `data/scenario_parameters.csv`, `data/kitchen_parameters.csv`, `data/order_size_distribution.csv`.

| Resource | Servers | Service time (Gamma, CV≈0.29) |
|---|---|---|
| Human Cashier (order stage) | 2 | 35s base + 7s/item |
| SOT (order stage) | 4 | 60s base + 8s/item |
| Kitchen (shared) | 3 | 55s base + 15s/item |

**Reality check on these numbers:** field studies put kiosk order-entry around 45–90s and counter/cashier order-entry around 60–120s — this model's ~55–82s means (depending on order size and channel) sit inside those bands. An earlier pass used a much steeper 30s/item kitchen figure, which capped the shared kitchen at just ~78/hr — *below* even the 2-register cashier's own ~132/hr ceiling. Checked against real benchmarks (a single McDonald's drive-thru lane reportedly handles on the order of 90 cars/hour at busy locations; industry-wide average drive-thru order-to-food time runs ~5.5–6 minutes even at that volume), 78/hr was an unrealistically low kitchen throughput — no restaurant stays staffed that unevenly for long. **15s/item (kitchen ceiling ≈111/hr) is the more defensible calibration used here.**

**Metric:** P99 of total time from order to food ≤ **600 seconds (10 minutes)**.

## Result
**Theoretical capacity ceilings** (`servers × 3600 / mean service time`): Cashier registers cap at **132/hr**, SOT kiosks at **175/hr**, and the shared **kitchen caps at ~111/hr** — close to the cashier's own ceiling and clearly below the SOT's.

**Below the shared collapse zone (arrival rates up to ~95/hr, where the SLA is still largely met), the cashier scenario is consistently better** — lower P99 and higher service level at all 8 tested points in that range. P99-safe capacity: cashier ≈**92.2/hr**, SOT ≈**89.6/hr** — a modest ~3% edge for the cashier. Past that point, deep in the congestion zone (100–118/hr) where both scenarios are already badly failing the SLA, the ranking becomes noisy and inconsistent between the two — unsurprising once both queues are effectively unstable.

At a **95/hr** stress test, close to the shared kitchen ceiling:

| Scenario | Order queue | Order service | Kitchen queue | Kitchen prep | P99 total | % within 10 min |
|---|---|---|---|---|---|---|
| Human Cashier | 33s | 55s | 69s | 97s | 708s | 97.7% |
| Self-Order Terminal | 6s | 82s | 90s | 97s | 726s | 97.4% |

Once the kitchen is a genuine co-bottleneck for both, the SOT's extra kiosks buy little, while its slower per-transaction AHT is a small but consistent added cost. **Both scenarios collapse at essentially the same arrival rate** (observed between 110–114/hr), matching the shared kitchen's ~111/hr theoretical ceiling — instead of the cashier failing early while the SOT sails on (the fast-kitchen finding) or the two collapsing far below any front-end capacity (the earlier, overly-slow-kitchen finding).

**The headline finding across all three calibrations is the same, and it's the most important one: the conclusion is highly sensitive to kitchen speed relative to front-end capacity.** A fast kitchen lets the SOT's extra kiosks translate into a real capacity edge; a kitchen that's a genuine co-bottleneck (the realistic case modeled here) mostly erases that edge. A real deployment decision needs a measured kitchen-throughput number for the specific store, not an assumed one — the front end is only ever as fast as the kitchen behind it.

**Validated against a realistic day (lunch peak 90/hr, dinner peak 65/hr, 07:00–23:00):** both scenarios hold the SLA all day — **zero breaches across 1,276 simulated customers**, because the lunch peak sits right at, not past, the P99-safe capacity found above. But the SOT still runs meaningfully hotter throughout: mean total time 201s vs. the cashier's 163s, worst case 516s vs. 363s. Even on a day the SLA is never breached, the SOT scenario has noticeably less headroom.

**A third front end — both channels open at once — doesn't raise capacity at all under today's kitchen.** Cashier-only (≈92/hr), kiosk-only (≈90/hr), and combined (≈90/hr) all collapse at essentially the same arrival rate, since all three still share the same 3-station kitchen. Combining channels changes headcount, not throughput — and since kiosks need no order-taking staff, **kiosk-only is the most labor-efficient configuration today**: ~29.9 customers/hour per staff member (3 total staff) vs. ~18.4 (cashier, 5 staff) and ~18.0 (combined, 5 staff).

Combining channels only pays off paired with a kitchen expansion. The combined front end's intrinsic (kitchen-unconstrained) capacity is ~290/hr; reaching it takes **9 kitchen stations — 6 more than today's 3**. And that investment only pays off with the combined front end: with that same 9-station kitchen, kiosk-only still tops out at ~153/hr (its own 4-kiosk ceiling becomes the new limit, wasting most of the extra kitchen), while combined reaches ~280/hr (25.4/hr per staff, 11 total staff) — actually *more* labor-efficient than either of today's under-provisioned setups, once the kitchen and the front end are sized together.

## How to run
Open `code/order_to_food_simulation.ipynb` — all tunable assumptions live in the `CONFIG` dict in its second cell. It reads the three CSVs in `data/`, imports the modular code below, and writes:
- Charts to `result/figures/` (order-size distribution, capacity ceilings, P99 & service-level curves, stress-test breakdown, daily demand profile, hourly wait time, 3-way comparison, workforce efficiency, kitchen sizing)
- `result/simulation_sweep.csv`, `result/combined_sweep.csv`, `result/stress_test_breakdown.csv`, `result/kitchen_sizing.csv`, `result/event_log.csv` (the full per-customer log for one simulated operating day)
- `result/slides/Executive_SOT_Staffing_Report_<date>.pptx` — an 8-slide executive deck built by `code/report_builder.py` on the portfolio's shared `Executive_Report_Template`

Code structure: `code/queueing_model.py` (simulation + capacity math, including the combined-channel and operating-day simulations), `code/simulation_runner.py` (sweep / stress-test / operating-day / kitchen-sizing orchestration), `code/plotting.py` (charts), `code/report_builder.py` (executive report).
