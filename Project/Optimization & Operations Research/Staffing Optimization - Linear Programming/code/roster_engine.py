"""
roster_engine.py
Everything behind the monthly workforce roster, in one module:

  1. CONSTANTS     -- planning month, shifts, breaks, overtime, labor rules, penalties, pay
  2. DEMAND        -- customers per hour -> agents needed per hour
  3. SHIFT CODES   -- every concrete shift an agent can work, e.g. M10, E18_A2, N01_B1
  4. STAGE 1       -- staffing plan: how many agents, and how many work each shift code each day
  5. STAGE 2       -- day pattern: which days each agent works or is off
  6. STAGE 3       -- shift assignment: which shift code each agent works on each work day
  7. RESULTS       -- roster tables, coverage, penalties, compliance checks, pay
  8. CHARTS        -- figures used by the notebook and the executive report

The notebook only calls these functions and shows the results.

Shift codes: <shift><break hour>[_<B|A><overtime hours>]
  M10     morning shift 07:00-16:00, break 10:00-11:00
  M10_B1  the same with 1h overtime Before -> 06:00-16:00
  E18_A2  evening shift with 2h overtime After -> 14:00-01:00
Decision variables carry the full decision in their name, e.g.
  Agent01_20270101_M10_B1 = 1  ->  Agent01 works M10_B1 on 1 January 2027

Why three stages: one monthly model deciding head count, days off and shifts at once is
too large for the open-source CBC solver (a first attempt found no valid roster in 10
minutes). Each stage below is small enough to solve well, and each fixes what the next
one builds on.

Time convention: an OPERATING DAY runs 06:00 -> 06:00, so a night shift (21:00-06:00)
belongs to the day it starts. Hour k of an operating day is clock hour (6 + k) % 24.
"""

import calendar
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import date, timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pulp
import seaborn as sns
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch

# =============================================================================
# 1. CONSTANTS
# =============================================================================

YEAR, MONTH = 2027, 1                    # planning month: January 2027
DAY_START_HOUR = 6                       # operating day runs 06:00 -> 06:00

# Shifts: code -> (name, clock start hour, length incl. the 1h break, break-start options)
SHIFTS = {
    "M": ("Morning", 7, 9, (10, 11, 12)),     # 07:00-16:00, break M10 / M11 / M12
    "E": ("Evening", 14, 9, (17, 18, 19)),    # 14:00-23:00, break E17 / E18 / E19
    "N": ("Night", 21, 9, (0, 1, 2)),         # 21:00-06:00, break N00 / N01 / N02
}
OVERTIME_OPTIONS = (1, 2)                # hours Before (B) or After (A) a shift -- company cap: 2h

# Indonesian labor law (UU 13/2003 as amended by UU 6/2023; PP 35/2021) -- hard rules
MAX_WORK_DAYS_PER_7_DAYS = 5             # 8h x 5 days = 40h/week -> 2 rest days in any 7 days
MAX_OVERTIME_HOURS_PER_DAY = 4           # legal cap (the company cap above is stricter)
MAX_OVERTIME_HOURS_PER_7_DAYS = 18
OVERTIME_FIRST_HOUR_MULTIPLIER = 1.5
OVERTIME_LATER_HOURS_MULTIPLIER = 2.0

# Company policies -- hard rules
MAX_CONSECUTIVE_WORK_DAYS = 5
MAX_CONSECUTIVE_DAYS_OFF = 2
MIN_REST_HOURS = 8                       # between the end of one shift and the start of the next
MIDNIGHT_K = 24 - DAY_START_HOUR         # a shift ending after midnight is a night shift
                                         # (N always; E with overtime after) -> no M the next day
WEEKEND = (5, 6)                         # Saturday, Sunday: every agent is off on at least one

# Soft rules: penalty weights -- each stage minimizes its weighted sum
PENALTY = {
    "under_demand": 1000,    # per missing agent-hour -- the top priority
    "agents": 10000,         # per agent (stage 1, after under-demand): fewest agents before less overtime
    "overtime": 20,          # per overtime hour
    "overtime_balance": 10,  # per hour of the highest individual overtime total
    "shift_balance": 5,      # per shift an agent is away from the team average, for M, E and N
    "over_demand": 1,        # per surplus agent-hour -- only to stabilize the solution
}

# Pay -- reported, not optimized
HOURLY_WAGE = 31_000                     # Rp, ~ Jakarta minimum wage / 173

# =============================================================================
# 2. DEMAND
# =============================================================================

# Customers per hour on a typical day, by clock hour (identical every day of the month).
# 07:00-23:00 follows the Self-Order Terminal project's demand curve; overnight is assumed.
CUSTOMERS_PER_HOUR = {
    6: 10, 7: 15, 8: 20, 9: 21, 10: 28, 11: 58, 12: 87, 13: 67, 14: 33, 15: 23, 16: 30, 17: 49,
    18: 64, 19: 54, 20: 34, 21: 23, 22: 18, 23: 14, 0: 10, 1: 8, 2: 5, 3: 4, 4: 4, 5: 6,
}
MINUTES_PER_CUSTOMER = 3.5   # ~2.5 min order-taking + kitchen (SOT project) + ~1 min packing/serving
TARGET_UTILIZATION = 0.80    # share of time spent serving; the rest absorbs bursts of arrivals
SUPPORT_AGENTS = {h: (2 if 7 <= h <= 21 else 1) for h in range(24)}   # cleaning, restocking
MIN_CREW = 3                 # never fewer than 3 agents on duty


def agents_needed_per_hour():
    """Agents needed for each clock hour of a typical day (workload standard):
    ceil(customers x minutes / 60 / utilization) + support, at least MIN_CREW."""
    return {h: max(MIN_CREW, math.ceil(CUSTOMERS_PER_HOUR[h] * MINUTES_PER_CUSTOMER / 60 / TARGET_UTILIZATION)
                   + SUPPORT_AGENTS[h])
            for h in range(24)}


def planning_days():
    n = calendar.monthrange(YEAR, MONTH)[1]
    return [date(YEAR, MONTH, 1) + timedelta(days=i) for i in range(n)]


def demand_table():
    """One row per hour of the month (operating-day convention): day, clock hour, agents needed."""
    need = agents_needed_per_hour()
    rows = []
    for d, day in enumerate(planning_days()):
        for k in range(24):
            clock = (DAY_START_HOUR + k) % 24
            rows.append({"t": d * 24 + k, "day_index": d, "date": day, "weekday": day.strftime("%a"),
                         "clock_hour": clock, "required": need[clock]})
    return pd.DataFrame(rows)

# =============================================================================
# 3. SHIFT CODES
# =============================================================================


@dataclass(frozen=True)
class ShiftCode:
    """One concrete shift: base shift, break hour, and optional overtime Before (B) or After (A)."""
    shift: str
    break_clock: int
    ot_side: str = ""      # "", "B" or "A"
    ot_hours: int = 0

    @property
    def code(self):
        base = f"{self.shift}{self.break_clock:02d}"
        return base if not self.ot_side else f"{base}_{self.ot_side}{self.ot_hours}"

    @property
    def timing(self):
        """Shift and overtime without the break -- what the rest / rotation rules care about."""
        return self.shift, self.ot_side, self.ot_hours

    @property
    def start_k(self):
        """Start in hours from the operating day's 06:00 (negative = starts the evening before)."""
        return (SHIFTS[self.shift][1] - DAY_START_HOUR) % 24 - (self.ot_hours if self.ot_side == "B" else 0)

    @property
    def end_k(self):
        base = (SHIFTS[self.shift][1] - DAY_START_HOUR) % 24 + SHIFTS[self.shift][2]
        return base + (self.ot_hours if self.ot_side == "A" else 0)

    @property
    def break_k(self):
        return (self.break_clock - DAY_START_HOUR) % 24

    def working_k(self):
        """Hours on duty (break excluded), relative to the operating day's 06:00."""
        return [k for k in range(self.start_k, self.end_k) if k != self.break_k]

    @property
    def work_hours(self):
        return SHIFTS[self.shift][2] - 1 + self.ot_hours

    def clock(self):
        return f"{(DAY_START_HOUR + self.start_k) % 24:02d}:00-{(DAY_START_HOUR + self.end_k) % 24:02d}:00"


def shift_catalog():
    """All shift codes: 3 shifts x 3 break hours x (no overtime, B1, B2, A1, A2) = 45."""
    codes = []
    for s, (_, _, _, breaks) in SHIFTS.items():
        for b in breaks:
            codes.append(ShiftCode(s, b))
            for side in ("B", "A"):
                codes += [ShiftCode(s, b, side, h) for h in OVERTIME_OPTIONS]
    return codes


def overtime_pay_hours(ot_hours):
    """Overtime hours -> paid hours at the normal rate (1.5x first hour, 2x after)."""
    if ot_hours <= 0:
        return 0.0
    return OVERTIME_FIRST_HOUR_MULTIPLIER + (ot_hours - 1) * OVERTIME_LATER_HOURS_MULTIPLIER


def catalog_table():
    """Every shift code with its working time and break, for display."""
    return pd.DataFrame([{"code": c.code, "time": c.clock(),
                          "break": f"{c.break_clock:02d}:00-{(c.break_clock + 1) % 24:02d}:00",
                          "work_hours": c.work_hours, "overtime_hours": c.ot_hours} for c in shift_catalog()])


def _clash(prev, nxt):
    """True if `nxt` may NOT follow `prev` on the next day: rest too short, or night -> morning."""
    rest = 24 + nxt.start_k - prev.end_k
    return rest < MIN_REST_HOURS or (prev.end_k > MIDNIGHT_K and nxt.shift == "M")


def _date_key(day):
    return day.strftime("%Y%m%d")


def _solution_status(prob):
    """CBC's real outcome. PuLP reports 'Optimal' even when CBC stopped on the time limit,
    so read the solution status instead: optimal, feasible-but-unproven, or no solution."""
    return {1: "Optimal", 2: "Feasible (time limit)", 0: "No solution (time limit)",
            -1: "Infeasible", -2: "Unbounded", -3: "No solution"}.get(prob.sol_status, str(prob.sol_status))


def _solve(prob, time_limit, gap, msg, stage, raise_on_fail=True, warm_start=False):
    solver = pulp.PULP_CBC_CMD(msg=msg, timeLimit=time_limit, gapRel=gap, warmStart=warm_start,
                               keepFiles=warm_start)
    if warm_start:
        # On Windows, CBC only reads the starting solution from kept files; keep them in a
        # throw-away folder instead of the project folder.
        cwd = os.getcwd()
        with tempfile.TemporaryDirectory() as tmp:
            os.chdir(tmp)
            try:
                prob.solve(solver)
            finally:
                os.chdir(cwd)
    else:
        prob.solve(solver)
    status = _solution_status(prob)
    if raise_on_fail and status not in ("Optimal", "Feasible (time limit)"):
        raise RuntimeError(f"{stage}: no valid solution ({status}). Raise the time limit.")
    return status


def _add_coverage(prob, n, codes, under, over, demand, n_days):
    """sum of agents on duty + under - over = required, for every hour of the month.
    n[d, c] = number of agents working shift code c on day d (a variable or an expression)."""
    H = n_days * 24
    on_duty = {t: [] for t in range(H)}
    for (d, c), expr in n.items():
        for k in codes[c].working_k():
            if 0 <= d * 24 + k < H:
                on_duty[d * 24 + k].append(expr)
    required = demand.set_index("t")["required"]
    for t in range(H):
        prob += pulp.lpSum(on_duty[t]) + under[t] - over[t] == required[t]

# =============================================================================
# 4. STAGE 1 -- STAFFING PLAN
# =============================================================================


class StaffingPlan:
    """
    How many agents are needed, and how many work each shift code on each day.

    Variables
      n_<date>_<shift>[_<B|A><h>]  agents on that shift + overtime that day (integer), e.g. n_20270101_M_B1
      break_<date>_<shift><hour>   agents of that shift taking their break at that hour (integer)
      agents                       head count (integer)
      under[t], over[t]            missing / surplus agents in hour t
    Choosing the break separately from shift + overtime is equivalent to choosing full
    shift codes (any break can go with any overtime) but far smaller to solve; the counts
    are combined back into the 45 shift codes by code_counts().

    The per-agent day rules are applied in aggregate (e.g. any 7 days hold at most
    5 x agents shifts), so that the later stages can actually staff the plan.
    Solved in priority order: (1) least under-demand, (2) then the fewest agents,
    (3) then the least overtime and surplus. Putting agents before overtime matters:
    otherwise the model hires extra people just to avoid overtime (see staffing_frontier).
    """

    def __init__(self, demand, fixed_agents=None, max_agents=60, weekend_rule=True):
        self.demand = demand
        self.days = planning_days()
        self.D, self.H = len(self.days), len(self.days) * 24
        self.codes = shift_catalog()
        reps = {}
        for c in self.codes:
            reps.setdefault(c.timing, c)
        self.timings = list(reps.values())          # one representative code per shift + overtime
        T, D = range(len(self.timings)), range(self.D)
        p = self.prob = pulp.LpProblem("staffing_plan", pulp.LpMinimize)

        def timing_name(c):
            return c.shift if not c.ot_side else f"{c.shift}_{c.ot_side}{c.ot_hours}"

        self.n = {(d, i): pulp.LpVariable(f"n_{_date_key(self.days[d])}_{timing_name(self.timings[i])}",
                                          lowBound=0, cat="Integer") for d in D for i in T}
        self.brk = {(d, sh, b): pulp.LpVariable(f"break_{_date_key(self.days[d])}_{sh}{b:02d}", lowBound=0,
                                                cat="Integer")
                    for d in D for sh, (_, _, _, breaks) in SHIFTS.items() for b in breaks}
        self.agents = pulp.LpVariable("agents", lowBound=0, upBound=max_agents, cat="Integer")
        if fixed_agents is not None:
            p += self.agents == fixed_agents
        self.under = {t: pulp.LpVariable(f"under_{t}", lowBound=0) for t in range(self.H)}
        self.over = {t: pulp.LpVariable(f"over_{t}", lowBound=0) for t in range(self.H)}

        tm = self.timings
        self.shifts_on = {d: pulp.lpSum(self.n[d, i] for i in T) for d in D}
        ot_shifts = {d: pulp.lpSum(self.n[d, i] for i in T if tm[i].ot_hours) for d in D}
        ot_hours = {d: pulp.lpSum(tm[i].ot_hours * self.n[d, i] for i in T) for d in D}

        # Hourly coverage: agents whose shift spans the hour, minus those on break in it
        on_duty, on_break = {t: [] for t in range(self.H)}, {t: [] for t in range(self.H)}
        for (d, i), var in self.n.items():
            for k in range(tm[i].start_k, tm[i].end_k):
                if 0 <= d * 24 + k < self.H:
                    on_duty[d * 24 + k].append(var)
        for (d, sh, b), var in self.brk.items():
            t = d * 24 + (b - DAY_START_HOUR) % 24
            if t < self.H:
                on_break[t].append(var)
        for d in D:
            for sh, (_, _, _, breaks) in SHIFTS.items():
                p += pulp.lpSum(self.brk[d, sh, b] for b in breaks) ==                     pulp.lpSum(self.n[d, i] for i in T if tm[i].shift == sh)
        required = demand.set_index("t")["required"]
        for t in range(self.H):
            p += pulp.lpSum(on_duty[t]) - pulp.lpSum(on_break[t]) + self.under[t] - self.over[t] == required[t]

        # Per-agent day rules, in aggregate over the team
        A = self.agents
        for d in D:
            p += self.shifts_on[d] <= A
        for d in range(self.D - 6):
            p += pulp.lpSum(self.shifts_on[d + i] for i in range(7)) <= MAX_WORK_DAYS_PER_7_DAYS * A
            p += pulp.lpSum(ot_hours[d + i] for i in range(7)) <= MAX_OVERTIME_HOURS_PER_7_DAYS * A
        for d in range(self.D - MAX_CONSECUTIVE_DAYS_OFF):
            p += pulp.lpSum(self.shifts_on[d + i] for i in range(MAX_CONSECUTIVE_DAYS_OFF + 1)) >= A
        for d in range(self.D - 1):
            p += ot_shifts[d] + ot_shifts[d + 1] <= A
            night = pulp.lpSum(self.n[d, i] for i in T if tm[i].end_k > MIDNIGHT_K)
            morning = pulp.lpSum(self.n[d + 1, i] for i in T if tm[i].shift == "M")
            p += night + morning <= A
        for d, day in enumerate(self.days):
            if weekend_rule and day.weekday() == WEEKEND[0] and d + 1 < self.D:
                p += self.shifts_on[d] + self.shifts_on[d + 1] <= A

        self.terms = {
            "under_demand": pulp.lpSum(self.under.values()),
            "agents": self.agents,
            "overtime": pulp.lpSum(ot_hours.values()),
            "over_demand": pulp.lpSum(self.over.values()),
        }

    def solve(self, time_limit=120, msg=False):
        self.prob.setObjective(self.terms["under_demand"])
        _solve(self.prob, time_limit, 0.0, msg, "Stage 1 (demand)")
        self.prob += self.terms["under_demand"] <= pulp.value(self.terms["under_demand"]) + 1e-6
        self.prob.setObjective(PENALTY["agents"] * self.terms["agents"] +
                               PENALTY["overtime"] * self.terms["overtime"] +
                               PENALTY["over_demand"] * self.terms["over_demand"])
        self.status = _solve(self.prob, time_limit, 0.0, msg, "Stage 1 (agents)")
        return self.status

    @property
    def n_agents(self):
        return int(round(self.agents.value()))

    def agents_per_day(self):
        return {d: int(round(pulp.value(self.shifts_on[d]))) for d in range(self.D)}

    def code_counts(self):
        """Planned agents per (day, shift code index): break counts paired with shift + overtime counts."""
        index = {c.code: i for i, c in enumerate(self.codes)}
        counts = {}
        for d in range(self.D):
            for sh, (_, _, _, breaks) in SHIFTS.items():
                timing_slots = [tmg for i, tmg in enumerate(self.timings) if tmg.shift == sh
                                for _ in range(int(round(self.n[d, i].value())))]
                break_slots = [b for b in breaks for _ in range(int(round(self.brk[d, sh, b].value())))]
                for tmg, b in zip(timing_slots, break_slots):
                    key = (d, index[ShiftCode(sh, b, tmg.ot_side, tmg.ot_hours).code])
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def plan_table(self):
        """Day x shift code counts (only codes used)."""
        return pd.DataFrame([{"day_index": d, "date": self.days[d], "code": self.codes[c].code, "count": n}
                             for (d, c), n in sorted(self.code_counts().items())])


def rule_impact(demand, time_limit=90):
    """Stage 1 with and without the weekend-off rule: how much head count the rule costs."""
    rows = []
    for label, rule in [("All rules", True), ("Without the weekend-off rule", False)]:
        plan = StaffingPlan(demand, weekend_rule=rule)
        plan.solve(time_limit=time_limit)
        rows.append({"scenario": label, "agents": plan.n_agents,
                     "shifts_per_month": sum(plan.agents_per_day().values()),
                     "short_hours": round(pulp.value(plan.terms["under_demand"])),
                     "overtime_hours": round(pulp.value(plan.terms["overtime"])),
                     "surplus_hours": round(pulp.value(plan.terms["over_demand"]))})
    return pd.DataFrame(rows).set_index("scenario")


def staffing_frontier(demand, agent_counts, time_limit=60):
    """Stage 1 with the head count fixed: shortage, overtime and surplus for each head count."""
    rows = []
    for a in agent_counts:
        plan = StaffingPlan(demand, fixed_agents=a)
        plan.prob.setObjective(PENALTY["under_demand"] * plan.terms["under_demand"] +
                               PENALTY["overtime"] * plan.terms["overtime"] +
                               PENALTY["over_demand"] * plan.terms["over_demand"])
        status = _solve(plan.prob, time_limit, 0.0, False, "frontier", raise_on_fail=False)
        rows.append({"agents": a, "status": status,
                     "short_hours": round(pulp.value(plan.terms["under_demand"]) or 0),
                     "overtime_hours": round(pulp.value(plan.terms["overtime"]) or 0),
                     "surplus_hours": round(pulp.value(plan.terms["over_demand"]) or 0)})
    return pd.DataFrame(rows)

# =============================================================================
# 5. STAGE 2 -- WORK / DAY-OFF PATTERN
# =============================================================================


class DayPattern:
    """
    Which days each agent works: Agent01_20270101_work = 1 if Agent01 works on 1 January.
    Hard: max 5 work days in any 7, max 5 in a row, max 2 days off in a row,
    no DO-Work-DO, at least one of Saturday / Sunday off.
    Soft: match the planned number of agents per day; keep days worked even across agents.
    """

    def __init__(self, plan):
        self.plan, self.days, self.D = plan, plan.days, plan.D
        self.agent_ids = [f"Agent{i + 1:02d}" for i in range(plan.n_agents)]
        A, D = range(len(self.agent_ids)), range(self.D)
        p = self.prob = pulp.LpProblem("day_pattern", pulp.LpMinimize)
        w = self.w = {(a, d): pulp.LpVariable(f"{self.agent_ids[a]}_{_date_key(self.days[d])}_work", cat="Binary")
                      for a in A for d in D}

        for a in A:
            for d in range(self.D - 6):
                p += pulp.lpSum(w[a, d + i] for i in range(7)) <= MAX_WORK_DAYS_PER_7_DAYS
            for d in range(self.D - MAX_CONSECUTIVE_WORK_DAYS):
                p += pulp.lpSum(w[a, d + i] for i in range(MAX_CONSECUTIVE_WORK_DAYS + 1)) <= MAX_CONSECUTIVE_WORK_DAYS
            for d in range(self.D - MAX_CONSECUTIVE_DAYS_OFF):
                p += pulp.lpSum(w[a, d + i] for i in range(MAX_CONSECUTIVE_DAYS_OFF + 1)) >= 1
            for d in range(1, self.D - 1):
                p += w[a, d] <= w[a, d - 1] + w[a, d + 1]
            for d, day in enumerate(self.days):
                if day.weekday() == WEEKEND[0] and d + 1 < self.D:
                    p += w[a, d] + w[a, d + 1] <= 1

        target = plan.agents_per_day()
        self.missing = {d: pulp.LpVariable(f"missing_{_date_key(self.days[d])}", lowBound=0) for d in D}
        self.extra = {d: pulp.LpVariable(f"extra_{_date_key(self.days[d])}", lowBound=0) for d in D}
        for d in D:
            p += pulp.lpSum(w[a, d] for a in A) + self.missing[d] - self.extra[d] == target[d]
        hi, lo = pulp.LpVariable("most_days", lowBound=0), pulp.LpVariable("fewest_days", lowBound=0)
        for a in A:
            p += pulp.lpSum(w[a, d] for d in D) <= hi
            p += pulp.lpSum(w[a, d] for d in D) >= lo
        for a in range(len(A) - 1):   # symmetry breaking: agents are interchangeable
            p += pulp.lpSum(w[a, d] for d in D) >= pulp.lpSum(w[a + 1, d] for d in D)

        work_hours = SHIFTS["M"][2] - 1
        self.terms = {
            "missing_agent_days": pulp.lpSum(self.missing.values()),
            "extra_agent_days": pulp.lpSum(self.extra.values()),
            "days_spread": hi - lo,
        }
        p += (PENALTY["under_demand"] * work_hours * self.terms["missing_agent_days"] +
              PENALTY["over_demand"] * work_hours * self.terms["extra_agent_days"] +
              PENALTY["shift_balance"] * self.terms["days_spread"])

    def solve(self, time_limit=120, msg=False):
        self.status = _solve(self.prob, time_limit, 0.0, msg, "Stage 2 (day pattern)")
        return self.status

    def works(self, a, d):
        return self.w[a, d].value() > 0.5

# =============================================================================
# 6. STAGE 3 -- SHIFT ASSIGNMENT
# =============================================================================


class ShiftAssignment:
    """
    For every day an agent works (from stage 2), exactly one shift code:
      Agent01_20270101_M10_B1 = 1  ->  Agent01 works M10_B1 on 1 January 2027.

    Hard: one shift code per work day; rest and no night -> morning between consecutive
    days; no overtime on consecutive days; max 18h overtime in any 7 days.
    Soft (PENALTY): under-demand, overtime, shift balance (each agent's M / E / N count
    close to the team average), overtime balance, over-demand.
    """

    def __init__(self, pattern, demand):
        self.pattern, self.days, self.D = pattern, pattern.days, pattern.D
        self.H = self.D * 24
        self.agent_ids = pattern.agent_ids
        self.codes = shift_catalog()
        A, D, C = range(len(self.agent_ids)), range(self.D), range(len(self.codes))
        p = self.prob = pulp.LpProblem("shift_assignment", pulp.LpMinimize)
        self.worked = {(a, d) for a in A for d in D if pattern.works(a, d)}
        self.x = {(a, d, c): pulp.LpVariable(
                      f"{self.agent_ids[a]}_{_date_key(self.days[d])}_{self.codes[c].code}", cat="Binary")
                  for (a, d) in self.worked for c in C}
        self.under = {t: pulp.LpVariable(f"under_{t}", lowBound=0) for t in range(self.H)}
        self.over = {t: pulp.LpVariable(f"over_{t}", lowBound=0) for t in range(self.H)}

        # Codes grouped by timing (shift + overtime): the break hour doesn't affect rest rules
        timings = {}
        for c in C:
            timings.setdefault(self.codes[c].timing, []).append(c)
        rep = {tm: self.codes[cs[0]] for tm, cs in timings.items()}

        def picked(a, d, cs):
            return pulp.lpSum(self.x[a, d, c] for c in cs) if (a, d) in self.worked else 0

        def ot_hours(a, d):
            return pulp.lpSum(self.codes[c].ot_hours * self.x[a, d, c] for c in C) if (a, d) in self.worked else 0

        ot_codes = [c for c in C if self.codes[c].ot_hours]
        for (a, d) in self.worked:
            p += picked(a, d, C) == 1
            if (a, d + 1) in self.worked:
                p += picked(a, d, ot_codes) + picked(a, d + 1, ot_codes) <= 1
                for tm, cs in timings.items():
                    clash = [c for tn, cn in timings.items() if _clash(rep[tm], rep[tn]) for c in cn]
                    if clash:
                        p += picked(a, d, cs) + picked(a, d + 1, clash) <= 1
        for a in A:
            for d in range(self.D - 6):
                p += pulp.lpSum(ot_hours(a, d + i) for i in range(7)) <= MAX_OVERTIME_HOURS_PER_7_DAYS

        n = {(d, c): pulp.lpSum(self.x[a, d, c] for a in A if (a, d) in self.worked) for d in D for c in C}
        _add_coverage(p, n, self.codes, self.under, self.over, demand, self.D)

        # Shift balance: each agent's number of M, E and N close to the team average for that shift
        devs, n_agents = [], len(self.agent_ids)
        for s in SHIFTS:
            s_codes = [c for c in C if self.codes[c].shift == s]
            count = {a: pulp.lpSum(picked(a, d, s_codes) for d in D) for a in A}
            team_total = pulp.lpSum(count.values())
            for a in A:
                dev = pulp.LpVariable(f"{self.agent_ids[a]}_{s}_deviation", lowBound=0)
                p += n_agents * dev >= n_agents * count[a] - team_total
                p += n_agents * dev >= team_total - n_agents * count[a]
                devs.append(dev)
        # Overtime balance: the highest individual monthly overtime total
        self.max_ot = pulp.LpVariable("max_overtime_per_agent", lowBound=0)
        for a in A:
            p += pulp.lpSum(ot_hours(a, d) for d in D) <= self.max_ot

        self.terms = {
            "under_demand": pulp.lpSum(self.under.values()),
            "overtime": pulp.lpSum(ot_hours(a, d) for (a, d) in self.worked),
            "overtime_balance": self.max_ot,
            "shift_balance": pulp.lpSum(devs),
            "over_demand": pulp.lpSum(self.over.values()),
        }
        p += pulp.lpSum(PENALTY[k] * v for k, v in self.terms.items())

    def warm_start(self):
        """
        A valid starting roster for the solver, built greedily day by day: each working
        agent takes a planned shift code (from stage 1) that is allowed after what they
        worked the day before, else any allowed code without overtime. CBC then only has
        to improve this roster instead of searching for a first valid one.
        """
        plan_counts = self.pattern.plan.code_counts()
        no_ot = [c for c in range(len(self.codes)) if not self.codes[c].ot_hours]
        chosen = {}
        shift_count = {(a, sh): 0 for a in range(len(self.agent_ids)) for sh in SHIFTS}
        ot_total = {a: 0 for a in range(len(self.agent_ids))}
        for d in range(self.D):
            todo = [c for (dd, c), n in sorted(plan_counts.items()) if dd == d for _ in range(n)]
            workers = [a for a in range(len(self.agent_ids)) if (a, d) in self.worked]

            def allowed(a, c):
                prev = chosen.get((a, d - 1))
                if prev is None:
                    return True
                p, q = self.codes[prev], self.codes[c]
                if _clash(p, q) or (p.ot_hours and q.ot_hours):
                    return False
                week_ot = sum(self.codes[chosen[a, d - i]].ot_hours for i in range(1, 7) if (a, d - i) in chosen)
                return week_ot + q.ot_hours <= MAX_OVERTIME_HOURS_PER_7_DAYS

            workers.sort(key=lambda a: (sum(allowed(a, c) for c in set(todo)), a))   # most constrained first
            for a in workers:
                # Fairness: prefer the shift this agent has worked least, then spread overtime
                options = sorted((c for c in set(todo) if allowed(a, c)),
                                 key=lambda c: (shift_count[a, self.codes[c].shift],
                                                self.codes[c].ot_hours * ot_total[a], c))
                if options:
                    pick = options[0]
                    todo.remove(pick)
                else:
                    pick = min((c for c in no_ot if allowed(a, c)),
                               key=lambda c: (shift_count[a, self.codes[c].shift], c))
                chosen[a, d] = pick
                shift_count[a, self.codes[pick].shift] += 1
                ot_total[a] += self.codes[pick].ot_hours
        for (a, d, c), var in self.x.items():
            var.setInitialValue(1 if chosen[a, d] == c else 0)
        return chosen

    def solve(self, time_limit=300, gap=0.01, msg=False):
        self.warm_start()
        self.status = _solve(self.prob, time_limit, gap, msg, "Stage 3 (shifts)", warm_start=True)
        return self.status

    def chosen_variables(self):
        """Names of the decision variables set to 1, e.g. 'Agent01_20270101_M10_B1'."""
        return sorted(v.name for v in self.x.values() if v.value() and v.value() > 0.5)


def build_roster(demand, time_limits=(120, 120, 600), log=print):
    """Run the three stages. Returns (plan, pattern, roster)."""
    plan = StaffingPlan(demand)
    plan.solve(time_limit=time_limits[0])
    log(f"Stage 1 - staffing plan: {plan.status}, {plan.n_agents} agents")
    pattern = DayPattern(plan)
    pattern.solve(time_limit=time_limits[1])
    log(f"Stage 2 - day pattern:   {pattern.status}, "
        f"{pulp.value(pattern.terms['missing_agent_days']):.0f} planned agent-days not staffed")
    roster = ShiftAssignment(pattern, demand)
    roster.solve(time_limit=time_limits[2])
    log(f"Stage 3 - shift codes:   {roster.status}, "
        f"{pulp.value(roster.terms['under_demand']):.0f} agent-hours short")
    return plan, pattern, roster

# =============================================================================
# 7. RESULTS
# =============================================================================


def roster_long(roster):
    """One row per worked shift: agent, date, shift code, times, hours and pay."""
    rows = []
    for (a, d, c), v in roster.x.items():
        if not (v.value() and v.value() > 0.5):
            continue
        sc = roster.codes[c]
        rows.append({
            "variable": v.name, "agent_id": roster.agent_ids[a], "day_index": d, "date": roster.days[d],
            "weekday": roster.days[d].strftime("%a"), "shift": sc.shift, "code": sc.code, "time": sc.clock(),
            "break": f"{sc.break_clock:02d}:00-{(sc.break_clock + 1) % 24:02d}:00",
            "work_hours": sc.work_hours, "overtime_hours": sc.ot_hours,
            "pay": HOURLY_WAGE * (sc.work_hours - sc.ot_hours + overtime_pay_hours(sc.ot_hours)),
            "_start_k": sc.start_k, "_end_k": sc.end_k, "_break_k": sc.break_k,
        })
    return pd.DataFrame(rows).sort_values(["agent_id", "day_index"]).reset_index(drop=True)


def day_labels(days):
    """Column headers for date-based tables: the date itself, e.g. '2027-01-01'."""
    return [d.isoformat() for d in days]


def roster_pivot(long, roster):
    """The roster as a pivot table: rows = agents, columns = dates, cells = shift code
    (e.g. 'M10', 'E18_A2'), 'DO' = day off."""
    pv = long.pivot_table(index="agent_id", columns="day_index", values="code", aggfunc="first")
    pv = pv.reindex(index=roster.agent_ids, columns=range(roster.D)).fillna("DO")
    pv.columns = day_labels(roster.days)
    return pv


def style_roster(pivot):
    """Colour the roster pivot by shift (for display in the notebook); bold = overtime."""
    colors = {"DO": "#F2F4F7", "M": "#FBE3A6", "E": "#BBD3F0", "N": "#14324B"}

    def cell(v):
        key = "DO" if v == "DO" else v[0]
        fg = "white" if key == "N" else "#1F2933"
        weight = "bold" if "_" in v else "normal"
        return f"background-color: {colors[key]}; color: {fg}; font-weight: {weight}; font-size: 9px"

    return pivot.style.map(cell)


def daily_shift_counts(long, days):
    """Shift x date: how many agents work M, E, N each day."""
    pv = long.pivot_table(index="shift", columns="day_index", values="agent_id", aggfunc="count", fill_value=0)
    pv = pv.reindex(index=list(SHIFTS), columns=range(len(days)), fill_value=0)
    pv.columns = day_labels(days)
    return pv


def coverage_table(long, demand, days):
    """Per hour of the month: agents required vs on duty (breaks excluded)."""
    H = len(days) * 24
    staffed = np.zeros(H, dtype=int)
    for _, r in long.iterrows():
        for k in range(r["_start_k"], r["_end_k"]):
            t = r["day_index"] * 24 + k
            if 0 <= t < H and k != r["_break_k"]:
                staffed[t] += 1
    cov = demand.copy()
    cov["on_duty"] = staffed
    cov["gap"] = cov["on_duty"] - cov["required"]
    return cov


def penalty_breakdown(plan, pattern, roster):
    """Every penalty term of every stage."""
    rows = [("1 Staffing plan", k, pulp.value(v) or 0.0) for k, v in plan.terms.items()]
    rows += [("2 Day pattern", k, pulp.value(v) or 0.0) for k, v in pattern.terms.items()]
    rows += [("3 Shift codes", k, pulp.value(v) or 0.0) for k, v in roster.terms.items()]
    return pd.DataFrame(rows, columns=["stage", "term", "amount"])


def agent_summary(long):
    """Per agent: days worked, M / E / N mix, overtime and pay."""
    mix = long.pivot_table(index="agent_id", columns="shift", values="code", aggfunc="count", fill_value=0)
    totals = long.groupby("agent_id").agg(days=("code", "size"), work_hours=("work_hours", "sum"),
                                          overtime_hours=("overtime_hours", "sum"), pay=("pay", "sum"))
    return totals.join(mix.reindex(columns=list(SHIFTS), fill_value=0))


def _runs(row, value):
    best = cur = 0
    for v in row:
        cur = cur + 1 if v == value else 0
        best = max(best, cur)
    return best


def compliance_checks(long, roster):
    """Re-check every hard rule from the finished roster, independently of the solver."""
    pv = roster_pivot(long, roster)
    work = (pv != "DO").astype(int).to_numpy()
    ot = long.pivot_table(index="agent_id", columns="day_index", values="overtime_hours", aggfunc="sum") \
        .reindex(index=pv.index, columns=range(roster.D)).fillna(0).to_numpy()
    D = roster.D
    by_day = long.set_index(["agent_id", "day_index"])
    night_to_morning = short_rest = 0
    for (agent, d), r in by_day.iterrows():
        if (agent, d + 1) in by_day.index:
            nxt = by_day.loc[(agent, d + 1)]
            short_rest += (24 + nxt["_start_k"] - r["_end_k"]) < MIN_REST_HOURS
            night_to_morning += (r["_end_k"] > MIDNIGHT_K) and nxt["shift"] == "M"
    sat = [d for d, day in enumerate(roster.days) if day.weekday() == WEEKEND[0] and d + 1 < D]
    checks = [
        ("Max work days in any 7 days", max(work[:, d:d + 7].sum(1).max() for d in range(D - 6)),
         MAX_WORK_DAYS_PER_7_DAYS),
        ("Longest run of work days", max(_runs(r, 1) for r in work), MAX_CONSECUTIVE_WORK_DAYS),
        ("Longest run of days off", max(_runs(r, 0) for r in work), MAX_CONSECUTIVE_DAYS_OFF),
        ("Max overtime hours in one shift", int(long["overtime_hours"].max()), max(OVERTIME_OPTIONS)),
        ("Max overtime hours in any 7 days", int(max(ot[:, d:d + 7].sum(1).max() for d in range(D - 6))),
         MAX_OVERTIME_HOURS_PER_7_DAYS),
        ("Overtime on two consecutive days", sum(1 for r in ot for d in range(D - 1) if r[d] and r[d + 1]), 0),
        ("DO-Work-DO patterns", sum(1 for r in work for d in range(1, D - 1)
                                    if r[d] and not r[d - 1] and not r[d + 1]), 0),
        ("Agents working both Sat and Sun", sum(1 for d in sat for r in work if r[d] and r[d + 1]), 0),
        ("Night -> morning rotations", int(night_to_morning), 0),
        (f"Rest under {MIN_REST_HOURS}h between shifts", int(short_rest), 0),
    ]
    df = pd.DataFrame(checks, columns=["Hard rule", "Result", "Limit"])
    df["OK"] = df["Result"] <= df["Limit"]
    return df

# =============================================================================
# 8. CHARTS
# =============================================================================

NAVY, GOLD, GREY, RED = "#14324B", "#C9772F", "#9AA5B1", "#C0392B"
SHIFT_COLORS = {"DO": "#F2F4F7", "M": "#F6C85F", "E": "#3A7DC9", "N": NAVY}
plt.rcParams.update({"font.size": 11, "axes.titlesize": 14, "axes.labelsize": 12})


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def plot_hourly_demand(path):
    """Agents needed per hour (bars), customers per hour (line), and when each shift runs."""
    need = agents_needed_per_hour()
    hours = [(DAY_START_HOUR + k) % 24 for k in range(24)]
    fig, ax1 = plt.subplots(figsize=(12, 5))
    ax1.bar(range(24), [need[h] for h in hours], color=NAVY, alpha=0.85, label="Agents needed")
    ax1.set_xticks(range(24), [f"{h:02d}" for h in hours])
    ax1.set_xlabel("Clock hour (operating day starts 06:00)")
    ax1.set_ylabel("Agents needed")
    ax2 = ax1.twinx()
    ax2.plot(range(24), [CUSTOMERS_PER_HOUR[h] for h in hours], color=GOLD, marker="o", label="Customers / hour")
    ax2.set_ylabel("Customers per hour")
    for i, (name, (_, start, length, _)) in enumerate(SHIFTS.items()):
        k0 = (start - DAY_START_HOUR) % 24
        y = -1.0 - 0.9 * i
        ax1.plot([k0 - 0.5, min(k0 + length, 24) - 0.5], [y, y], color=SHIFT_COLORS[name], lw=7, solid_capstyle="butt")
        ax1.text(k0 - 0.45, y, f" {name} ", va="center", fontsize=10, fontweight="bold",
                 color="white" if name != "M" else "#1F2933")
    ax1.set_ylim(-3.6, max(need.values()) + 2)
    ax1.set_title("Hourly Demand on a Typical Day, and When Each Shift Works")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper left")
    return _save(fig, path)


def plot_roster(long, roster, path):
    """Agent x date roster coloured by shift; red dot = overtime that day."""
    pv = roster_pivot(long, roster)
    codes = {"DO": 0, "M": 1, "E": 2, "N": 3}
    grid = pv.apply(lambda col: col.map(lambda v: codes["DO" if v == "DO" else v[0]])).to_numpy()
    fig, ax = plt.subplots(figsize=(16, max(5, 0.34 * len(pv))))
    ax.imshow(grid, aspect="auto", cmap=ListedColormap([SHIFT_COLORS[k] for k in codes]), vmin=0, vmax=3)
    yy, xx = np.nonzero(pv.apply(lambda col: col.str.contains("_")).to_numpy())
    ax.scatter(xx, yy, s=16, color=RED, zorder=3)
    for d, day in enumerate(roster.days):
        if day.weekday() == WEEKEND[0]:
            ax.axvspan(d - 0.5, d + 1.5, facecolor="none", edgecolor=GOLD, lw=1.2, zorder=4)
    ax.set_xticks(range(roster.D), [f"{d.day}\n{d.strftime('%a')[:2]}" for d in roster.days], fontsize=8)
    ax.set_yticks(range(len(pv)), pv.index, fontsize=9)
    ax.set_title(f"Roster, {calendar.month_name[MONTH]} {YEAR} (weekends outlined)")
    ax.legend(handles=[Patch(color=SHIFT_COLORS[k], label="Day off" if k == "DO" else k) for k in codes]
              + [plt.Line2D([], [], marker="o", color=RED, linestyle="", label="Overtime")],
              loc="upper center", bbox_to_anchor=(0.5, -0.07), ncol=5, frameon=False)
    return _save(fig, path)


def plot_coverage_heatmap(coverage, days, path):
    """Clock hour x date heatmap of on-duty minus required (red = short, blue = surplus)."""
    pv = coverage.pivot_table(index="clock_hour", columns="day_index", values="gap", aggfunc="sum")
    pv = pv.reindex([(DAY_START_HOUR + k) % 24 for k in range(24)])
    pv.columns = [d.day for d in days]
    lim = max(1, int(np.abs(pv.to_numpy()).max()))
    fig, ax = plt.subplots(figsize=(16, 6.5))
    sns.heatmap(pv, cmap="RdBu", center=0, vmin=-lim, vmax=lim, linewidths=0.3, linecolor="white",
                annot=True, fmt="d", annot_kws={"fontsize": 7}, cbar_kws={"label": "On duty minus required"}, ax=ax)
    ax.set_xlabel("Day of month")
    ax.set_ylabel("Clock hour")
    ax.set_title("Coverage Gap by Hour (red = short, blue = surplus, white = exact)")
    ax.tick_params(axis="y", rotation=0)
    return _save(fig, path)


def plot_day_coverage(coverage, day_index, path):
    """Required vs on-duty agents through one operating day."""
    day = coverage[coverage["day_index"] == day_index]
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(24)
    ax.bar(x, day["on_duty"], color=NAVY, alpha=0.8, label="On duty (excl. breaks)")
    ax.step(x, day["required"], where="mid", color=GOLD, lw=2.5, label="Required")
    ax.set_xticks(x, [f"{h:02d}" for h in day["clock_hour"]])
    ax.set_xlabel("Clock hour")
    ax.set_ylabel("Agents")
    first = day.iloc[0]
    ax.set_title(f"One Day of Coverage: {first['weekday']} {first['date'].day} {calendar.month_name[MONTH]}")
    ax.legend(loc="upper left")
    return _save(fig, path)


def plot_agent_balance(summary, path):
    """Each agent's M / E / N mix, and overtime hours."""
    s = summary.sort_index()
    fig, axes = plt.subplots(1, 2, figsize=(15, max(5, 0.3 * len(s))), gridspec_kw={"width_ratios": [3, 1]})
    left = np.zeros(len(s))
    for k in SHIFTS:
        axes[0].barh(s.index, s[k], left=left, color=SHIFT_COLORS[k], label=k)
        left += s[k].to_numpy()
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Shifts worked this month")
    axes[0].set_title("Shift Mix per Agent")
    axes[0].legend(ncol=3, loc="lower right")
    axes[1].barh(s.index, s["overtime_hours"], color=RED)
    axes[1].invert_yaxis()
    axes[1].set_yticks([])
    axes[1].set_xlabel("Overtime hours")
    axes[1].set_title("Overtime per Agent")
    return _save(fig, path)
