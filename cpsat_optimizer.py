"""Exact-style cutting optimiser using Google OR-Tools (CP-SAT).

How it works (per diameter):
  1. Group the required pieces by length (about 15-130 different lengths instead of ~38,000 pieces).
  2. A *pattern* is one way to fill a stock bar, e.g. "6575 + 5232".  There are far too many patterns to list,
     so we generate only the useful ones (column generation):
       a. Solve a relaxed version (LP) that may use fractions of bars. It tells us how "valuable" each length is.
       b. A small knapsack search finds the pattern that fills a bar best given those values.
       c. Add that pattern and repeat until no better pattern exists.
     The relaxed answer is also a *lower bound*: no plan can use fewer bars than it.
  3. CP-SAT then picks a whole number of bars for each pattern so every piece is covered, using the fewest bars
     (a short second pass then avoids making surplus pieces).
  4. The Best Fit Decreasing plan is given to CP-SAT as a starting point and as the fallback, so the result is
     never worse than it.
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from ortools.sat.python import cp_model

from optimizer import OptResult, _pack, build_result, plan_row, split_oversize


@dataclass
class CpsatResult:
    opt: OptResult
    stats: pd.DataFrame      # one row per diameter: bars, lower bound, status, time ...


# ---------------------------------------------------------------- pricing (knapsack)
def _best_pattern(sizes: np.ndarray, values: np.ndarray, cap: int):
    """Unbounded knapsack: choose counts a_i >= 0 with sum(size_i * a_i) <= cap maximising sum(value_i * a_i).
    Returns (best value, counts array)."""
    dp = np.zeros(cap + 1)
    choice = np.full(cap + 1, -1, dtype=np.int32)
    for i in range(len(sizes)):
        w, v = int(sizes[i]), float(values[i])
        if v <= 1e-9 or w > cap:
            continue
        for k in range(w, cap + 1, w):
            hi = min(k + w, cap + 1)
            cand = dp[k - w:hi - w] + v
            seg = dp[k:hi]
            better = cand > seg + 1e-12
            if better.any():
                dp[k:hi] = np.where(better, cand, seg)
                choice[k:hi] = np.where(better, i, choice[k:hi])
    counts = np.zeros(len(sizes), dtype=np.int64)
    c = cap
    while c > 0 and choice[c] >= 0:
        i = choice[c]
        counts[i] += 1
        c -= int(sizes[i])
    return float(dp[cap]), counts


# ---------------------------------------------------------------- one diameter
def _solve_diameter(lengths: list[int], demand: list[int], stock_mm: int, kerf_mm: int,
                    bfd_patterns: list[tuple[int, ...]], time_limit_s: float, log=None):
    """Returns (patterns, copies, lp_bound, cg_converged, cg_iterations, cp_status, cp_seconds).
    patterns[p] is a tuple of counts aligned with `lengths`; copies[p] how many bars use it."""
    n = len(lengths)
    sizes = np.array([l + kerf_mm for l in lengths], dtype=np.int64)
    cap = stock_mm + kerf_mm
    d = np.array(demand, dtype=np.int64)

    patterns: dict[tuple[int, ...], int] = {}

    def add(counts) -> bool:
        key = tuple(int(x) for x in counts)
        if key in patterns or sum(key) == 0:
            return False
        patterns[key] = len(patterns)
        return True

    # starting columns: one-length-only bars, and the Best Fit Decreasing bars
    for i in range(n):
        c = [0] * n
        c[i] = max(1, int(cap // sizes[i]))
        add(c)
    for p in bfd_patterns:
        add(p)

    # ---- column generation on the LP relaxation ----
    solver = pywraplp.Solver.CreateSolver("GLOP")
    cons = [solver.Constraint(float(d[i]), solver.infinity()) for i in range(n)]
    objective = solver.Objective()
    objective.SetMinimization()
    xvars = []

    def add_column(key):
        x = solver.NumVar(0.0, solver.infinity(), "")
        objective.SetCoefficient(x, 1.0)
        for i, a in enumerate(key):
            if a:
                cons[i].SetCoefficient(x, float(a))
        xvars.append(x)

    for key in list(patterns):
        add_column(key)

    cg_budget = time_limit_s * 0.5
    t0 = time.time()
    iterations = 0
    converged = False
    lp_value = 0.0
    while True:
        solver.Solve()
        lp_value = objective.Value()
        duals = np.array([c.dual_value() for c in cons])
        value, counts = _best_pattern(sizes, duals, cap)
        iterations += 1
        if value <= 1.0 + 1e-7:
            converged = True
            break
        if not add(counts):
            converged = True       # the best pattern is already in the model
            break
        add_column(tuple(int(x) for x in counts))
        if time.time() - t0 > cg_budget:
            break
        if log and iterations % 25 == 0:
            log(f"  pattern search: {iterations} rounds, {len(patterns)} patterns, LP bound {lp_value:,.1f} bars")

    keys = list(patterns)

    # ---- integer step with CP-SAT ----
    model = cp_model.CpModel()
    n_p = []
    for key in keys:
        ub = max(math.ceil(d[i] / key[i]) for i in range(n) if key[i]) if any(key) else 0
        n_p.append(model.NewIntVar(0, int(ub), ""))
    made = []                                   # pieces produced of each length
    for i in range(n):
        made.append(sum(key[i] * n_p[p] for p, key in enumerate(keys) if key[i]))
        model.Add(made[i] >= int(d[i]))         # at least the required number of pieces (surplus is allowed)
    total_bars = sum(n_p)
    model.Minimize(total_bars)                  # stage 1: the fewest bars (scrap length/weight depend only on this)

    hint = Counter(tuple(int(x) for x in p) for p in bfd_patterns)
    if hint:
        for p, key in enumerate(keys):
            model.AddHint(n_p[p], hint.get(key, 0))

    def solver(seconds):
        cp = cp_model.CpSolver()
        cp.parameters.max_time_in_seconds = seconds
        cp.parameters.num_workers = max(1, min(8, os.cpu_count() or 1))
        return cp

    cp = solver(max(5.0, time_limit_s - (time.time() - t0)))
    t1 = time.time()
    status = cp.Solve(model)
    status_name = cp.StatusName(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return keys, None, lp_value, converged, iterations, status_name, time.time() - t1
    best_bars = int(round(cp.ObjectiveValue()))
    copies = [int(cp.Value(v)) for v in n_p]

    # stage 2 (tidy-up, short): keep that many bars, but prefer plans that make no surplus pieces
    model.Add(total_bars == best_bars)
    model.Minimize(sum(int(sizes[i]) * made[i] for i in range(n)))
    model.ClearHints()
    for p in range(len(keys)):
        model.AddHint(n_p[p], copies[p])
    cp2 = solver(min(3.0, max(1.0, 0.2 * (time_limit_s - (time.time() - t0)))))
    if cp2.Solve(model) in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        copies = [int(cp2.Value(v)) for v in n_p]
    return keys, copies, lp_value, converged, iterations, status_name, time.time() - t1


# ---------------------------------------------------------------- whole job
def optimize_cpsat(bars: pd.DataFrame, stock_mm: int = 12000, kerf_mm: int = 0,
                   time_limit_s: float = 30.0, progress=None) -> CpsatResult:
    """Same inputs/outputs as optimizer.optimize(), solved with CP-SAT. `progress(msg)` is optional."""
    return optimize_patterns(bars, stock_mm, kerf_mm, time_limit_s, progress)


def optimize_patterns(bars: pd.DataFrame, stock_mm: int = 12000, kerf_mm: int = 0,
                      time_limit_s: float = 30.0, progress=None) -> CpsatResult:
    """Pattern-based optimiser (column generation + CP-SAT integer step)."""
    log = progress or (lambda m: None)
    oversize, ok = split_oversize(bars, stock_mm)

    plan_rows: list[dict] = []
    stat_rows: list[dict] = []
    bar_no = 0

    for dia, g in ok.groupby("dia"):
        dia = int(dia)
        # marks grouped by length: queue of marks to hand out for each length
        queue: dict[int, list[int]] = defaultdict(list)
        for r in g.itertuples():
            queue[int(r.length_mm)].extend([int(r.mark)] * int(r.qty))
        lengths = sorted(queue)
        demand = [len(queue[l]) for l in lengths]
        idx = {l: i for i, l in enumerate(lengths)}
        log(f"Ø{dia}: {sum(demand):,} pieces in {len(lengths)} different lengths")

        # Best Fit Decreasing as the starting point and as the fallback
        pieces = [(l, m) for l in lengths for m in queue[l]]
        bfd_bins = _pack(pieces, stock_mm, kerf_mm)
        bfd_patterns = []
        for b in bfd_bins:
            c = [0] * len(lengths)
            for l, _ in b:
                c[idx[l]] += 1
            bfd_patterns.append(tuple(c))
        bfd_bars = len(bfd_bins)

        t0 = time.time()
        keys, copies, lp_value, converged, iters, status, cp_sec = _solve_diameter(
            lengths, demand, stock_mm, kerf_mm, bfd_patterns, time_limit_s, log=log)
        elapsed = time.time() - t0

        # ---- turn the chosen patterns into real bars with real marks ----
        new_bars: list[list[tuple[int, int]]] = []
        if copies is not None:
            q = {l: list(reversed(queue[l])) for l in lengths}
            for key, n_copies in zip(keys, copies):
                for _ in range(n_copies):
                    cuts: list[tuple[int, int]] = []
                    for i, a in enumerate(key):
                        for _ in range(a):
                            if q[lengths[i]]:               # skips pieces that would be surplus
                                cuts.append((lengths[i], q[lengths[i]].pop()))
                    if cuts:
                        new_bars.append(cuts)
            leftover = sum(len(v) for v in q.values())
            if leftover:                                    # should not happen; safe fallback
                new_bars = []
        used_fallback = (not new_bars) or len(new_bars) > bfd_bars
        final_bars = bfd_bins if used_fallback else new_bars
        if used_fallback:
            status = "Standard plan kept (" + status + ")"

        for cuts in final_bars:
            bar_no += 1
            plan_rows.append(plan_row(bar_no, dia, cuts, stock_mm, kerf_mm))

        lower = math.ceil(lp_value - 1e-6) if converged else None
        stat_rows.append(dict(
            dia=f"Ø{dia}", bfd_bars=bfd_bars, cpsat_bars=len(final_bars),
            lp_relaxation_bars=round(lp_value, 2) if converged else None, lower_bound_bars=lower,
            status=status, proven_optimal=bool(lower is not None and len(final_bars) == lower),
            lengths=len(lengths), patterns_generated=len(keys), search_rounds=iters,
            seconds=round(elapsed, 1)))
        log(f"Ø{dia}: Standard plan {bfd_bars:,} bars -> Advanced plan {len(final_bars):,} bars "
            f"(minimum possible {lower if lower is not None else '?'}) in {elapsed:.1f}s")

    return CpsatResult(build_result(plan_rows, bars, stock_mm, oversize), pd.DataFrame(stat_rows))