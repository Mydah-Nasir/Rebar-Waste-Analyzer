"""Cutting optimiser (backend): decide which pieces share a stock bar.

Method: Best Fit Decreasing, run separately for each diameter.
  1. Sort all required pieces longest first.
  2. Put each piece into the partly used bar where it fits most tightly.
  3. Open a new stock bar only when no open bar has room.
It is a fast heuristic, not a proof of the absolute minimum (an exact OR-Tools mode can be added later).
"""
from __future__ import annotations

import bisect
from collections import Counter, defaultdict
from dataclasses import dataclass

import pandas as pd

from bbs_parser import kg_per_m


@dataclass
class OptResult:
    plan: pd.DataFrame        # one row per stock bar
    patterns: pd.DataFrame    # identical bars grouped: "this cut pattern x N bars"
    summary: pd.DataFrame     # per diameter + Total
    by_mark: pd.DataFrame     # scrap attributed to each bar mark
    oversize: pd.DataFrame    # marks longer than the stock bar (not planned)


def _pack(pieces: list[tuple[int, int]], stock_mm: int, kerf_mm: int):
    """Best Fit Decreasing for one diameter. pieces = [(length, mark), ...].
    Returns a list of bins, each a list of (length, mark) in cutting order.

    With a saw kerf, each cut uses length+kerf of the bar; a bar can hold stock+kerf of those
    (the last cut needs no extra kerf), so kerf=0 gives the plain sum-of-lengths rule.
    """
    capacity = stock_mm + kerf_mm
    smallest = min(l for l, _ in pieces) + kerf_mm
    bins: list[list[tuple[int, int]]] = []
    open_by_room: dict[int, list[int]] = defaultdict(list)   # remaining room -> bin ids
    rooms: list[int] = []                                    # sorted distinct remaining rooms

    for length, mark in sorted(pieces, key=lambda p: -p[0]):
        need = length + kerf_mm
        i = bisect.bisect_left(rooms, need)
        if i < len(rooms):
            room = rooms[i]
            b = open_by_room[room].pop()
            if not open_by_room[room]:
                del open_by_room[room]
                rooms.pop(i)
        else:
            b = len(bins)
            bins.append([])
            room = capacity
        bins[b].append((length, mark))
        left = room - need
        if left >= smallest:                                  # can still fit something
            if left not in open_by_room:
                bisect.insort(rooms, left)
            open_by_room[left].append(b)
    return bins


def _cut_label(cuts: list[tuple[int, int]]) -> str:
    """e.g. 'mark 223 (5232 mm) + mark 356 (6575 mm)'; repeated cuts become 'mark 212 (2576 mm) x4'."""
    c = Counter((m, l) for l, m in cuts)
    return " + ".join(f"mark {m} ({l:,} mm)" + (f" x{n}" if n > 1 else "")
                      for (m, l), n in sorted(c.items(), key=lambda kv: -kv[0][1]))


def plan_row(bar_no: int, dia: int, cuts: list[tuple[int, int]], stock_mm: int, kerf_mm: int) -> dict:
    """One stock bar in the plan. cuts = [(length, mark), ...]."""
    used = sum(l for l, _ in cuts)
    left = stock_mm - used - kerf_mm * (len(cuts) - 1)   # kerf is lost steel too
    return dict(
        bar_no=bar_no, dia=int(dia), cuts=_cut_label(cuts), pieces=len(cuts),
        used_mm=used, left_mm=left, left_kg=left / 1000 * kg_per_m(int(dia)),
        _marks=tuple((m, l) for l, m in cuts))


def split_oversize(bars: pd.DataFrame, stock_mm: int):
    """Marks longer than the stock bar can't be planned; return (oversize table, plannable rows)."""
    oversize = bars[bars["length_mm"] > stock_mm][["mark", "dia", "qty", "length_mm"]].reset_index(drop=True)
    return oversize, bars[bars["length_mm"] <= stock_mm]


def optimize(bars: pd.DataFrame, stock_mm: int = 12000, kerf_mm: int = 0) -> OptResult:
    """Best Fit Decreasing. bars: columns mark, dia, qty, length_mm (and optionally component)."""
    oversize, ok = split_oversize(bars, stock_mm)

    plan_rows = []
    bar_no = 0
    for dia, g in ok.groupby("dia"):
        pieces = [(int(r.length_mm), int(r.mark)) for r in g.itertuples() for _ in range(int(r.qty))]
        if not pieces:
            continue
        for cuts in _pack(pieces, stock_mm, kerf_mm):
            bar_no += 1
            plan_rows.append(plan_row(bar_no, int(dia), cuts, stock_mm, kerf_mm))
    return build_result(plan_rows, bars, stock_mm, oversize)


def build_result(plan_rows: list[dict], bars: pd.DataFrame, stock_mm: int, oversize: pd.DataFrame) -> OptResult:
    """Turn a list of planned bars into the plan / patterns / summary / by-mark tables (shared by all optimisers)."""
    plan = pd.DataFrame(plan_rows)

    if plan.empty:
        empty = pd.DataFrame()
        return OptResult(plan, empty, empty, empty, oversize)

    # ---- grouped patterns ----
    patterns = (plan.groupby(["dia", "cuts"], as_index=False)
                .agg(bars=("bar_no", "count"), first_bar=("bar_no", "min"), left_mm_each=("left_mm", "first"),
                     left_kg_total=("left_kg", "sum"))
                .sort_values("bars", ascending=False).reset_index(drop=True))

    # ---- summary per diameter ----
    rows = []
    for dia, g in plan.groupby("dia"):
        purchased = len(g) * stock_mm
        left = int(g["left_mm"].sum())
        required = int(g["used_mm"].sum())
        rows.append(dict(dia=f"Ø{dia}", stock_bars=len(g), required_m=required / 1000,
                         purchased_m=purchased / 1000, waste_m=left / 1000,
                         waste_pct=left / purchased * 100, waste_kg=g["left_kg"].sum()))
    summary = pd.DataFrame(rows)
    tot = summary.sum(numeric_only=True)
    tot_row = dict(dia="Total", **tot.to_dict())
    tot_row["waste_pct"] = tot["waste_m"] / tot["purchased_m"] * 100
    summary = pd.concat([summary, pd.DataFrame([tot_row])], ignore_index=True)

    # ---- scrap attributed to marks (each bar's leftover shared by its cuts, by length) ----
    attrib: dict[int, list[float]] = defaultdict(lambda: [0.0, 0.0, 0])
    for r in plan.to_dict("records"):
        tot_len = sum(l for _, l in r["_marks"])
        for m, l in r["_marks"]:
            share = r["left_mm"] * l / tot_len
            a = attrib[m]
            a[0] += share
            a[1] += share / 1000 * kg_per_m(r["dia"])
            a[2] += 1
    by_mark = pd.DataFrame(
        [dict(mark=m, pieces_cut=v[2], scrap_mm=v[0], scrap_kg=v[1]) for m, v in attrib.items()]
    ).merge(bars[["mark", "dia", "length_mm", "component"]], on="mark")
    by_mark = by_mark.sort_values("scrap_kg", ascending=False).reset_index(drop=True)

    return OptResult(plan.drop(columns="_marks"), patterns, summary, by_mark, oversize)




def compare(baseline: pd.DataFrame, optimised: pd.DataFrame) -> pd.DataFrame:
    """Before/after table per diameter + Total. Both inputs are the summary frames (same dia labels)."""
    b = baseline.set_index("dia")
    o = optimised.set_index("dia")
    rows = []
    for dia in b.index:
        if dia not in o.index:
            continue
        rows.append(dict(
            dia=dia,
            bars_before=int(b.loc[dia, "stock_bars"]), bars_after=int(o.loc[dia, "stock_bars"]),
            waste_m_before=b.loc[dia, "waste_m"], waste_m_after=o.loc[dia, "waste_m"],
            waste_kg_before=b.loc[dia, "waste_kg"], waste_kg_after=o.loc[dia, "waste_kg"]))
    out = pd.DataFrame(rows)
    out["bars_saved"] = out["bars_before"] - out["bars_after"]
    out["waste_kg_saved"] = out["waste_kg_before"] - out["waste_kg_after"]
    out["reduction_pct"] = out["waste_kg_saved"] / out["waste_kg_before"] * 100
    return out
