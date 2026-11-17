"""Baseline (no optimisation) waste: every required piece is cut from its own stock bar."""
from __future__ import annotations

import pandas as pd

from bbs_parser import kg_per_m


def baseline_by_mark(bars: pd.DataFrame, stock_mm: int = 12000) -> pd.DataFrame:
    """Per bar mark: waste if each piece is cut from a separate stock bar.

    Pieces longer than the stock bar can't be cut from one bar, so they are flagged
    (`oversize`) and left out of the waste figure instead of guessed at.
    """
    df = bars.copy()
    df["oversize"] = df["length_mm"] > stock_mm
    ok = ~df["oversize"]
    df["stock_bars"] = df["qty"].where(ok, 0)
    df["required_mm"] = (df["qty"] * df["length_mm"]).where(ok, 0)
    df["waste_per_piece_mm"] = (stock_mm - df["length_mm"]).where(ok, 0)
    df["waste_mm"] = df["qty"] * df["waste_per_piece_mm"]
    df["waste_kg"] = df["waste_mm"] / 1000 * df["dia"].map(kg_per_m)
    return df


def baseline_summary(per_mark: pd.DataFrame, stock_mm: int = 12000) -> pd.DataFrame:
    """Per diameter plus a Total row."""
    rows = []
    for dia, g in per_mark.groupby("dia"):
        stock_used = int(g["stock_bars"].sum())
        bought = stock_used * stock_mm
        rows.append(dict(
            dia=f"Ø{dia}",
            pieces=int(g["qty"].sum()),
            stock_bars=stock_used,
            required_m=g["required_mm"].sum() / 1000,
            purchased_m=bought / 1000,
            waste_m=g["waste_mm"].sum() / 1000,
            waste_pct=(g["waste_mm"].sum() / bought * 100) if bought else 0.0,
            waste_kg=g["waste_kg"].sum(),
        ))
    out = pd.DataFrame(rows)
    total = dict(
        dia="Total", pieces=out["pieces"].sum(), stock_bars=out["stock_bars"].sum(),
        required_m=out["required_m"].sum(), purchased_m=out["purchased_m"].sum(),
        waste_m=out["waste_m"].sum(),
        waste_pct=out["waste_m"].sum() / out["purchased_m"].sum() * 100 if out["purchased_m"].sum() else 0.0,
        waste_kg=out["waste_kg"].sum(),
    )
    return pd.concat([out, pd.DataFrame([total])], ignore_index=True)
