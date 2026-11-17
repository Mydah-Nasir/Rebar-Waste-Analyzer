"""Parse a Bar Bending Schedule (BBS) PDF into a table of bar marks.

The PDF has a real text layer, so we read it directly with PyMuPDF (no OCR).
In this drawing format the pages are stored rotated, so each table *row* is a
set of words sharing the same x coordinate, and the *columns* are found by their
y-distance from the bar-mark number.  Column offsets below were measured on
drawing ZWTF2-3-2-CV-DWG-4002-49 (BBS-004); other layouts may need new offsets.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field

import pandas as pd
import pymupdf

# kg per metre of plain round bar: d^2 / 162.2  (Ø20 -> 2.466, Ø25 -> 3.853)
def kg_per_m(dia_mm: int) -> float:
    return round(dia_mm ** 2 / 162.2, 3)


# Column positions, measured as y-distance from the mark-number word.
DIA_RANGE = (40, 75)
NOS_RANGE = (230, 290)
LEN_RANGE = (300, 360)
WT_RANGE = (380, 440)
NOTE_START, NOTE_END = 450, 880   # note column; stops before the second table on two-table pages
VALID_DIAS = ("20", "25")  # diameters present in this drawing; extend if needed


@dataclass
class ParseResult:
    bars: pd.DataFrame                       # one row per bar mark
    stated_totals: dict = field(default_factory=dict)  # {dia: (length_mm, mass_kg)} from the PDF
    page_count: int = 0
    warnings: list = field(default_factory=list)


def _row_groups(page):
    """Group words into table rows (same x), each sorted along y."""
    rows = collections.defaultdict(list)
    for x0, y0, x1, y1, text, *_ in page.get_text("words"):
        rows[round((x0 + x1) / 2 / 2.5)].append((y0, text))
    for ws in rows.values():
        ws.sort()
    return [(k * 2.5, ws) for k, ws in rows.items()]


def _between(ws, y, lo, hi):
    return [t for yy, t in ws if lo < yy - y < hi]


def parse_bbs_pdf(source) -> ParseResult:
    """`source` is a file path (str) or the PDF bytes."""
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, (bytes, bytearray)) else pymupdf.open(source)
    bars: dict[int, dict] = {}
    stated: dict[int, list] = {}

    for page_no, page in enumerate(doc, start=1):
        page_words = page.get_text("words")
        for row_x, ws in _row_groups(page):
            # --- summary table (total length / total mass per diameter) ---
            # (this row can share a line with a schedule row, so don't skip the row below)
            toks = [t for _, t in ws]
            dia_tok = [t for t in toks if t in VALID_DIAS]
            big = [t for t in toks if t.isdigit() and len(t) >= 8]
            mass = [t for t in toks if re.fullmatch(r"\d{4,}\.\d+", t)]
            if dia_tok and big and mass:
                stated[int(dia_tok[0])] = [int(big[0]), float(mass[0])]

            # --- bar-schedule rows ---
            for y, t in ws:
                if not t.isdigit():
                    continue
                dia = _between(ws, y, *DIA_RANGE)
                dia = [d for d in dia if d in VALID_DIAS]
                nos = _between(ws, y, *NOS_RANGE)
                ln = _between(ws, y, *LEN_RANGE)
                wt = _between(ws, y, *WT_RANGE)
                if not (dia and nos and ln and wt):
                    continue
                if not (nos[0].isdigit() and ln[0].isdigit()):
                    continue
                try:
                    weight = float(wt[0])
                except ValueError:
                    continue
                mark = int(t)
                note = _note_text(page_words, row_x, y)
                row = dict(
                    mark=mark, page=page_no, dia=int(dia[0]), qty=int(nos[0]),
                    length_mm=int(ln[0]), unit_kg=weight,
                    component=_clean_note(note),
                )
                # A row cut by a page edge is repeated on the next page: keep the copy whose
                # weight agrees with length x kg/m (the clipped copy has split digits).
                if mark in bars and _consistent(bars[mark]) or (mark in bars and not _consistent(row)):
                    continue
                bars[mark] = row

    df = pd.DataFrame(sorted(bars.values(), key=lambda r: r["mark"]))
    warnings = []
    if df.empty:
        warnings.append("No bar rows were found. This PDF layout may not match the parser.")
    return ParseResult(
        bars=df,
        stated_totals={d: tuple(v) for d, v in stated.items()},
        page_count=len(doc),
        warnings=warnings,
    )


def _note_text(page_words, row_x, mark_y):
    """Component text. Some rows have their note split into pieces that sit a few points off the
    row line, so match words with a small x tolerance instead of an exact row."""
    ws = [(w[1], w[4]) for w in page_words
          if abs((w[0] + w[2]) / 2 - row_x) <= 8 and NOTE_START < w[1] - mark_y < NOTE_END
          and not re.fullmatch(r"[\d.]+", w[4])]
    return " ".join(t for _, t in sorted(ws))[:60]


def _consistent(row: dict) -> bool:
    return abs(row["length_mm"] / 1000 * kg_per_m(row["dia"]) - row["unit_kg"]) <= 0.05


def _clean_note(note: str) -> str:
    # drop title-block words that sometimes sit on the same line as a row
    for stop in ("ATTIQ", "PROJECT", "INFLUENT", "CHECKED", "ISSUE", "SUB-CONTRACTOR", "REVIEW", "DRAWN"):
        i = note.find(stop)
        if i >= 0:
            note = note[:i]
    return note.strip()


def validate(result: ParseResult, stock_mm: int = 12000) -> list[dict]:
    """Automatic checks on the extracted table. Each check -> dict(name, ok, detail)."""
    df = result.bars
    checks: list[dict] = []
    if df.empty:
        return [dict(name="Rows extracted", ok=False, detail="0 rows")]

    # 1. Mark numbers continuous
    marks = set(df["mark"])
    missing = sorted(set(range(1, max(marks) + 1)) - marks)
    checks.append(dict(name="Bar marks continuous (1..N)", ok=not missing,
                       detail=f"{len(df)} marks found" + (f", missing {missing[:10]}" if missing else "")))

    # 2. Unit weight = length x kg/m
    exp = df["length_mm"] / 1000 * df["dia"].map(kg_per_m)
    bad = df[(exp - df["unit_kg"]).abs() > 0.05]
    checks.append(dict(
        name="Unit weight = length x kg/m", ok=bad.empty,
        detail="all rows consistent" if bad.empty else
        "check mark(s): " + ", ".join(f"{r.mark} (PDF {r.unit_kg} vs calc {exp[i]:.2f})" for i, r in bad.iterrows())))

    # 3. Totals vs totals printed on the drawing
    for dia, (s_len, s_mass) in sorted(result.stated_totals.items()):
        sub = df[df["dia"] == dia]
        our_len = int((sub["qty"] * sub["length_mm"]).sum())
        pct = abs(our_len - s_len) / s_len * 100
        checks.append(dict(
            name=f"Ø{dia} total length vs drawing summary", ok=pct < 0.05,
            detail=f"parsed {our_len:,} mm vs stated {s_len:,} mm ({pct:.4f}% diff)"))
    if not result.stated_totals:
        checks.append(dict(name="Drawing summary totals found", ok=False,
                           detail="could not read the totals table, so totals were not cross-checked"))

    # 4. Bars longer than the stock bar
    long_ = df[df["length_mm"] > stock_mm]
    checks.append(dict(
        name=f"All bars fit in a {stock_mm:,} mm stock bar", ok=long_.empty,
        detail="yes" if long_.empty else
        "longer than stock: " + ", ".join(f"mark {r.mark} ({r.length_mm} mm)" for r in long_.itertuples())))
    return checks
