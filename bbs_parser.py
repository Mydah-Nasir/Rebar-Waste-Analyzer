"""Parse a Bar Bending Schedule (BBS) PDF into a table of bar marks.

Two PDF layouts are supported and detected automatically:

  1. "drawing"  - a BBS drawing sheet (e.g. ZWTF2-3-2-CV-DWG-4002-49 BBS-004).
     Real text layer, pages stored rotated, so each table *row* is a set of words sharing the same
     x coordinate and the *columns* are found by their y-distance from the bar-mark number.
     Column offsets below were measured on that drawing.
  2. "brc"     - a fabricator bar list (e.g. BRC-PRD-FRM-101, ATRC-82_DCJE_BBS).
     Normal upright pages. One text line per bar mark:
       Bar Mark | Bar Diam | No of Mem | No of Bars | Total Bars | Length (mm) | [shape image] | Weight (kgs) | Notes
     The weight printed here is the TOTAL for the row (all bars), not per bar.
     A "WEIGHT SUMMARY" table at the end gives items and kgs per diameter.

Both give the same table: mark, page, dia, qty, length_mm, unit_kg (weight of ONE piece), component.
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


# ---------------------------------------------------------------- layout 1 (drawing) settings
# Column positions, measured as y-distance from the mark-number word.
DIA_RANGE = (40, 75)
NOS_RANGE = (230, 290)
LEN_RANGE = (300, 360)
WT_RANGE = (380, 440)
NOTE_START, NOTE_END = 450, 880   # note column; stops before the second table on two-table pages
VALID_DIAS = ("8", "10", "12", "16", "20", "25", "28", "32", "36", "40")  # standard bar sizes

# ---------------------------------------------------------------- layout 2 (brc) settings
BRC_MIN_DIA, BRC_MAX_DIA = 6, 50          # plausible bar diameters (mm)
BRC_MIN_LEN, BRC_MAX_LEN = 50, 40000      # plausible cut lengths (mm)


@dataclass
class ParseResult:
    bars: pd.DataFrame                       # one row per bar mark
    # {dia: (length_mm or None, mass_kg)} as printed in the PDF. The BRC layout prints no total length -> None.
    stated_totals: dict = field(default_factory=dict)
    page_count: int = 0
    warnings: list = field(default_factory=list)
    layout: str = ""                         # "drawing" or "brc"
    stated_items: dict = field(default_factory=dict)      # {dia: number of bar marks} (BRC summary)
    stated_scrap_kg: dict = field(default_factory=dict)   # {dia: offcut & scrap kg} (BRC summary)


# ================================================================ entry point
def parse_bbs_pdf(source) -> ParseResult:
    """`source` is a file path (str) or the PDF bytes. The layout is detected automatically."""
    doc = pymupdf.open(stream=source, filetype="pdf") if isinstance(source, (bytes, bytearray)) else pymupdf.open(source)

    guess = _detect_layout(doc)
    order = [guess] + [k for k in ("drawing", "brc") if k != guess]
    parsers = {"drawing": _parse_drawing, "brc": _parse_brc}

    for key in order:
        result = parsers[key](doc)
        if not result.bars.empty:
            result.layout = key
            return result

    return ParseResult(
        bars=pd.DataFrame(), page_count=len(doc),
        warnings=["No bar rows were found. This PDF layout is not recognised by the parser."])


def _detect_layout(doc) -> str:
    text = " ".join(page.get_text() for page in doc)
    if re.search(r"No\s+of\s+Mem|BRC-PRD-FRM|Offcut\s*&\s*Scrap", text, re.I):
        return "brc"
    if re.search(r"Bending\s+shape", text, re.I):
        return "drawing"
    return "drawing"


# ================================================================ layout 1: drawing
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


def _parse_drawing(doc) -> ParseResult:
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


# ================================================================ layout 2: brc bar list
_INT_RE = re.compile(r"\d{1,3}(?:,\d{3})+|\d+")
_DEC_RE = re.compile(r"(?:\d{1,3}(?:,\d{3})+|\d+)\.\d+")


def _as_int(tok: str):
    return int(tok.replace(",", "")) if _INT_RE.fullmatch(tok) else None


def _as_dec(tok: str):
    return float(tok.replace(",", "")) if _DEC_RE.fullmatch(tok) else None


def _text_lines(page, tol: float = 2.5):
    """Words grouped into text lines (same vertical position), each sorted left to right."""
    words = sorted(page.get_text("words"), key=lambda w: ((w[1] + w[3]) / 2, w[0]))
    lines: list[list] = []          # [y_centre, [(x0, x1, text), ...]]
    for x0, y0, x1, y1, text, *_ in words:
        yc = (y0 + y1) / 2
        if lines and abs(lines[-1][0] - yc) <= tol:
            lines[-1][1].append((x0, x1, text))
        else:
            lines.append([yc, [(x0, x1, text)]])
    for ln in lines:
        ln[1].sort()
    return lines


def _brc_row(items):
    """One text line -> dict, or None if it is not a bar row.
    Expected order: mark, dia, no of members, no of bars, total bars, length ... weight (decimal) ... notes."""
    toks = [t for _, _, t in items]
    ints = []
    i = 0
    while i < len(toks) and len(ints) < 6:
        v = _as_int(toks[i])
        if v is None:
            break
        ints.append(v)
        i += 1
    if len(ints) < 6:
        return None
    mark, dia, mem, nbars, total, length = ints
    if not (BRC_MIN_DIA <= dia <= BRC_MAX_DIA and BRC_MIN_LEN <= length <= BRC_MAX_LEN and total >= 1):
        return None

    weight, widx = None, None
    for j in range(i, len(items)):                 # shape-image numbers are whole numbers; the weight has decimals
        w = _as_dec(items[j][2])
        if w is not None:
            weight, widx = w, j
            break
    if weight is None:
        return None

    note = " ".join(t for _, _, t in items[widx + 1:])
    return dict(mark=mark, dia=dia, mem=mem, nbars=nbars, total=total, length=length, weight=weight, note=note)


def _parse_brc(doc) -> ParseResult:
    rows: dict[int, dict] = {}
    warnings: list[str] = []
    stated: dict[int, tuple] = {}
    items_by_dia: dict[int, int] = {}
    scrap_by_dia: dict[int, float] = {}
    element = ""
    in_summary = False

    for page_no, page in enumerate(doc, start=1):
        for _, items in _text_lines(page):
            text = " ".join(t for _, _, t in items)

            m = re.match(r"Element:?\s*(\S+)", text, re.I)
            if m:
                element = m.group(1)
                continue
            if re.search(r"WEIGHT\s+SUMMARY", text, re.I):
                in_summary = True
                continue

            r = _brc_row(items)
            if r is not None:
                if r["total"] != r["mem"] * r["nbars"]:
                    warnings.append(f"Mark {r['mark']}: Total Bars ({r['total']}) is not No of Mem x No of Bars "
                                    f"({r['mem']} x {r['nbars']}); Total Bars was used.")
                mark = r["mark"]
                if mark in rows:
                    old = rows[mark]
                    if old["dia"] == r["dia"] and old["length_mm"] == r["length"]:
                        old["qty"] += r["total"]          # same bar listed under another element: add quantities
                        old["_kg"] += r["weight"]
                        warnings.append(f"Mark {mark} appears more than once (same size); quantities were added.")
                    else:
                        warnings.append(f"Mark {mark} appears twice with different size or length; the first one was kept.")
                    continue
                rows[mark] = dict(
                    mark=mark, page=page_no, dia=r["dia"], qty=r["total"], length_mm=r["length"],
                    _kg=r["weight"], component=(r["note"].strip()[:60] or (f"Element {element}" if element else "")))
                continue

            # --- WEIGHT SUMMARY rows:  <dia> <items> [<offcut&scrap kg>] <kg> ---
            if in_summary:
                toks = [t for _, _, t in items]
                if toks and _as_int(toks[0]) is not None and len(toks) >= 3 and _as_int(toks[1]) is not None:
                    dia = _as_int(toks[0])
                    decs = [d for d in (_as_dec(t) for t in toks[2:]) if d is not None]
                    if decs and BRC_MIN_DIA <= dia <= BRC_MAX_DIA:
                        stated[dia] = (None, decs[-1])
                        items_by_dia[dia] = _as_int(toks[1])
                        if len(decs) >= 2:
                            scrap_by_dia[dia] = decs[0]

    out = []
    for r in rows.values():
        unit = round(r["_kg"] / r["qty"], 6) if r["qty"] else 0.0
        out.append(dict(mark=r["mark"], page=r["page"], dia=r["dia"], qty=r["qty"], length_mm=r["length_mm"],
                        unit_kg=unit, component=r["component"]))
    df = pd.DataFrame(sorted(out, key=lambda r: r["mark"]))
    if df.empty:
        warnings.append("No bar rows were found. This PDF layout may not match the parser.")
    return ParseResult(bars=df, stated_totals=stated, page_count=len(doc), warnings=warnings,
                       stated_items=items_by_dia, stated_scrap_kg=scrap_by_dia)


# ================================================================ validation
def validate(result: ParseResult, stock_mm: int = 12000) -> list[dict]:
    """Automatic checks on the extracted table. Each check -> dict(name, ok, detail)."""
    df = result.bars
    checks: list[dict] = []
    if df.empty:
        return [dict(name="Rows extracted", ok=False, detail="0 rows")]
    brc = result.layout == "brc"

    # 1. Mark numbers
    marks = set(df["mark"])
    if brc:
        # a fabricator bar list normally shows only some marks of the drawing, so continuity is not expected
        n_stated = sum(result.stated_items.values()) if result.stated_items else None
        if n_stated is not None:
            checks.append(dict(name="Bar marks found vs items in summary", ok=len(df) == n_stated,
                               detail=f"{len(df)} marks read, summary table says {n_stated}"))
        else:
            checks.append(dict(name="Bar marks found", ok=True, detail=f"{len(df)} marks found"))
    else:
        missing = sorted(set(range(1, max(marks) + 1)) - marks)
        checks.append(dict(name="Bar marks continuous (1..N)", ok=not missing,
                           detail=f"{len(df)} marks found" + (f", missing {missing[:10]}" if missing else "")))

    # 2. Unit weight = length x kg/m  (BRC prints kg/m rounded slightly differently, so allow 0.5%)
    exp = df["length_mm"] / 1000 * df["dia"].map(kg_per_m)
    tol = (exp * 0.005).clip(lower=0.05) if brc else 0.05
    bad = df[(exp - df["unit_kg"]).abs() > tol]
    checks.append(dict(
        name="Unit weight = length x kg/m", ok=bad.empty,
        detail="all rows consistent" if bad.empty else
        "check mark(s): " + ", ".join(f"{r.mark} (PDF {r.unit_kg} vs calc {exp[i]:.2f})" for i, r in bad.iterrows())))

    # 3. Totals vs totals printed in the PDF
    for dia, (s_len, s_mass) in sorted(result.stated_totals.items()):
        sub = df[df["dia"] == dia]
        if s_len is not None:
            our_len = int((sub["qty"] * sub["length_mm"]).sum())
            pct = abs(our_len - s_len) / s_len * 100
            checks.append(dict(
                name=f"Ø{dia} total length vs drawing summary", ok=pct < 0.05,
                detail=f"parsed {our_len:,} mm vs stated {s_len:,} mm ({pct:.4f}% diff)"))
        else:
            our_kg = float((sub["qty"] * sub["unit_kg"]).sum())
            pct = abs(our_kg - s_mass) / s_mass * 100 if s_mass else 0.0
            checks.append(dict(
                name=f"Ø{dia} total weight vs summary", ok=pct < 0.05,
                detail=f"parsed {our_kg:,.3f} kg vs stated {s_mass:,.3f} kg ({pct:.4f}% diff)"))
    if not result.stated_totals:
        checks.append(dict(name="Summary totals found", ok=False,
                           detail="could not read the totals table, so totals were not cross-checked"))

    # 4. Bars longer than the stock bar
    long_ = df[df["length_mm"] > stock_mm]
    checks.append(dict(
        name=f"All bars fit in a {stock_mm:,} mm stock bar", ok=long_.empty,
        detail="yes" if long_.empty else
        "longer than stock: " + ", ".join(f"mark {r.mark} ({r.length_mm} mm)" for r in long_.itertuples())))
    return checks