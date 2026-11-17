"""Rebar cutting-waste demo.  Run:  streamlit run app.py"""
import pandas as pd
import streamlit as st

from baseline import baseline_by_mark, baseline_summary
from optimizer import compare, optimize
from bbs_parser import parse_bbs_pdf, validate
import cpsat_tab

st.set_page_config(page_title="Rebar Waste Analyzer", page_icon="🏗️", layout="wide")
st.title("🏗️ Rebar Waste Analyzer")
st.caption("Upload a Bar Bending Schedule (BBS) PDF, check the extracted data, and see how much steel is wasted "
           "if no cutting optimisation is done, then optimise the cutting plan.")
st.info(
    "**How weight is calculated:** weight = length in metres × kg per metre. "
    "kg per metre is fixed for each diameter by the standard steel formula **diameter² ÷ 162.2** "
    "(Ø20 = 2.466 kg/m, Ø25 = 3.853 kg/m). It is not read from the PDF; the unit weights printed in the PDF "
    "are only used to check that the data was read correctly.",
    icon="⚖️")

# ---------------- sidebar ----------------
with st.sidebar:
    st.header("Settings")
    stock_mm = st.number_input("Stock bar length (mm)", min_value=1000, max_value=24000, value=12000, step=500,
                               help="Not stated in the drawing. 12 m is assumed; confirm with the client.")
    kerf_mm = st.number_input("Saw cut width / kerf (mm)", min_value=0, max_value=20, value=0, step=1,
                              help="Steel lost at each cut. 0 ignores it.")
    st.markdown("**Baseline:** each required piece is cut from its own stock bar and the rest is scrap.\n\n"
                "**Optimised:** pieces of the same diameter share stock bars.")

uploaded = st.file_uploader("Upload BBS PDF", type=["pdf"])
if uploaded is None:
    st.info("Upload the BBS PDF to begin.")
    st.stop()


@st.cache_data(show_spinner="Reading PDF...")
def load(pdf_bytes: bytes):
    return parse_bbs_pdf(pdf_bytes)


result = load(uploaded.getvalue())
if result.bars.empty:
    st.error("; ".join(result.warnings))
    st.stop()

tab_data, tab_check, tab_base, tab_opt, tab_cpsat = st.tabs(
    ["1. Extracted data", "2. Validation", "3. Baseline waste", "4. Optimised plan & report",
     "5. Advanced optimisation"])
# ---------------- tab 1: extracted data (editable) ----------------
with tab_data:
    st.write(f"**{len(result.bars)} bar marks** read from {result.page_count} pages. "
             "You can fix any value in the table; the rest of the app uses your edits.")
    edited = st.data_editor(
        result.bars,
        hide_index=True, width="stretch", height=450, num_rows="fixed",
        column_config={
            "mark": st.column_config.NumberColumn("Mark", disabled=True, help="Bar mark number from the BBS."),
            "page": st.column_config.NumberColumn("PDF page", disabled=True, help="PDF page the row was read from."),
            "dia": st.column_config.NumberColumn("Dia (mm)", help="Bar diameter in mm."),
            "qty": st.column_config.NumberColumn("Qty", help="How many pieces of this mark are required."),
            "length_mm": st.column_config.NumberColumn("Cut length (mm)", help="Length of one piece before bending."),
            "unit_kg": st.column_config.NumberColumn("Unit weight (kg)", help="Weight of one piece as printed in the PDF (used only to check the data)."),
            "component": st.column_config.TextColumn("Component", help="Part of the structure the bar is used in."),
        },
    )

# keep validation/baseline in sync with edits
result.bars = edited


# ---------------- tab 2: validation ----------------
with tab_check:
    checks = validate(result, stock_mm)
    for c in checks:
        (st.success if c["ok"] else st.warning)(f"**{c['name']}**: {c['detail']}")
    if result.stated_totals:
        st.subheader("Totals printed on the drawing")
        st.dataframe(pd.DataFrame(
            [dict(Dia=f"Ø{d}", total_length_mm=l, total_mass_kg=m) for d, (l, m) in sorted(result.stated_totals.items())]
        ), hide_index=True, column_config={
            "Dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "total_length_mm": st.column_config.NumberColumn("Total length (mm)", format="%d", help="Total bar length for this diameter, as printed on page 7 of the drawing."),
            "total_mass_kg": st.column_config.NumberColumn("Total mass (kg)", format="%.2f", help="Total weight for this diameter, as printed on page 7 of the drawing."),
        })

# ---------------- tab 3: baseline ----------------
with tab_base:
    per_mark = baseline_by_mark(edited, stock_mm)
    summary = baseline_summary(per_mark, stock_mm)
    total = summary.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock bars needed", f"{int(total.stock_bars):,}")
    c2.metric("Steel required", f"{total.required_m / 1000:,.1f} km")
    c3.metric("Total waste", f"{total.waste_m / 1000:,.1f} km", f"{total.waste_pct:.1f}% of purchased", delta_color="inverse")
    c4.metric("Waste weight", f"{total.waste_kg / 1000:,.1f} t", delta_color="inverse")

    oversize = per_mark[per_mark["oversize"]]
    if not oversize.empty:
        st.warning(
            f"{len(oversize)} bar mark(s) are longer than the {stock_mm:,} mm stock bar and were left out of the "
            "waste figure: " + ", ".join(f"mark {r.mark} ({r.length_mm:,} mm)" for r in oversize.itertuples())
            + ". How to handle them (longer stock or a splice).")

    st.subheader("Summary by diameter")
    st.dataframe(
        summary, hide_index=True, width="stretch",
        column_config={
            "dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "pieces": st.column_config.NumberColumn("Pieces", format="%d", help="Total pieces required (sum of Qty)."),
            "stock_bars": st.column_config.NumberColumn("Stock bars", format="%d", help="12 m bars to buy: one new bar for every piece."),
            "required_m": st.column_config.NumberColumn("Required (m)", format="%.0f", help="Total length of the pieces needed: sum of qty x length."),
            "purchased_m": st.column_config.NumberColumn("Purchased (m)", format="%.0f", help="Total steel bought: stock bars x stock length."),
            "waste_m": st.column_config.NumberColumn("Waste (m)", format="%.0f", help="Purchased minus required."),
            "waste_pct": st.column_config.NumberColumn("Waste %", format="%.1f%%", help="Waste as a share of the steel bought."),
            "waste_kg": st.column_config.NumberColumn("Waste (kg)", format="%.0f", help="Waste metres x kg per metre (diameter squared / 162.2)."),
        })

    st.subheader("Waste per bar mark (largest first)")
    top = per_mark.sort_values("waste_kg", ascending=False)
    st.bar_chart(top.head(25).set_index("mark")["waste_kg"], y_label="Waste (kg)", x_label="Bar mark")
    st.dataframe(
        top[["mark", "dia", "qty", "length_mm", "waste_per_piece_mm", "waste_mm", "waste_kg", "oversize", "component"]],
        hide_index=True, width="stretch", height=350,
        column_config={
            "mark": st.column_config.NumberColumn("Mark", help="Bar mark number."),
            "dia": st.column_config.NumberColumn("Dia (mm)", help="Bar diameter."),
            "qty": st.column_config.NumberColumn("Qty", help="Pieces required."),
            "length_mm": st.column_config.NumberColumn("Cut length (mm)", help="Length of one piece."),
            "waste_per_piece_mm": st.column_config.NumberColumn("Waste per piece (mm)", help="Stock length minus piece length."),
            "waste_mm": st.column_config.NumberColumn("Waste (mm)", help="Qty x waste per piece."),
            "waste_kg": st.column_config.NumberColumn("Waste (kg)", format="%.1f", help="Waste length converted to weight."),
            "oversize": st.column_config.CheckboxColumn("Too long", help="Ticked if the piece is longer than the stock bar (left out of the waste figure)."),
            "component": st.column_config.TextColumn("Component", help="Part of the structure the bar is used in."),
        })

    st.download_button("Download baseline per mark (CSV)", top.to_csv(index=False).encode(),
                       file_name="baseline_waste_per_mark.csv", mime="text/csv")


# ---------------- tab 4: optimised plan & report ----------------
with tab_opt:
    @st.cache_data(show_spinner="Optimising cutting plan...")
    def run_opt(bars, stock, kerf):
        return optimize(bars, stock, kerf)

    opt = run_opt(edited, int(stock_mm), int(kerf_mm))
    if opt.plan.empty:
        st.warning("Nothing to optimise.")

    def plan_view(p):
        """Friendly column names for the bar-by-bar plan."""
        v = p.copy()
        v["dia"] = "Ø" + v["dia"].astype(str)
        v["left_kg"] = v["left_kg"].round(2)
        return v[["bar_no", "dia", "cuts", "pieces", "used_mm", "left_mm", "left_kg"]].rename(columns={
            "bar_no": "Bar #", "dia": "Dia", "cuts": "Cuts from this bar", "pieces": "Pieces",
            "used_mm": "Used (mm)", "left_mm": "Scrap (mm)", "left_kg": "Scrap (kg)"})

    PLAN_CFG = {
        "Bar #": st.column_config.NumberColumn("Bar #", format="%d", help="Number of this stock bar in the plan."),
        "Dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
        "Cuts from this bar": st.column_config.TextColumn("Cuts from this bar", help="Marks and lengths cut from this bar. xN means N pieces of that mark."),
        "Pieces": st.column_config.NumberColumn("Pieces", format="%d", help="How many pieces are cut from this bar."),
        "Used (mm)": st.column_config.NumberColumn("Used (mm)", format="%d", help="Total length of the pieces on this bar."),
        "Scrap (mm)": st.column_config.NumberColumn("Scrap (mm)", format="%d", help="Stock length minus used: what is left over."),
        "Scrap (kg)": st.column_config.NumberColumn("Scrap (kg)", format="%.2f", help="That scrap converted to weight."),
    }

    cmp_ = compare(summary, opt.summary)
    t = cmp_[cmp_["dia"] == "Total"].iloc[0]
    o_tot = opt.summary.iloc[-1]

    st.subheader("Before vs after")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Stock bars needed", f"{int(t.bars_after):,}", f"-{int(t.bars_saved):,} vs no optimisation", delta_color="inverse")
    c2.metric("Waste length", f"{o_tot.waste_m / 1000:,.1f} km", f"was {t.waste_m_before / 1000:,.1f} km", delta_color="off")
    c3.metric("Waste weight", f"{t.waste_kg_after / 1000:,.1f} t", f"-{t.waste_kg_saved / 1000:,.1f} t saved", delta_color="inverse")
    c4.metric("Waste reduced by", f"{t.reduction_pct:.1f}%")

    st.dataframe(
        cmp_, hide_index=True, width="stretch",
        column_config={
            "dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "bars_before": st.column_config.NumberColumn("Bars (before)", format="%d", help="Stock bars needed with no optimisation."),
            "bars_after": st.column_config.NumberColumn("Bars (after)", format="%d", help="Stock bars needed after optimisation."),
            "bars_saved": st.column_config.NumberColumn("Bars saved", format="%d", help="Before minus after."),
            "waste_m_before": st.column_config.NumberColumn("Scrap m (before)", format="%.0f", help="Scrap length with no optimisation."),
            "waste_m_after": st.column_config.NumberColumn("Scrap m (after)", format="%.0f", help="Scrap length after optimisation."),
            "waste_kg_before": st.column_config.NumberColumn("Scrap kg (before)", format="%.0f", help="Scrap weight with no optimisation."),
            "waste_kg_after": st.column_config.NumberColumn("Scrap kg (after)", format="%.0f", help="Scrap weight after optimisation."),
            "waste_kg_saved": st.column_config.NumberColumn("Kg saved", format="%.0f", help="Scrap weight before minus after."),
            "reduction_pct": st.column_config.NumberColumn("Reduction", format="%.1f%%", help="Kg saved as a share of the scrap before."),
        })
    st.caption("Whatever is left on a stock bar after its pieces are cut is counted as scrap.")

    if not opt.oversize.empty:
        st.warning("Not planned (longer than the stock bar): "
                   + ", ".join(f"mark {r.mark} ({r.length_mm:,} mm)" for r in opt.oversize.itertuples()))

    st.subheader("Cutting patterns (identical bars grouped)")
    st.caption("Each row is one way of cutting a stock bar, repeated for the number of bars shown.")
    st.dataframe(
        opt.patterns, hide_index=True, width="stretch", height=300,
        column_order=["dia", "cuts", "bars", "left_mm_each", "left_kg_total"],
        column_config={
            "dia": st.column_config.NumberColumn("Dia (mm)", help="Bar diameter."),
            "cuts": st.column_config.TextColumn("Cuts from one bar", help="Marks and lengths cut from one bar. xN means N pieces of that mark."),
            "bars": st.column_config.NumberColumn("Number of bars", format="%d", help="How many stock bars are cut exactly this way."),
            "left_mm_each": st.column_config.NumberColumn("Scrap per bar (mm)", format="%d", help="What is left over on each of these bars."),
            "left_kg_total": st.column_config.NumberColumn("Total scrap (kg)", format="%.1f", help="Scrap per bar x number of bars, as weight."),
        })

    st.subheader("Which marks cause the most scrap")
    st.caption("Each bar's scrap is shared among the marks cut from it. Marks at the top are candidates to review "
               "(any change to a length needs the structural engineer's approval).")
    st.bar_chart(opt.by_mark.head(15).set_index("mark")["scrap_kg"], y_label="Scrap (kg)", x_label="Bar mark")
    st.dataframe(
        opt.by_mark.head(15), hide_index=True, width="stretch",
        column_order=["mark", "component", "dia", "length_mm", "pieces_cut", "scrap_mm", "scrap_kg"],
        column_config={
            "mark": st.column_config.NumberColumn("Mark", help="Bar mark number."),
            "component": st.column_config.TextColumn("Component", help="Part of the structure the bar is used in."),
            "dia": st.column_config.NumberColumn("Dia (mm)", help="Bar diameter."),
            "length_mm": st.column_config.NumberColumn("Cut length (mm)", help="Length of one piece."),
            "pieces_cut": st.column_config.NumberColumn("Pieces cut", format="%d", help="How many pieces of this mark were cut."),
            "scrap_mm": st.column_config.NumberColumn("Scrap (mm)", format="%d", help="Scrap assigned to this mark: each bar's leftover shared among its cuts by length."),
            "scrap_kg": st.column_config.NumberColumn("Scrap (kg)", format="%.0f", help="That scrap converted to weight."),
        })

    st.subheader("Bar-by-bar cutting plan")
    only = st.radio("Show", ["All bars", "Only bars with scrap"], horizontal=True)
    plan = opt.plan
    if only == "Only bars with scrap":
        plan = plan[plan["left_mm"] > 0]
    st.write(f"{len(plan):,} bars")
    st.dataframe(plan_view(plan), hide_index=True, width="stretch", height=350, column_config=PLAN_CFG)

    d1, d2 = st.columns(2)
    d1.download_button("Download full cutting plan (CSV)", plan_view(opt.plan).to_csv(index=False).encode(),
                       file_name="cutting_plan.csv", mime="text/csv")
    d2.download_button("Download cutting patterns (CSV)", opt.patterns.to_csv(index=False).encode(),
                       file_name="cutting_patterns.csv", mime="text/csv")


# ---------------- tab 5: CP-SAT optimiser ----------------
with tab_cpsat:
    if opt.plan.empty:
        st.warning("Nothing to optimise.")
    else:
        cpsat_tab.render(edited, int(stock_mm), int(kerf_mm), summary, opt)

