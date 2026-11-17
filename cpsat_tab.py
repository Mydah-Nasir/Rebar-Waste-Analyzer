"""Tab 5 of the app: advanced optimiser compared with the standard plan."""
import hashlib
import time

import pandas as pd
import streamlit as st

from cpsat_optimizer import optimize_cpsat


def _plan_view(p: pd.DataFrame) -> pd.DataFrame:
    v = p.copy()
    v["dia"] = "Ø" + v["dia"].astype(str)
    v["left_kg"] = v["left_kg"].round(2)
    return v[["bar_no", "dia", "cuts", "pieces", "used_mm", "left_mm", "left_kg"]].rename(columns={
        "bar_no": "Bar #", "dia": "Dia", "cuts": "Cuts from this bar", "pieces": "Pieces",
        "used_mm": "Used (mm)", "left_mm": "Scrap (mm)", "left_kg": "Scrap (kg)"})


def render(bars: pd.DataFrame, stock_mm: int, kerf_mm: int, baseline: pd.DataFrame, bfd) -> None:
    """bars: current (edited) bar table; baseline: baseline summary frame; bfd: standard plan OptResult (tab 4)."""
    st.caption("Finds the cutting plan with the fewest stock bars, and shows whether anything better is possible. "
               "Compared here with the faster standard plan from tab 4.")

    c1, c2 = st.columns([1, 3])
    time_limit = c1.number_input("Time limit per diameter (s)", min_value=5, max_value=600, value=30, step=5,
                                 help="The optimiser stops after this long and keeps the best plan found.")
    run = c2.button("Run advanced optimisation", type="primary")

    sig = hashlib.md5((bars.to_csv(index=False) + f"|{stock_mm}|{kerf_mm}|{time_limit}").encode()).hexdigest()
    if run:
        with st.status("Running advanced optimisation...", expanded=True) as status:
            t0 = time.time()
            res = optimize_cpsat(bars, stock_mm, kerf_mm, float(time_limit), progress=lambda m: st.write(m))
            status.update(label=f"Done in {time.time() - t0:.1f} s", state="complete", expanded=False)
        st.session_state["cpsat"] = (sig, res)

    saved = st.session_state.get("cpsat")
    if saved is None:
        st.info("Press **Run advanced optimisation** to start.")
        return
    if saved[0] != sig:
        st.warning("The data or settings changed since the last run. Press **Run advanced optimisation** again.")
        return

    res = saved[1]
    opt = res.opt
    if opt.plan.empty:
        st.warning("Nothing to optimise.")
        return
    stats = res.stats

    # ------------------------------------------------------------------ comparison
    b, f, c = baseline.set_index("dia"), bfd.summary.set_index("dia"), opt.summary.set_index("dia")
    rows = []
    for dia in c.index:
        rows.append(dict(
            dia=dia,
            bars_none=int(b.loc[dia, "stock_bars"]), bars_bfd=int(f.loc[dia, "stock_bars"]),
            bars_cpsat=int(c.loc[dia, "stock_bars"]),
            scrap_m_bfd=f.loc[dia, "waste_m"], scrap_m_cpsat=c.loc[dia, "waste_m"],
            scrap_kg_none=b.loc[dia, "waste_kg"], scrap_kg_bfd=f.loc[dia, "waste_kg"],
            scrap_kg_cpsat=c.loc[dia, "waste_kg"], scrap_pct_cpsat=c.loc[dia, "waste_pct"]))
    cmp_ = pd.DataFrame(rows)
    cmp_["bars_saved"] = cmp_["bars_bfd"] - cmp_["bars_cpsat"]
    cmp_["kg_saved"] = cmp_["scrap_kg_bfd"] - cmp_["scrap_kg_cpsat"]
    tot = cmp_[cmp_["dia"] == "Total"].iloc[0]

    st.subheader("Standard plan vs Advanced plan")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Stock bars (Advanced)", f"{int(tot.bars_cpsat):,}", f"-{int(tot.bars_saved):,} vs Standard", delta_color="inverse")
    m2.metric("Scrap length (Advanced)", f"{tot.scrap_m_cpsat / 1000:,.2f} km", f"Standard {tot.scrap_m_bfd / 1000:,.2f} km", delta_color="off")
    m3.metric("Scrap weight (Advanced)", f"{tot.scrap_kg_cpsat / 1000:,.1f} t", f"-{tot.kg_saved / 1000:,.1f} t vs Standard", delta_color="inverse")
    m4.metric("Scrap rate (Advanced)", f"{tot.scrap_pct_cpsat:.2f}%")

    st.dataframe(
        cmp_, hide_index=True, width="stretch",
        column_config={
            "dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "bars_none": st.column_config.NumberColumn("Bars (no optimisation)", format="%d", help="One new stock bar for every piece."),
            "bars_bfd": st.column_config.NumberColumn("Bars (Standard)", format="%d", help="Stock bars with the standard plan."),
            "bars_cpsat": st.column_config.NumberColumn("Bars (Advanced)", format="%d", help="Stock bars with the advanced plan."),
            "bars_saved": st.column_config.NumberColumn("Bars saved vs Standard", format="%d", help="Standard bars minus Advanced bars."),
            "scrap_m_bfd": st.column_config.NumberColumn("Scrap m (Standard)", format="%.0f", help="Scrap length with the standard plan."),
            "scrap_m_cpsat": st.column_config.NumberColumn("Scrap m (Advanced)", format="%.0f", help="Scrap length with the advanced plan."),
            "scrap_kg_none": st.column_config.NumberColumn("Scrap kg (no optimisation)", format="%.0f", help="Scrap weight if every piece gets its own bar."),
            "scrap_kg_bfd": st.column_config.NumberColumn("Scrap kg (Standard)", format="%.0f", help="Scrap weight with the standard plan."),
            "scrap_kg_cpsat": st.column_config.NumberColumn("Scrap kg (Advanced)", format="%.0f", help="Scrap weight with the advanced plan."),
            "kg_saved": st.column_config.NumberColumn("Kg saved vs Standard", format="%.0f", help="Standard scrap weight minus Advanced scrap weight."),
            "scrap_pct_cpsat": st.column_config.NumberColumn("Scrap % (Advanced)", format="%.2f%%", help="Scrap as a share of the steel bought."),
        })

    st.bar_chart(
        pd.DataFrame({"Scrap (tonnes)": [tot.scrap_kg_none / 1000, tot.scrap_kg_bfd / 1000, tot.scrap_kg_cpsat / 1000]},
                     index=["No optimisation", "Standard plan", "Advanced plan"]),
        y_label="Scrap (tonnes)")

    # ------------------------------------------------------------------ how good is it
    st.subheader("How good is the Advanced plan?")
    if stats["proven_optimal"].all():
        st.success("Proven optimal for every diameter: no cutting plan can use fewer stock bars than the Advanced plan "
                   "(for these lengths and settings).")
    else:
        st.warning("Not proven optimal for every diameter. The optimiser stopped at its time limit or the search did not "
                   "finish; try a longer time limit.")
    st.dataframe(
        stats, hide_index=True, width="stretch",
        column_config={
            "dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "bfd_bars": st.column_config.NumberColumn("Bars (Standard)", format="%d", help="Starting plan given to the optimiser."),
            "cpsat_bars": st.column_config.NumberColumn("Bars (Advanced)", format="%d", help="Stock bars in the final plan."),
            "lower_bound_bars": st.column_config.NumberColumn("Minimum possible bars", format="%d", help="Proven lower bound: no plan can use fewer bars."),
            "status": st.column_config.TextColumn("Status", help="OPTIMAL = best found and proven; FEASIBLE = good plan, not proven."),
            "proven_optimal": st.column_config.CheckboxColumn("Proven optimal", help="Ticked when Advanced bars equal the minimum possible."),
            "lengths": st.column_config.NumberColumn("Different lengths", format="%d", help="Distinct piece lengths in this diameter."),
            "patterns_generated": st.column_config.NumberColumn("Patterns tried", format="%d", help="Cutting patterns the optimiser chose between."),
            "search_rounds": st.column_config.NumberColumn("Search rounds", format="%d", help="Rounds of the pattern search."),
            "seconds": st.column_config.NumberColumn("Seconds", format="%.1f", help="Solving time for this diameter."),
        })

    if not opt.oversize.empty:
        st.warning("Not planned (longer than the stock bar): "
                   + ", ".join(f"mark {r.mark} ({r.length_mm:,} mm)" for r in opt.oversize.itertuples()))

    # ------------------------------------------------------------------ Advanced plan data
    st.subheader("Advanced plan: cutting patterns (identical bars grouped)")
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

    st.subheader("Advanced plan: bar-by-bar cutting plan")
    only = st.radio("Show", ["All bars", "Only bars with scrap"], horizontal=True, key="cpsat_filter")
    plan = opt.plan if only == "All bars" else opt.plan[opt.plan["left_mm"] > 0]
    st.write(f"{len(plan):,} bars")
    st.dataframe(
        _plan_view(plan), hide_index=True, width="stretch", height=350,
        column_config={
            "Bar #": st.column_config.NumberColumn("Bar #", format="%d", help="Number of this stock bar in the plan."),
            "Dia": st.column_config.TextColumn("Dia", help="Bar diameter."),
            "Cuts from this bar": st.column_config.TextColumn("Cuts from this bar", help="Marks and lengths cut from this bar. xN means N pieces of that mark."),
            "Pieces": st.column_config.NumberColumn("Pieces", format="%d", help="How many pieces are cut from this bar."),
            "Used (mm)": st.column_config.NumberColumn("Used (mm)", format="%d", help="Total length of the pieces on this bar."),
            "Scrap (mm)": st.column_config.NumberColumn("Scrap (mm)", format="%d", help="Stock length minus used: what is left over."),
            "Scrap (kg)": st.column_config.NumberColumn("Scrap (kg)", format="%.2f", help="That scrap converted to weight."),
        })

    d1, d2 = st.columns(2)
    d1.download_button("Download Advanced cutting plan (CSV)", _plan_view(opt.plan).to_csv(index=False).encode(),
                       file_name="advanced_cutting_plan.csv", mime="text/csv")
    d2.download_button("Download comparison (CSV)", cmp_.to_csv(index=False).encode(),
                       file_name="standard_vs_advanced_comparison.csv", mime="text/csv")