"""
app.py — Supplier Receiving Inspection Analyzer (Streamlit UI).

Run locally:  streamlit run app.py
All calculations live in sri_core.py (unit-tested). Data is synthetic.
"""

from __future__ import annotations

from datetime import date

import altair as alt
import pandas as pd
import streamlit as st

import sri_core as s

st.set_page_config(page_title="Receiving Inspection Analyzer", page_icon="📦", layout="wide")

BLUE, RED, GREEN, AMBER, GRAY = "#4C9AFF", "#F25C54", "#2FBF71", "#F2A541", "#8A94A6"
FLOW_COLORS = [GREEN, AMBER, RED, "#B07AF2", BLUE]


@st.cache_data
def demo_data(today: date):
    lots = s.generate_lots(today=today)
    return lots, s.build_movements(lots, integrity_demo=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.title("📦 Receiving Inspection")
st.sidebar.caption("Supplier incoming-inspection analytics on SAP MM/QM-style data.")

source = st.sidebar.radio("Data", ["Demo data (synthetic)", "Upload inspection lots (CSV)"])
if source.startswith("Demo"):
    lots, mvts = demo_data(date.today())
else:
    template = s.generate_lots(today=date.today()).head(5)[s.REQUIRED_COLUMNS + s.OPTIONAL_COLUMNS]
    st.sidebar.download_button("Download CSV template", template.to_csv(index=False),
                               "inspection_lots_template.csv", "text/csv")
    up = st.sidebar.file_uploader("Inspection lots CSV", type=["csv"])
    if up is None:
        st.info("Upload an inspection-lot CSV, or switch to the demo data. The template shows the required columns.")
        st.stop()
    try:
        lots = s.validate_lots(pd.read_csv(up))
    except s.DataError as ex:
        st.error(str(ex))
        st.stop()
    except Exception as ex:  # noqa: BLE001 — show any parse problem to the user
        st.error(f"Couldn't read the file: {ex}")
        st.stop()
    mvts = s.build_movements(lots)

st.sidebar.subheader("Filters")
suppliers = sorted(lots["supplier_name"].unique())
pick = st.sidebar.multiselect("Suppliers", suppliers, placeholder="All suppliers")
dmin, dmax = lots["gr_date"].min().date(), lots["gr_date"].max().date()
rng = st.sidebar.date_input("GR date range", (dmin, dmax), min_value=dmin, max_value=dmax)

st.sidebar.subheader("Action thresholds")
th = s.Thresholds(
    scar_rejects=st.sidebar.number_input("SCAR: rejected lots in window", 1, 20, 3),
    scar_window_days=st.sidebar.number_input("SCAR: window (days)", 30, 365, 90, step=15),
    ppm_target=st.sidebar.number_input("PPM target", 0, 200_000, 15_000, step=1_000),
    skip_lot_run=st.sidebar.number_input("Skip-lot: consecutive accepted lots", 3, 50, 10),
    cycle_target_days=st.sidebar.number_input("Inspection cycle target (days)", 1, 30, 3),
)
st.sidebar.divider()
st.sidebar.caption("All suppliers, materials, POs and people are fictional.")

view = lots.copy()
if pick:
    view = view[view["supplier_name"].isin(pick)]
if isinstance(rng, tuple) and len(rng) == 2:
    view = view[(view["gr_date"].dt.date >= rng[0]) & (view["gr_date"].dt.date <= rng[1])]
if view.empty:
    st.warning("No inspection lots match these filters.")
    st.stop()
view_mvts = mvts[mvts["inspection_lot"].isin(view["inspection_lot"])]
today = pd.Timestamp(date.today())

# ---------------------------------------------------------------------------
# Header + KPIs
# ---------------------------------------------------------------------------
st.title("Supplier Receiving Inspection Analyzer")
st.caption("GR (101) → QI stock → inspection lot (origin 01) → usage decision → release (321) "
           "or block (350) → return to vendor (122) / scrap (551)")

k = s.kpis(view, today)
c = st.columns(6)
c[0].metric("Lots inspected", f"{k['decided']:,}", help="Lots with a usage decision")
c[1].metric("Lot acceptance", f"{k['lar']:.1f}%")
c[2].metric("Rejected PPM", f"{k['ppm']:,.0f}", help="Rejected quantity per million received")
c[3].metric("Median cycle", f"{k['median_cycle']:.0f} d", help="GR date to usage decision")
def lots_label(n: int) -> str:
    return f"{n} lot" if n == 1 else f"{n} lots"


c[4].metric(f"QI stock · {lots_label(k['qi_lots'])}", f"${k['qi_value']:,.0f}",
            help=f"Awaiting inspection; oldest lot {k['oldest_qi_days']} days")
c[5].metric(f"Blocked · {lots_label(k['blocked_lots'])}", f"${k['blocked_value']:,.0f}",
            help="Rejected stock still waiting for a disposition (RTV or scrap)")

scorecard = s.supplier_scorecard(view, th, today)
flagged = scorecard[scorecard["SCAR flag"] != ""]
if len(flagged):
    # Amber, not red: this is a supplier-quality flag, not an app error.
    st.warning("**Supplier alert: SCAR recommended.** " +
               " · ".join(f"**{name}** ({why})" for name, why in zip(flagged["Supplier"], flagged["SCAR flag"])),
               icon="⚠️")

tabs = st.tabs(["Overview", "Supplier scorecard", "Defects", "Stock & aging", "Movements (MB51)", "About"])

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tabs[0]:
    a, b = st.columns([3, 2])
    trend = s.monthly_trend(view)
    a.markdown("**Lot acceptance rate by month**")
    base = alt.Chart(trend).encode(x=alt.X("month:T", title=None, axis=alt.Axis(format="%b %y")))
    a.altair_chart(
        (base.mark_bar(color="#2A3550", opacity=0.9).encode(
            y=alt.Y("lots:Q", title="Lots"), tooltip=["month:T", "lots:Q"])
         + base.mark_line(color=GREEN, point=True, strokeWidth=2.5).encode(
            y=alt.Y("acceptance_pct:Q", title="Acceptance %", scale=alt.Scale(domain=[70, 100])),
            tooltip=[alt.Tooltip("acceptance_pct:Q", format=".1f", title="Acceptance %")]))
        .resolve_scale(y="independent").properties(height=300),
        width="stretch")

    flow = s.flow_summary(view)
    b.markdown("**Where received quantity went**")
    b.altair_chart(
        alt.Chart(flow).mark_bar().encode(
            y=alt.Y("outcome:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
            x=alt.X("pct:Q", title="% of received quantity", scale=alt.Scale(type="symlog")),
            color=alt.Color("outcome:N", scale=alt.Scale(domain=list(flow["outcome"]), range=FLOW_COLORS),
                            legend=None),
            tooltip=["outcome", alt.Tooltip("qty:Q", format=","), alt.Tooltip("pct:Q", format=".2f")])
        .properties(height=300),
        width="stretch")
    b.caption("Log scale so the small reject flows stay visible next to the releases.")

# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------
with tabs[1]:
    st.dataframe(
        scorecard.style.format({"Acceptance %": "{:.1f}", "PPM": "{:,.0f}", "Median cycle (d)": "{:.0f}",
                                "On-time %": "{:.0f}"}, na_rep="—")
        .apply(lambda r: ["background-color: rgba(242,92,84,.18)" if r["SCAR flag"] else
                          ("background-color: rgba(47,191,113,.14)" if r["Skip-lot candidate"] else "")] * len(r),
               axis=1),
        hide_index=True, width="stretch",
        column_config={"SCAR flag": st.column_config.TextColumn(width="large"),
                       "Skip-lot candidate": st.column_config.CheckboxColumn("Skip-lot")})
    x, y = st.columns(2)
    x.markdown("**🔴 SCAR candidates**")
    if len(flagged):
        x.dataframe(flagged[["Supplier", "SCAR flag", "Top defect"]], hide_index=True, width="stretch")
    else:
        x.write("None at current thresholds.")
    skip = scorecard[scorecard["Skip-lot candidate"]]
    y.markdown("**🟢 Skip-lot candidates** (reduced inspection)")
    if len(skip):
        y.dataframe(skip[["Supplier", "Accept run", "PPM"]], hide_index=True, width="stretch",
                    column_config={"PPM": st.column_config.NumberColumn(format="%.0f")})
    else:
        y.write("None at current thresholds.")
    st.caption(f"SCAR flag: ≥ {th.scar_rejects} rejected lots in {th.scar_window_days} days, or PPM above "
               f"{th.ppm_target:,.0f}. Skip-lot: last {th.skip_lot_run}+ lots accepted, no recent rejects, no flag. "
               f"On-time: usage decision within {th.cycle_target_days:g} days of GR.")

# ---------------------------------------------------------------------------
# Defects
# ---------------------------------------------------------------------------
with tabs[2]:
    by_qty = st.toggle("Weight by rejected quantity (off = count of rejected lots)", value=False)
    p = s.defect_pareto(view, by_qty=by_qty)
    a, b = st.columns([3, 2])
    a.markdown("**Defect Pareto**")
    base = alt.Chart(p).encode(x=alt.X("defect_category:N", sort=None, title=None, axis=alt.Axis(labelAngle=0)))
    a.altair_chart(
        (base.mark_bar(color=RED).encode(y=alt.Y("count:Q", title="Rejected qty" if by_qty else "Rejected lots"),
                                         tooltip=["defect_category", "count"])
         + base.mark_line(color=AMBER, point=True).encode(
            y=alt.Y("cum_pct:Q", title="Cumulative %", scale=alt.Scale(domain=[0, 100])),
            tooltip=[alt.Tooltip("cum_pct:Q", format=".0f")]))
        .resolve_scale(y="independent").properties(height=300),
        width="stretch")

    rej = view[view["usage_decision"] == s.UD_REJECT]
    heat = rej.groupby(["supplier_name", "defect_category"]).size().rename("lots").reset_index()
    b.markdown("**Rejected lots by supplier and defect**")
    if heat.empty:
        b.info("No rejected lots in this selection.")
    else:
        b.altair_chart(
            alt.Chart(heat).mark_rect().encode(
                x=alt.X("defect_category:N", title=None, sort=s.DEFECT_CATEGORIES,
                        axis=alt.Axis(labelAngle=-30, labelOverlap=False)),
                y=alt.Y("supplier_name:N", title=None, axis=alt.Axis(labelLimit=200)),
                color=alt.Color("lots:Q", scale=alt.Scale(scheme="reds"), legend=None),
                tooltip=["supplier_name", "defect_category", "lots"])
            + alt.Chart(heat).mark_text(color="white", fontWeight="bold").encode(
                x=alt.X("defect_category:N", sort=s.DEFECT_CATEGORIES), y="supplier_name:N", text="lots:Q")
            .properties(height=300),
            width="stretch")
    st.markdown("**Rejected lots**")
    st.dataframe(rej[["inspection_lot", "gr_date", "supplier_name", "material", "material_desc", "qty_received",
                      "qty_rejected", "defect_category", "disposition", "inspector"]]
                 .sort_values("gr_date", ascending=False),
                 hide_index=True, width="stretch",
                 column_config={"inspection_lot": "Insp. lot", "gr_date": st.column_config.DateColumn("GR date"),
                                "supplier_name": "Supplier", "material": "Material", "material_desc": "Description",
                                "qty_received": "Received", "qty_rejected": "Rejected",
                                "defect_category": "Defect", "disposition": "Disposition", "inspector": "Inspector"})

# ---------------------------------------------------------------------------
# Stock & aging
# ---------------------------------------------------------------------------
with tabs[3]:
    ag = s.aging(view, today)
    if ag.empty:
        st.success("Nothing waiting: no open QI lots or blocked stock without a disposition.")
    else:
        a, b = st.columns([2, 3])
        bucket = ag.groupby(["bucket", "stock"], observed=False)["value_usd"].sum().reset_index()
        a.markdown("**Value waiting, by age**")
        a.altair_chart(
            alt.Chart(bucket).mark_bar().encode(
                x=alt.X("bucket:N", title=None, sort=["0–3 d", "4–7 d", "8–14 d", "15–30 d", "30+ d"],
                        axis=alt.Axis(labelAngle=0)),
                y=alt.Y("value_usd:Q", title="USD"),
                color=alt.Color("stock:N", scale=alt.Scale(range=[RED, BLUE]),
                                legend=alt.Legend(orient="top", direction="vertical", title=None, labelLimit=300)),
                tooltip=["stock", "bucket", alt.Tooltip("value_usd:Q", format="$,.0f")])
            .properties(height=300),
            width="stretch")
        b.markdown("**Open items, oldest first**")
        b.dataframe(ag[["inspection_lot", "stock", "supplier_name", "material", "qty", "unit", "value_usd",
                        "age_days"]].rename(columns={"inspection_lot": "Insp. lot", "stock": "Stock",
                                                     "supplier_name": "Supplier", "material": "Material",
                                                     "qty": "Qty", "unit": "Unit"}),
                    hide_index=True, width="stretch",
                    column_config={"value_usd": st.column_config.NumberColumn("Value", format="$%.0f"),
                                   "age_days": st.column_config.ProgressColumn(
                                       "Age (days)", min_value=0, max_value=max(30, int(ag["age_days"].max())),
                                       format="%d")})

# ---------------------------------------------------------------------------
# Movements + reconciliation
# ---------------------------------------------------------------------------
with tabs[4]:
    st.markdown("**Material document list** — derived from the inspection lots, like an MB51 report.")
    types = st.multiselect("Movement types", sorted(s.MOVEMENT_TEXT),
                           format_func=lambda m: f"{m} – {s.MOVEMENT_TEXT[m]}", placeholder="All movement types")
    mv = view_mvts if not types else view_mvts[view_mvts["movement_type"].isin(types)]
    st.dataframe(mv.sort_values("posting_date", ascending=False), hide_index=True, width="stretch",
                 column_config={"material_doc": "Mat. doc", "posting_date": st.column_config.DateColumn("Posting date"),
                                "movement_type": "Mvt", "movement_text": "Movement", "material": "Material",
                                "supplier_name": "Supplier", "inspection_lot": "Insp. lot", "quantity": "Qty",
                                "unit": "Unit", "amount_usd": st.column_config.NumberColumn("Amount", format="$%.2f"),
                                "stock_from": "From", "stock_to": "To"})

    st.markdown("**Reconciliation check** — rebuilds each lot from its movements and compares the quantities.")
    issues = s.reconcile(view, view_mvts)
    if issues.empty:
        st.success("All lots balance: GR = released + blocked, and nothing returned or scrapped beyond what was blocked.")
    else:
        st.warning(f"{len(issues)} lot(s) don't balance. Investigate the posting before trusting stock figures.")
        st.dataframe(issues, hide_index=True, width="stretch")
        if source.startswith("Demo"):
            st.caption("The demo data contains one deliberately short 321 posting to show this check working.")

# ---------------------------------------------------------------------------
# About
# ---------------------------------------------------------------------------
with tabs[5]:
    st.markdown("""
**What this is.** An analytics app for incoming (receiving) inspection, modeled on how SAP MM and QM
record it: a goods receipt posts stock into quality inspection, an inspection lot is created, and the
usage decision releases the stock or blocks it for return or scrap.

**Movement types used**

| Mvt | Meaning |
|---|---|
| 101 | Goods receipt for purchase order, into QI stock |
| 321 | Transfer QI stock → unrestricted (accepted) |
| 350 | Transfer QI stock → blocked (rejected) |
| 122 | Return delivery to vendor |
| 551 | Withdrawal for scrapping |

**Metrics**
- **Lot acceptance rate:** decided lots accepted with zero rejections.
- **Rejected PPM:** rejected quantity ÷ received quantity × 1,000,000.
- **Cycle time:** days from goods receipt to usage decision.
- **SCAR candidate:** too many rejected lots in the window, or PPM above target.
- **Skip-lot candidate:** a run of accepted lots with no recent rejects and no open flag.
  Reduced inspection still needs QE approval under your quality plan.

**Data notice.** All suppliers, materials, PO numbers, lot numbers and inspectors are fictional.
This is a portfolio project. It is not affiliated with SAP and does not connect to an SAP system.
""")
