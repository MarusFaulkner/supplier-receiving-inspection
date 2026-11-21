"""
sri_core.py — Supplier Receiving Inspection analysis engine.

Models the incoming-inspection flow the way SAP MM/QM records it:

    Goods receipt (mvt 101) -> quality inspection (QI) stock
        -> inspection lot (origin 01: goods receipt for purchase order)
        -> usage decision
             accept  -> mvt 321  QI stock      -> unrestricted
             reject  -> mvt 350  QI stock      -> blocked
                         -> mvt 122  return to vendor
                         -> mvt 551  scrap
                         -> (pending disposition: stays blocked)
        -> no decision yet: stays in QI stock

Everything here is SYNTHETIC. Supplier names, materials, PO numbers and
inspectors are invented; any resemblance to real companies or people is
coincidental. No UI code lives in this module, so it can be unit-tested.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reference data (all fictional)
# ---------------------------------------------------------------------------

#: Check-sheet categories used for rejections.
DEFECT_CATEGORIES = [
    "Packaging",       # damaged packaging, missing/incorrect ID labels
    "Appearance",      # rust, dents, burrs, paint runs, missing components
    "Joint",           # weld, solder, braze, adhesive or bonding defects
    "Dimensional",     # out of tolerance against the drawing
    "Documentation",   # missing or wrong test reports / certificates
]

#: id, name, quality profile (base reject probability, defect weights)
SUPPLIERS = [
    ("V-10001", "Kestrel Coil Components",   0.03, [1, 2, 3, 4, 1]),
    ("V-10002", "Marlowe Insulation Works",  0.05, [3, 3, 0, 2, 2]),
    ("V-10003", "Oakridge Fastener Supply",  0.02, [1, 2, 0, 5, 1]),
    ("V-10004", "Tessaro Castings",          0.16, [1, 5, 1, 4, 1]),  # chronic problems
    ("V-10005", "Brightwater Gasket Co.",    0.04, [2, 3, 0, 3, 1]),
    ("V-10006", "Halvorsen Copper & Wire",   0.00, [1, 1, 1, 1, 1]),  # near-perfect
    ("V-10007", "Pinecrest Paint Systems",   0.07, [2, 6, 0, 0, 3]),
    ("V-10008", "Vantor Electrical Parts",   0.05, [1, 2, 3, 2, 4]),
]

#: material, description, unit, unit price (USD), supplier id
MATERIALS = [
    ("MAT-WI-1040", "Winding insulation board, 4 mm",   "EA",  38.50, "V-10002"),
    ("MAT-WI-1041", "Insulation spacer kit",             "SET", 112.00, "V-10002"),
    ("MAT-CU-2210", "Copper conductor, CTC 6x12",        "KG",  14.20, "V-10006"),
    ("MAT-CU-2215", "Copper busbar 10x100",              "EA",  96.00, "V-10006"),
    ("MAT-FS-3301", "Hex bolt M16x60, 8.8 HDG",          "EA",   1.85, "V-10003"),
    ("MAT-FS-3302", "Stud M20x120, stainless",           "EA",   6.40, "V-10003"),
    ("MAT-CS-4410", "Radiator valve body, cast",         "EA", 145.00, "V-10004"),
    ("MAT-CS-4415", "Bushing turret flange, cast",       "EA", 389.00, "V-10004"),
    ("MAT-GK-5501", "Nitrile gasket set, cover",         "SET",  58.00, "V-10005"),
    ("MAT-GK-5502", "Cork-nitrile gasket, 600 mm",       "EA",  22.75, "V-10005"),
    ("MAT-PT-6601", "Epoxy primer, gray (5 gal)",        "EA", 240.00, "V-10007"),
    ("MAT-PT-6602", "Polyurethane topcoat ANSI 70",      "EA", 310.00, "V-10007"),
    ("MAT-EL-7701", "Winding temperature indicator",     "EA", 865.00, "V-10008"),
    ("MAT-EL-7702", "Pressure relief device",            "EA", 540.00, "V-10008"),
    ("MAT-CL-8801", "Lead exit coil assembly",           "EA", 210.00, "V-10001"),
    ("MAT-CL-8802", "Tap changer lead set",              "SET", 175.00, "V-10001"),
]

INSPECTORS = ["J. Avery", "R. Delgado", "T. Nakamura", "S. Whitfield", "K. Osei"]

#: Columns an uploaded inspection-lot file must have.
REQUIRED_COLUMNS = [
    "inspection_lot", "po_number", "supplier_id", "supplier_name", "material",
    "material_desc", "gr_date", "qty_received", "unit", "unit_price",
    "qty_rejected", "defect_category", "usage_decision", "ud_date", "disposition",
]

UD_ACCEPT, UD_REJECT, UD_PENDING = "A", "R", "PENDING"
DISPOSITIONS = {"", "RTV", "SCRAP", "PENDING"}

MOVEMENT_TEXT = {
    101: "GR for PO into QI stock",
    321: "QI stock to unrestricted",
    350: "QI stock to blocked",
    122: "Return delivery to vendor",
    551: "Withdrawal for scrapping",
}


class DataError(ValueError):
    """Raised when uploaded data can't be analyzed."""


# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------

def generate_lots(n_months: int = 12, seed: int = 42, today: date | None = None) -> pd.DataFrame:
    """Generate a synthetic inspection-lot history ending today."""
    rng = np.random.default_rng(seed)
    today = today or date.today()
    start = today - timedelta(days=30 * n_months)
    sup = {s[0]: s for s in SUPPLIERS}
    rows = []
    lot_no = 10_000_000_000
    po_no = 4_500_100_000

    for mat, desc, unit, price, sid in MATERIALS:
        _, sname, p_rej, weights = sup[sid]
        n_lots = int(rng.integers(28, 46))
        days = np.sort(rng.integers(0, (today - start).days + 1, n_lots))
        for d in days:
            gr = start + timedelta(days=int(d))
            lot_no += int(rng.integers(1, 40))
            po_no += int(rng.integers(1, 25))
            qty = int(rng.choice([10, 20, 25, 50, 100, 200, 500])) if price < 50 else int(rng.integers(2, 25))
            age = (today - gr).days

            # Recent receipts may still be waiting for a decision.
            pending = (age <= 6 and rng.random() < 0.55) or (age <= 35 and rng.random() < 0.03)
            # Tessaro gets worse in the most recent quarter (drives a SCAR flag).
            p = p_rej * (1.6 if sid == "V-10004" and age < 90 else 1.0)
            rejected = (not pending) and rng.random() < p

            qty_rej, cat, ud, ud_date, disp = 0, "", UD_PENDING, None, ""
            if not pending:
                cycle = int(rng.gamma(2.0, 1.3)) + (int(rng.integers(3, 9)) if rejected else 0)
                ud_date = min(gr + timedelta(days=cycle), today)
                if rejected:
                    qty_rej = max(1, int(round(qty * rng.uniform(0.02, 0.25))))
                    cat = DEFECT_CATEGORIES[int(rng.choice(5, p=np.array(weights) / sum(weights)))]
                    ud = UD_REJECT
                    disp = "PENDING" if (today - ud_date).days <= 45 and rng.random() < 0.9 else \
                        str(rng.choice(["RTV", "SCRAP"], p=[0.75, 0.25]))
                else:
                    ud = UD_ACCEPT

            rows.append({
                "inspection_lot": f"0{lot_no}",
                "lot_origin": "01",
                "plant": "1000",
                "po_number": str(po_no),
                "supplier_id": sid, "supplier_name": sname,
                "material": mat, "material_desc": desc,
                "gr_date": gr, "qty_received": qty, "unit": unit, "unit_price": price,
                "qty_rejected": qty_rej, "defect_category": cat,
                "usage_decision": ud, "ud_date": ud_date, "disposition": disp,
                "inspector": str(rng.choice(INSPECTORS)),
            })

    df = pd.DataFrame(rows).sort_values("gr_date").reset_index(drop=True)
    df["gr_date"] = pd.to_datetime(df["gr_date"])
    df["ud_date"] = pd.to_datetime(df["ud_date"])
    return df


def build_movements(lots: pd.DataFrame, integrity_demo: bool = False) -> pd.DataFrame:
    """
    Derive an MB51-style material document list from inspection lots.

    With integrity_demo=True one lot gets a deliberately short 321 posting,
    so the reconciliation check has something to find.
    """
    rows = []
    doc = 5_000_000_000
    demo_lot = None
    if integrity_demo:
        decided = lots[lots["usage_decision"] == UD_ACCEPT]
        demo_lot = decided.iloc[len(decided) // 2]["inspection_lot"] if len(decided) else None

    for r in lots.itertuples(index=False):
        def mv(mvt, qty, when, stock_from="", stock_to=""):
            nonlocal doc
            doc += 1
            rows.append({
                "material_doc": str(doc), "posting_date": when, "movement_type": mvt,
                "movement_text": MOVEMENT_TEXT[mvt], "material": r.material,
                "supplier_name": r.supplier_name, "inspection_lot": r.inspection_lot,
                "quantity": qty, "unit": r.unit, "amount_usd": round(qty * r.unit_price, 2),
                "stock_from": stock_from, "stock_to": stock_to,
            })

        mv(101, r.qty_received, r.gr_date, "", "QI")
        if r.usage_decision == UD_PENDING:
            continue
        accepted = r.qty_received - r.qty_rejected
        if accepted:
            short = 1 if (demo_lot is not None and r.inspection_lot == demo_lot and accepted > 1) else 0
            mv(321, accepted - short, r.ud_date, "QI", "Unrestricted")
        if r.qty_rejected:
            mv(350, r.qty_rejected, r.ud_date, "QI", "Blocked")
            later = r.ud_date + pd.Timedelta(days=3)
            if r.disposition == "RTV":
                mv(122, r.qty_rejected, later, "Blocked", "Vendor")
            elif r.disposition == "SCRAP":
                mv(551, r.qty_rejected, later, "Blocked", "Scrap")

    mvts = pd.DataFrame(rows)
    mvts["posting_date"] = pd.to_datetime(mvts["posting_date"])
    return mvts.sort_values(["posting_date", "material_doc"]).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_lots(df: pd.DataFrame) -> pd.DataFrame:
    """Check an uploaded lot file; return a cleaned copy or raise DataError."""
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise DataError(f"Missing column(s): {', '.join(missing)}.")
    d = df.copy()
    for c in ("inspection_lot", "po_number", "supplier_id", "supplier_name", "material",
              "material_desc", "unit", "defect_category", "usage_decision", "disposition"):
        d[c] = d[c].fillna("").astype(str).str.strip()
    d["usage_decision"] = d["usage_decision"].str.upper()
    d["disposition"] = d["disposition"].str.upper()
    for c in ("qty_received", "qty_rejected", "unit_price"):
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d["gr_date"] = pd.to_datetime(d["gr_date"], errors="coerce")
    d["ud_date"] = pd.to_datetime(d["ud_date"], errors="coerce")

    problems = []
    def flag(mask, msg):
        if mask.any():
            rows = (d.index[mask] + 2).tolist()[:8]   # +2 = spreadsheet row incl. header
            problems.append(f"{msg} (rows {rows}{'…' if mask.sum() > 8 else ''})")

    flag(d["inspection_lot"].eq(""), "Blank inspection lot")
    flag(d["inspection_lot"].duplicated(), "Duplicate inspection lot")
    flag(d["gr_date"].isna(), "Missing or invalid GR date")
    flag(d["qty_received"].isna() | (d["qty_received"] <= 0), "Received quantity must be > 0")
    flag(d["qty_rejected"].isna() | (d["qty_rejected"] < 0), "Rejected quantity must be ≥ 0")
    flag(d["qty_rejected"] > d["qty_received"], "Rejected quantity exceeds received")
    flag(d["unit_price"].isna() | (d["unit_price"] < 0), "Unit price must be ≥ 0")
    flag(~d["usage_decision"].isin([UD_ACCEPT, UD_REJECT, UD_PENDING]), "Usage decision must be A, R or PENDING")
    decided = d["usage_decision"].isin([UD_ACCEPT, UD_REJECT])
    flag(decided & d["ud_date"].isna(), "Decided lot is missing its UD date")
    flag(decided & (d["ud_date"] < d["gr_date"]), "UD date is before GR date")
    flag((d["usage_decision"] == UD_REJECT) & (d["qty_rejected"] <= 0), "Rejected lot has no rejected quantity")
    flag((d["usage_decision"] == UD_ACCEPT) & (d["qty_rejected"] > 0), "Accepted lot has a rejected quantity")
    flag((d["usage_decision"] == UD_REJECT) & ~d["defect_category"].isin(DEFECT_CATEGORIES),
         f"Rejected lot needs a defect category ({', '.join(DEFECT_CATEGORIES)})")
    flag(~d["disposition"].isin(DISPOSITIONS), "Disposition must be RTV, SCRAP, PENDING or blank")
    if problems:
        raise DataError("Fix these issues and upload again:\n- " + "\n- ".join(problems))
    return d.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

@dataclass
class Thresholds:
    scar_rejects: int = 3          # rejected lots in the window that trigger a SCAR flag
    scar_window_days: int = 90
    ppm_target: float = 15000.0    # rejected parts per million received
    skip_lot_run: int = 10         # consecutive accepted lots for skip-lot eligibility
    cycle_target_days: float = 3.0


def kpis(lots: pd.DataFrame, today: pd.Timestamp | None = None) -> dict:
    today = pd.Timestamp(today or date.today())
    decided = lots[lots["usage_decision"] != UD_PENDING]
    pend = lots[lots["usage_decision"] == UD_PENDING]
    blocked = lots[(lots["usage_decision"] == UD_REJECT) & (lots["disposition"] == "PENDING")]
    cycle = (decided["ud_date"] - decided["gr_date"]).dt.days
    return {
        "lots": len(lots),
        "decided": len(decided),
        "lar": (decided["usage_decision"] == UD_ACCEPT).mean() * 100 if len(decided) else float("nan"),
        "ppm": decided["qty_rejected"].sum() / decided["qty_received"].sum() * 1e6 if len(decided) else float("nan"),
        "median_cycle": float(cycle.median()) if len(cycle) else float("nan"),
        "qi_value": float((pend["qty_received"] * pend["unit_price"]).sum()),
        "qi_lots": len(pend),
        "blocked_value": float((blocked["qty_rejected"] * blocked["unit_price"]).sum()),
        "blocked_lots": len(blocked),
        "oldest_qi_days": int((today - pend["gr_date"]).dt.days.max()) if len(pend) else 0,
    }


def supplier_scorecard(lots: pd.DataFrame, th: Thresholds = Thresholds(),
                       today: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per supplier with acceptance, PPM, cycle time and action flags."""
    today = pd.Timestamp(today or date.today())
    out = []
    for (sid, name), g in lots.groupby(["supplier_id", "supplier_name"]):
        dec = g[g["usage_decision"] != UD_PENDING].sort_values("gr_date")
        rej = dec[dec["usage_decision"] == UD_REJECT]
        recent_rej = rej[rej["gr_date"] >= today - pd.Timedelta(days=th.scar_window_days)]
        ppm = dec["qty_rejected"].sum() / dec["qty_received"].sum() * 1e6 if len(dec) else np.nan
        cyc = (dec["ud_date"] - dec["gr_date"]).dt.days
        # Current run of consecutive accepted lots (most recent backwards).
        run = 0
        for ud in dec["usage_decision"].iloc[::-1]:
            if ud != UD_ACCEPT:
                break
            run += 1
        reasons = []
        if len(recent_rej) >= th.scar_rejects:
            reasons.append(f"{len(recent_rej)} rejects in {th.scar_window_days} d")
        if ppm > th.ppm_target:
            reasons.append(f"PPM {ppm:,.0f} > {th.ppm_target:,.0f}")
        top = rej["defect_category"].mode()
        out.append({
            "Supplier ID": sid, "Supplier": name, "Lots": len(g), "Decided": len(dec),
            "Rejected lots": len(rej),
            "Acceptance %": (len(dec) - len(rej)) / len(dec) * 100 if len(dec) else np.nan,
            "PPM": ppm,
            "Median cycle (d)": float(cyc.median()) if len(cyc) else np.nan,
            "On-time %": (cyc <= th.cycle_target_days).mean() * 100 if len(cyc) else np.nan,
            "Top defect": top.iloc[0] if len(top) else "—",
            "Accept run": run,
            "SCAR flag": "; ".join(reasons),
            # Reduced inspection only for a clean, recent record with no open action flag.
            "Skip-lot candidate": run >= th.skip_lot_run and len(recent_rej) == 0 and not reasons,
            "_severity": len(reasons) * 10 + len(recent_rej),
        })
    sc = pd.DataFrame(out).sort_values(["_severity", "PPM"], ascending=[False, False])
    return sc.drop(columns="_severity").reset_index(drop=True)


def defect_pareto(lots: pd.DataFrame, by_qty: bool = True) -> pd.DataFrame:
    rej = lots[lots["usage_decision"] == UD_REJECT]
    s = rej.groupby("defect_category")["qty_rejected"].sum() if by_qty else rej["defect_category"].value_counts()
    s = s.reindex(DEFECT_CATEGORIES, fill_value=0).sort_values(ascending=False)
    p = s.rename("count").reset_index().rename(columns={"index": "defect_category"})
    total = p["count"].sum()
    p["cum_pct"] = p["count"].cumsum() / total * 100 if total else 0.0
    return p


def monthly_trend(lots: pd.DataFrame) -> pd.DataFrame:
    dec = lots[lots["usage_decision"] != UD_PENDING].copy()
    dec["month"] = dec["gr_date"].dt.to_period("M").dt.to_timestamp()
    g = dec.groupby("month").agg(lots=("inspection_lot", "count"),
                                 accepted=("usage_decision", lambda s: (s == UD_ACCEPT).sum()),
                                 rejected_qty=("qty_rejected", "sum"),
                                 received_qty=("qty_received", "sum")).reset_index()
    g["acceptance_pct"] = g["accepted"] / g["lots"] * 100
    g["ppm"] = g["rejected_qty"] / g["received_qty"] * 1e6
    return g


def aging(lots: pd.DataFrame, today: pd.Timestamp | None = None) -> pd.DataFrame:
    """Open QI lots and blocked stock awaiting disposition, with age buckets."""
    today = pd.Timestamp(today or date.today())
    qi = lots[lots["usage_decision"] == UD_PENDING].assign(
        stock="QI (awaiting inspection)", qty=lambda d: d["qty_received"],
        age_days=lambda d: (today - d["gr_date"]).dt.days)
    bl = lots[(lots["usage_decision"] == UD_REJECT) & (lots["disposition"] == "PENDING")].assign(
        stock="Blocked (awaiting disposition)", qty=lambda d: d["qty_rejected"],
        age_days=lambda d: (today - d["ud_date"]).dt.days)
    a = pd.concat([qi, bl], ignore_index=True)
    if a.empty:
        return a
    a["value_usd"] = a["qty"] * a["unit_price"]
    a["bucket"] = pd.cut(a["age_days"], [-1, 3, 7, 14, 30, 10_000],
                         labels=["0–3 d", "4–7 d", "8–14 d", "15–30 d", "30+ d"])
    return a.sort_values("age_days", ascending=False).reset_index(drop=True)


def flow_summary(lots: pd.DataFrame) -> pd.DataFrame:
    """Where received quantity ended up."""
    acc = (lots["qty_received"] - lots["qty_rejected"]).where(lots["usage_decision"] != UD_PENDING, 0).sum()
    rej = lots[lots["usage_decision"] == UD_REJECT]
    out = [
        ("Released to unrestricted (321)", acc),
        ("Returned to vendor (122)", rej.loc[rej["disposition"] == "RTV", "qty_rejected"].sum()),
        ("Scrapped (551)", rej.loc[rej["disposition"] == "SCRAP", "qty_rejected"].sum()),
        ("Blocked, awaiting disposition", rej.loc[rej["disposition"] == "PENDING", "qty_rejected"].sum()),
        ("In QI stock, awaiting inspection", lots.loc[lots["usage_decision"] == UD_PENDING, "qty_received"].sum()),
    ]
    f = pd.DataFrame(out, columns=["outcome", "qty"])
    f["pct"] = f["qty"] / lots["qty_received"].sum() * 100
    return f


def reconcile(lots: pd.DataFrame, mvts: pd.DataFrame) -> pd.DataFrame:
    """
    Rebuild each lot's quantities from its movements and compare to the lot.
    Returns only the lots that don't balance.
    """
    p = mvts.pivot_table(index="inspection_lot", columns="movement_type", values="quantity",
                         aggfunc="sum", fill_value=0)
    for m in (101, 321, 350, 122, 551):
        if m not in p.columns:
            p[m] = 0
    m = lots.set_index("inspection_lot").join(p, how="left").fillna({101: 0, 321: 0, 350: 0, 122: 0, 551: 0})
    accepted_expected = (m["qty_received"] - m["qty_rejected"]).where(m["usage_decision"] != UD_PENDING, 0)
    issues = []
    for lot, r in m.iterrows():
        e = []
        if r[101] != r["qty_received"]:
            e.append(f"GR 101 = {r[101]:g}, lot received {r['qty_received']:g}")
        if r[321] != accepted_expected[lot]:
            e.append(f"Release 321 = {r[321]:g}, expected {accepted_expected[lot]:g}")
        rej_expected = r["qty_rejected"] if r["usage_decision"] == UD_REJECT else 0
        if r[350] != rej_expected:
            e.append(f"Block 350 = {r[350]:g}, expected {rej_expected:g}")
        if r[122] + r[551] > r[350]:
            e.append("More returned/scrapped than was blocked")
        if e:
            issues.append({"inspection_lot": lot, "supplier": r["supplier_name"], "material": r["material"],
                           "issue": "; ".join(e)})
    return pd.DataFrame(issues, columns=["inspection_lot", "supplier", "material", "issue"])
