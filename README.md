# 📦 Supplier Receiving Inspection Analyzer

A Python/Streamlit app for **incoming (receiving) inspection** analytics, modeled on how SAP MM and QM record the process: a goods receipt posts stock into quality inspection, an inspection lot is created, and the usage decision releases the stock or blocks it for return or scrap.

```
GR for PO (101) → QI stock → inspection lot (origin 01) → usage decision
     accept → 321  QI → unrestricted
     reject → 350  QI → blocked → 122 return to vendor / 551 scrap
```

> **All data is synthetic.** Supplier names, materials, PO and lot numbers, and inspectors are fictional; any resemblance to real companies or people is coincidental. This project is not affiliated with SAP and does not connect to an SAP system.

## What it shows

| Tab | Purpose |
|---|---|
| **Overview** | Lots inspected, lot acceptance rate, rejected PPM, inspection cycle time, QI and blocked stock value, monthly acceptance trend, and where received quantity ended up |
| **Supplier scorecard** | Acceptance %, PPM, cycle time, on-time inspection, and top defect per supplier, with automatic **SCAR** and **skip-lot** recommendations |
| **Defects** | Defect Pareto (by lots or by quantity) and a supplier × defect heatmap |
| **Stock & aging** | Dollars waiting in QI stock or in blocked stock without a disposition, bucketed by age |
| **Movements (MB51)** | Material-document list derived from the lots, plus a **reconciliation check** that rebuilds each lot from its movements and flags postings that don't balance |

Defect categories follow a standard receiving check sheet: **Packaging, Appearance, Joint, Dimensional, Documentation**.

## Decision rules (adjustable in the sidebar)

- **SCAR candidate:** ≥ 3 rejected lots in 90 days, or rejected PPM above 15,000.
- **Skip-lot candidate:** the last 10+ lots accepted, no recent rejects, and no open flag. Reduced inspection would still need QE approval under the quality plan.
- **On-time inspection:** usage decision within 3 days of goods receipt.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
python -m unittest discover tests -v   # 14 tests
```

## Use your own data

Choose **Upload inspection lots (CSV)** and download the template. Required columns:

`inspection_lot, po_number, supplier_id, supplier_name, material, material_desc, gr_date, qty_received, unit, unit_price, qty_rejected, defect_category, usage_decision (A / R / PENDING), ud_date, disposition (RTV / SCRAP / PENDING / blank)`

Uploads are validated before analysis, and problems are reported with spreadsheet row numbers (for example, rejected quantity larger than received, or a rejected lot without a defect category).

## Project structure

```
app.py          Streamlit UI
sri_core.py     Data model, synthetic data, metrics, validation, reconciliation (no UI code)
tests/          Unit tests with hand-calculated expected values
sample_data/    Synthetic inspection-lot CSV
```

---

Built by **Marcus Faulkner** · [Portfolio](https://marusfaulkner.github.io) · [LinkedIn](https://www.linkedin.com/in/marcus-f-0a622b70)
