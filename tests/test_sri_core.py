"""Unit tests for sri_core. Run: python -m pytest -q  (or python -m unittest discover tests)"""

import sys
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sri_core as s  # noqa: E402

TODAY = date(2026, 11, 21)


def tiny() -> pd.DataFrame:
    """Hand-built lots with known answers."""
    rows = [
        # id, supplier, received, rejected, UD, GR, UD date, category, disposition
        ("L1", "V-1", "Acme", 100, 0, "A", "2026-11-01", "2026-11-03", "", ""),
        ("L2", "V-1", "Acme", 100, 10, "R", "2026-11-05", "2026-11-12", "Dimensional", "RTV"),
        ("L3", "V-1", "Acme", 50, 0, "PENDING", "2026-11-18", None, "", ""),
        ("L4", "V-2", "Beta", 20, 5, "R", "2026-11-10", "2026-11-15", "Appearance", "PENDING"),
        ("L5", "V-2", "Beta", 20, 0, "A", "2026-11-11", "2026-11-12", "", ""),
    ]
    df = pd.DataFrame(rows, columns=["inspection_lot", "supplier_id", "supplier_name", "qty_received",
                                     "qty_rejected", "usage_decision", "gr_date", "ud_date",
                                     "defect_category", "disposition"])
    df["po_number"] = "4500000001"
    df["material"] = "MAT-1"
    df["material_desc"] = "Test part"
    df["unit"] = "EA"
    df["unit_price"] = 10.0
    return s.validate_lots(df)


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.lots = tiny()

    def test_kpis(self):
        k = s.kpis(self.lots, pd.Timestamp(TODAY))
        self.assertEqual(k["decided"], 4)
        self.assertAlmostEqual(k["lar"], 50.0)                        # 2 of 4 decided accepted
        self.assertAlmostEqual(k["ppm"], 15 / 240 * 1e6)              # 15 rejected of 240 received
        self.assertEqual(k["qi_lots"], 1)
        self.assertAlmostEqual(k["qi_value"], 500.0)                  # 50 x $10
        self.assertAlmostEqual(k["blocked_value"], 50.0)              # 5 x $10 awaiting disposition
        self.assertEqual(k["oldest_qi_days"], 3)

    def test_scorecard(self):
        sc = s.supplier_scorecard(self.lots, s.Thresholds(scar_rejects=1, skip_lot_run=1), pd.Timestamp(TODAY))
        acme = sc.set_index("Supplier").loc["Acme"]
        self.assertAlmostEqual(acme["Acceptance %"], 50.0)
        self.assertEqual(acme["Top defect"], "Dimensional")
        self.assertIn("1 rejects", acme["SCAR flag"])
        beta = sc.set_index("Supplier").loc["Beta"]
        self.assertEqual(beta["Accept run"], 1)                       # L5 accepted after L4 rejected
        self.assertFalse(beta["Skip-lot candidate"])                  # a recent reject blocks skip-lot

    def test_pareto_and_flow(self):
        p = s.defect_pareto(self.lots)
        self.assertEqual(p.iloc[0]["defect_category"], "Dimensional")
        self.assertAlmostEqual(p["cum_pct"].iloc[-1], 100.0)
        f = s.flow_summary(self.lots).set_index("outcome")["qty"]
        self.assertEqual(f["Released to unrestricted (321)"], 100 + 90 + 15 + 20)
        self.assertEqual(f["Returned to vendor (122)"], 10)
        self.assertEqual(f["Blocked, awaiting disposition"], 5)
        self.assertEqual(f["In QI stock, awaiting inspection"], 50)
        self.assertEqual(f.sum(), self.lots["qty_received"].sum())    # every unit accounted for

    def test_aging(self):
        a = s.aging(self.lots, pd.Timestamp(TODAY))
        self.assertEqual(set(a["inspection_lot"]), {"L3", "L4"})

    def test_movements_reconcile_clean(self):
        m = s.build_movements(self.lots)
        self.assertEqual(sorted(m[m["inspection_lot"] == "L2"]["movement_type"]), [101, 122, 321, 350])
        self.assertTrue(s.reconcile(self.lots, m).empty)

    def test_reconcile_catches_short_posting(self):
        m = s.build_movements(self.lots)
        m.loc[(m["inspection_lot"] == "L1") & (m["movement_type"] == 321), "quantity"] = 99
        issues = s.reconcile(self.lots, m)
        self.assertEqual(issues["inspection_lot"].tolist(), ["L1"])
        self.assertIn("Release 321 = 99", issues.iloc[0]["issue"])


class TestValidation(unittest.TestCase):
    def test_rejects_bad_rows(self):
        df = tiny()
        df.loc[0, "qty_rejected"] = 500                                # more than received
        df.loc[1, "defect_category"] = ""                               # rejected without category
        with self.assertRaises(s.DataError) as ctx:
            s.validate_lots(df)
        msg = str(ctx.exception)
        self.assertIn("exceeds received", msg)
        self.assertIn("needs a defect category", msg)

    def test_optional_inspector_column_is_added(self):
        df = tiny()
        self.assertNotIn("inspector", df.drop(columns="inspector", errors="ignore").columns)
        clean = s.validate_lots(df.drop(columns="inspector", errors="ignore"))
        self.assertIn("inspector", clean.columns)

    def test_missing_column(self):
        with self.assertRaises(s.DataError):
            s.validate_lots(tiny().drop(columns="po_number"))


class TestSynthetic(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lots = s.generate_lots(today=TODAY)

    def test_generated_data_is_valid(self):
        s.validate_lots(self.lots)                                      # must not raise
        self.assertGreater(len(self.lots), 400)

    def test_story_is_present(self):
        sc = s.supplier_scorecard(self.lots, today=pd.Timestamp(TODAY)).set_index("Supplier")
        self.assertTrue(sc.loc["Tessaro Castings", "SCAR flag"])       # the problem supplier
        self.assertTrue(sc.loc["Halvorsen Copper & Wire", "Skip-lot candidate"])  # the clean supplier

    def test_flagged_supplier_is_never_skip_lot(self):
        sc = s.supplier_scorecard(self.lots, today=pd.Timestamp(TODAY))
        self.assertFalse((sc["SCAR flag"].astype(bool) & sc["Skip-lot candidate"]).any())
        self.assertEqual(sc.iloc[0]["Supplier"], "Tessaro Castings")   # worst supplier sorts first

    def test_integrity_demo_finds_exactly_one(self):
        m = s.build_movements(self.lots, integrity_demo=True)
        self.assertEqual(len(s.reconcile(self.lots, m)), 1)
        self.assertTrue(s.reconcile(self.lots, s.build_movements(self.lots)).empty)

    def test_deterministic(self):
        pd.testing.assert_frame_equal(self.lots, s.generate_lots(today=TODAY))


if __name__ == "__main__":
    unittest.main()
