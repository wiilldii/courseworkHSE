"""
Main pipeline: loads, cleans, computes, and exports all outputs.

Usage:
    cd Z:/coursework
    python main.py
"""

import sys
import logging
from pathlib import Path

# Ensure src/ is on the path when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import setup_logger
from src.config import OUT_TABLES, OUT_FIGS, OUT_LOGS, DATA_RAW

from src.loaders import (
    load_economic_census,
    load_meti_2digit,
    load_meti_4digit,
    load_rd_personnel_summary,
    load_rd_expenditure_summary,
    load_enterprise_rd_by_industry,
    load_tech_exports,
    load_tech_imports,
    load_tech_exchange_by_region,
    load_worldbank_hightech,
    load_jpo_patent_applications,
)
from src.clean import (
    clean_economic_census,
    clean_meti_2digit,
    clean_meti_4digit,
    clean_tech_exchange,
    build_selected_hightech_technology_balance,
)
from src.metrics import (
    lq_census_employment,
    lq_census_establishments,
    lq_mfg_2digit,
    lq_mfg_4digit,
    build_htp_census_category_diagnostic,
    build_htp_components,
    aggregate_jpo_patents_by_region,
    worldbank_summary,
    worldbank_period_summary,
)
from src.plots import (
    plot_lq_mfg_2digit,
    plot_lq_mfg_4digit_selected,
    plot_htp_components,
    plot_htp_profile_normalized,
    plot_wb_exports_usd,
    plot_wb_exports_share,
    plot_tech_trade_balance,
    plot_selected_hightech_technology_balance,
)
from src.advanced_metrics import (
    build_htp_profile_normalized,
    build_htp_composite_sensitivity,
    build_krugman_specialization_index,
    build_va_per_worker_by_sector,
    build_lq_emp_vs_va_divergence,
    build_lq_robustness_across_bases,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def save_csv(df, name: str, log: logging.Logger):
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    path = OUT_TABLES / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    log.info(f"  Saved: {path.name}  ({len(df)} rows)")


def write_provenance(name: str, text: str, log: logging.Logger):
    """Write a companion .txt next to a CSV explaining its formula and bases."""
    OUT_TABLES.mkdir(parents=True, exist_ok=True)
    stem = name.rsplit(".", 1)[0]
    path = OUT_TABLES / f"{stem}.txt"
    path.write_text(text.strip() + "\n", encoding="utf-8")
    log.info(f"  Provenance: {path.name}")


def summarize_output(df, name: str, key_cols: list[str], log: logging.Logger):
    """Print a short stdout summary: key numbers + any fully-null columns."""
    null_cols = [c for c in df.columns if df[c].isna().all()]
    log.info(f"  SUMMARY [{name}]: {len(df)} rows, {len(df.columns)} cols")
    if null_cols:
        log.warning(f"    fully-null columns: {null_cols}")
    for c in key_cols:
        if c in df.columns:
            series = df[c]
            num = __import__("pandas").to_numeric(series, errors="coerce").dropna()
            if not num.empty:
                log.info(f"    {c}: min={num.min():.4g} max={num.max():.4g} "
                         f"mean={num.mean():.4g}")


def run_quality_checks(outputs: dict, log: logging.Logger):
    log.info("=== Quality checks ===")
    errors = []

    # Check Japan total > metro totals for numeric columns
    for key in ["df_meti2", "df_meti4"]:
        df = outputs.get(key)
        if df is None:
            continue
        region_col = "region" if "region" in df.columns else None
        if region_col is None:
            continue
        for val_col in ["employment", "value_added_mn_yen", "shipments_mn_yen"]:
            if val_col not in df.columns:
                continue
            japan_val = df.loc[df[region_col] == "japan", val_col].sum()
            for metro in ["tokyo_ma", "osaka_kansai"]:
                metro_val = df.loc[df[region_col] == metro, val_col].sum()
                if metro_val > 0 and japan_val > 0:
                    if metro_val > japan_val:
                        msg = f"FAIL: {key} {val_col}: {metro} ({metro_val:.0f}) > Japan ({japan_val:.0f})"
                        log.error(msg)
                        errors.append(msg)
                    else:
                        log.info(f"  OK: {key} {val_col} Japan > {metro}")

    # Check LQ tables have no inf
    for key in ["lq_census_emp", "lq_mfg2_emp", "lq_mfg4_emp"]:
        df = outputs.get(key)
        if df is None:
            continue
        import math
        if "lq" in df.columns:
            inf_count = df["lq"].apply(
                lambda x: x is not None and not math.isnan(x) and math.isinf(x)
                if x is not None else False
            ).sum()
            if inf_count > 0:
                msg = f"FAIL: {key} has {inf_count} inf values in lq"
                log.error(msg)
                errors.append(msg)
            else:
                log.info(f"  OK: {key} LQ has no inf values")

    # HTP double-counting check: if parent G is used, no G child row may be included.
    htp_diag = outputs.get("htp_census_diag")
    if htp_diag is not None and not htp_diag.empty:
        included_codes = set(htp_diag.loc[htp_diag["included"], "industry_code"].astype(str))
        if "G" in included_codes:
            child_included = sorted(
                c for c in included_codes if c.startswith("G") and c != "G"
            )
            if child_included:
                msg = f"FAIL: HTP census diagnostic includes parent G and child ICT rows: {child_included}"
                log.error(msg)
                errors.append(msg)
            else:
                log.info("  OK: HTP parent G used without G1/G2 child double counting")

    # Patent check: if manual file exists, patent share must be included.
    patent_manual = DATA_RAW / "jpo_patent_applications_prefecture_manual.csv"
    legacy_patent_manual = DATA_RAW / "jpo_patents_prefecture_manual.csv"
    df_htp = outputs.get("df_htp")
    if patent_manual.exists() or legacy_patent_manual.exists():
        if df_htp is None or "patent_applications_share" not in df_htp.columns:
            msg = "FAIL: JPO manual patent file exists, but HTP patent_applications_share is missing"
            log.error(msg)
            errors.append(msg)
        elif df_htp["patent_applications_share"].isna().any():
            msg = "FAIL: JPO manual patent file exists, but HTP patent_applications_share has missing values"
            log.error(msg)
            errors.append(msg)
        else:
            log.info("  OK: JPO patent component included in HTP index")
    else:
        log.warning("  WARNING: JPO patent manual CSV not found; HTP patent component skipped")

    # Check required output CSVs exist and are non-empty
    required_csvs = [
        "htp_hightech_census_categories_used.csv",
        "jpo_patent_applications_by_region.csv",
        "lq_all_industry_employment.csv",
        "lq_all_industry_establishments.csv",
        "lq_mfg_2digit_employment.csv",
        "lq_mfg_2digit_value_added.csv",
        "lq_mfg_4digit_employment.csv",
        "htp_components.csv",
        "htp_index.csv",
        "japan_hightech_exports_2010_2024.csv",
        "rd_expenditure_by_research_performer.csv",
        "rd_personnel_by_research_performer_clean.csv",
        "rd_expenditure_by_research_performer_clean.csv",
        "enterprise_rd_by_industry.csv",
        "enterprise_rd_by_industry_clean.csv",
        "technology_trade_balance_by_industry.csv",
        "selected_hightech_technology_balance.csv",
        "japan_hightech_exports_summary.csv",
        "japan_hightech_exports_share_summary.csv",
        # corrected / additional indices
        "htp_profile_normalized.csv",
        "htp_composite_sensitivity.csv",
        "krugman_specialization_index.csv",
        "va_per_worker_by_sector.csv",
        "lq_emp_vs_va_divergence.csv",
        "lq_robustness_across_bases.csv",
    ]
    for csv_name in required_csvs:
        path = OUT_TABLES / csv_name
        if not path.exists():
            msg = f"FAIL: missing output CSV: {csv_name}"
            log.error(msg)
            errors.append(msg)
        elif path.stat().st_size < 10:
            msg = f"FAIL: empty CSV: {csv_name}"
            log.error(msg)
            errors.append(msg)
        else:
            log.info(f"  OK: {csv_name}")

    # MIC header and unit checks.
    import pandas as pd
    header_markers = {"階層", "項目", "管理番号"}
    for csv_name in [
        "rd_personnel_by_research_performer_clean.csv",
        "rd_expenditure_by_research_performer_clean.csv",
        "enterprise_rd_by_industry_clean.csv",
        "technology_export_receipts_by_industry.csv",
        "technology_import_payments_by_industry.csv",
        "technology_trade_balance_by_industry.csv",
    ]:
        path = OUT_TABLES / csv_name
        if not path.exists():
            continue
        df_check = pd.read_csv(path)
        marker_found = False
        for col in [c for c in ["hierarchy", "industry_jp", "row_no"] if c in df_check.columns]:
            if df_check[col].astype(str).isin(header_markers).any():
                marker_found = True
        if marker_found:
            msg = f"FAIL: Japanese header label found as data in {csv_name}"
            log.error(msg)
            errors.append(msg)
        else:
            log.info(f"  OK: {csv_name} has no Japanese header rows")

        bad_unit_cols = [
            c for c in df_check.columns
            if c.endswith("_100mn_yen") and (
                c.startswith("tech_") or "expenditure" in c or "research_funds" in c
            )
        ]
        if bad_unit_cols:
            msg = f"FAIL: {csv_name} has monetary columns mislabeled as 100mn_yen: {bad_unit_cols}"
            log.error(msg)
            errors.append(msg)
        else:
            log.info(f"  OK: {csv_name} monetary units use *_mn_yen where required")

    selected_path = OUT_TABLES / "selected_hightech_technology_balance.csv"
    if selected_path.exists():
        df_selected = pd.read_csv(selected_path)
        selected_names = set(df_selected.get("industry_jp", pd.Series(dtype=str)))
        parent_child_conflicts = {
            "製造業": {"医薬品製造業", "化学工業", "はん用機械器具製造業"},
            "化学工業": {"総合化学工業", "油脂･塗料製造業", "その他の化学工業"},
            "電気機械器具製造業": {"電子応用・電気計測器製造業", "その他の電気機械器具製造業"},
            "学術研究，専門・技術サービス業": {"学術・開発研究機関", "技術サービス業(他に分類されないもの)"},
        }
        conflicts = []
        for parent, children in parent_child_conflicts.items():
            if parent in selected_names and selected_names.intersection(children):
                conflicts.append(parent)
        if conflicts:
            msg = f"FAIL: selected high-tech technology balance has parent-child conflicts: {conflicts}"
            log.error(msg)
            errors.append(msg)
        else:
            log.info("  OK: selected high-tech technology balance avoids parent-child summing")

    # Check required figures exist
    required_figs = [
        "lq_mfg_2digit_profile.png",
        "lq_mfg_4digit_selected_profile.png",
        "htp_components_bar.png",
        "japan_hightech_exports_usd.png",
        "japan_hightech_exports_share_manufactured.png",
        "technology_export_import_balance.png",
        "htp_profile_normalized.png",
    ]
    for fig_name in required_figs:
        path = OUT_FIGS / fig_name
        if not path.exists():
            msg = f"FAIL: missing figure: {fig_name}"
            log.error(msg)
            errors.append(msg)
        else:
            log.info(f"  OK: {fig_name}")

    if errors:
        log.error(f"Quality check FAILED: {len(errors)} issue(s)")
    else:
        log.info("Quality check PASSED: all checks OK")
    return errors


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    log = setup_logger()
    outputs = {}

    # ── 1. Economic Census ──────────────────────────────────────────────────
    log.info("=== Step 1: Economic Census (b1_006_1e) ===")
    df_census_raw = load_economic_census()
    df_census = clean_economic_census(df_census_raw)
    outputs["df_census"] = df_census

    # ── 2. METI manufacturing ───────────────────────────────────────────────
    log.info("=== Step 2: METI manufacturing data ===")
    df_meti2_raw = load_meti_2digit()
    df_meti2 = clean_meti_2digit(df_meti2_raw)
    outputs["df_meti2"] = df_meti2

    df_meti4_raw = load_meti_4digit()
    df_meti4 = clean_meti_4digit(df_meti4_raw)
    outputs["df_meti4"] = df_meti4

    # ── 3. LQ calculations ──────────────────────────────────────────────────
    log.info("=== Step 3: LQ calculations ===")

    lq_emp = lq_census_employment(df_census)
    outputs["lq_census_emp"] = lq_emp
    save_csv(lq_emp, "lq_all_industry_employment.csv", log)

    lq_est = lq_census_establishments(df_census)
    outputs["lq_census_est"] = lq_est
    save_csv(lq_est, "lq_all_industry_establishments.csv", log)

    lq_mfg2_emp = lq_mfg_2digit(df_meti2, "employment")
    outputs["lq_mfg2_emp"] = lq_mfg2_emp
    save_csv(lq_mfg2_emp, "lq_mfg_2digit_employment.csv", log)

    lq_mfg2_est = lq_mfg_2digit(df_meti2, "establishments")
    save_csv(lq_mfg2_est, "lq_mfg_2digit_establishments.csv", log)

    lq_mfg2_va = lq_mfg_2digit(df_meti2, "value_added_mn_yen")
    outputs["lq_mfg2_va"] = lq_mfg2_va
    save_csv(lq_mfg2_va, "lq_mfg_2digit_value_added.csv", log)

    lq_mfg4_emp = lq_mfg_4digit(df_meti4, "employment")
    outputs["lq_mfg4_emp"] = lq_mfg4_emp
    save_csv(lq_mfg4_emp, "lq_mfg_4digit_employment.csv", log)

    lq_mfg4_va = lq_mfg_4digit(df_meti4, "value_added_mn_yen")
    save_csv(lq_mfg4_va, "lq_mfg_4digit_value_added.csv", log)

    # Additional manufacturing profile tables
    save_csv(
        df_meti2[df_meti2["industry_code_2d"] != "00"],
        "mfg_2digit_value_added.csv", log
    )
    save_csv(
        df_meti4[df_meti4["industry_code_4d"].isin(
            [c for codes in __import__("src.config", fromlist=["MFG_4DIGIT_HIGHTECH"]).MFG_4DIGIT_HIGHTECH.values() for c in codes]
        )],
        "hightech_mfg_4digit_profile.csv", log
    )

    # ── 4. World Bank high-tech exports ─────────────────────────────────────
    log.info("=== Step 4: World Bank high-tech exports ===")
    wb_data = load_worldbank_hightech()

    df_wb_usd = worldbank_summary(wb_data["usd"], "hightech_exports_usd")
    save_csv(df_wb_usd, "japan_hightech_exports_2010_2024.csv", log)
    save_csv(
        worldbank_period_summary(df_wb_usd),
        "japan_hightech_exports_summary.csv",
        log,
    )

    df_wb_share = worldbank_summary(wb_data["share_manufactured"],
                                    "hightech_exports_share_manufactured")
    save_csv(df_wb_share, "japan_hightech_exports_share_manufactured.csv", log)
    save_csv(
        worldbank_period_summary(df_wb_share),
        "japan_hightech_exports_share_summary.csv",
        log,
    )

    # ── 5. MIC R&D national context ─────────────────────────────────────────
    log.info("=== Step 5: MIC R&D national context ===")

    df_rd_personnel = load_rd_personnel_summary()
    save_csv(df_rd_personnel, "rd_personnel_by_research_performer.csv", log)
    save_csv(df_rd_personnel, "rd_personnel_by_research_performer_clean.csv", log)

    df_rd_exp = load_rd_expenditure_summary()
    save_csv(df_rd_exp, "rd_expenditure_by_research_performer.csv", log)
    save_csv(df_rd_exp, "rd_expenditure_by_research_performer_clean.csv", log)

    df_ent_rd = load_enterprise_rd_by_industry()
    save_csv(df_ent_rd, "enterprise_rd_by_industry.csv", log)
    save_csv(df_ent_rd, "enterprise_rd_by_industry_clean.csv", log)

    # ── 6. Technology exchange ──────────────────────────────────────────────
    log.info("=== Step 6: Technology exchange ===")

    df_tech_exp = load_tech_exports()
    save_csv(df_tech_exp, "technology_export_receipts_by_industry.csv", log)

    df_tech_imp = load_tech_imports()
    save_csv(df_tech_imp, "technology_import_payments_by_industry.csv", log)

    df_tech_region = load_tech_exchange_by_region()
    save_csv(df_tech_region, "technology_exchange_by_region.csv", log)

    df_tech_balance = clean_tech_exchange(df_tech_exp, df_tech_imp)
    save_csv(df_tech_balance, "technology_trade_balance_by_industry.csv", log)

    df_selected_tech = build_selected_hightech_technology_balance(df_tech_balance)
    save_csv(df_selected_tech, "selected_hightech_technology_balance.csv", log)

    # ── 7. JPO patent data (optional) ───────────────────────────────────────
    log.info("=== Step 7: JPO patent data (optional) ===")
    df_patents = load_jpo_patent_applications()
    if df_patents is not None and not df_patents.empty:
        df_pat_regions = aggregate_jpo_patents_by_region(df_patents)
        save_csv(df_pat_regions, "jpo_patent_applications_by_region.csv", log)
    else:
        import pandas as pd
        df_pat_regions = pd.DataFrame(columns=[
            "region",
            "region_label",
            "patent_applications_2024",
            "japan_patent_applications",
            "patent_applications_share",
        ])
        save_csv(df_pat_regions, "jpo_patent_applications_by_region.csv", log)

    # ── 8. HTP Index ────────────────────────────────────────────────────────
    log.info("=== Step 8: HTP Contribution Index ===")
    df_htp_census_diag = build_htp_census_category_diagnostic(df_census)
    outputs["htp_census_diag"] = df_htp_census_diag
    save_csv(
        df_htp_census_diag,
        "htp_hightech_census_categories_used.csv",
        log,
    )

    df_htp = build_htp_components(df_census, df_meti2, df_meti4, df_patents)
    outputs["df_htp"] = df_htp

    # Component shares
    component_cols = [
        "region", "region_label",
        "hightech_employment_share",
        "hightech_establishments_share",
        "hightech_mfg_va_share",
        "hightech_mfg_shipments_share",
        "patent_applications_share",
        "n_components_used",
        "components_used",
        "components_missing",
    ]
    save_csv(
        df_htp[[c for c in component_cols if c in df_htp.columns]],
        "htp_components.csv", log
    )

    # Index table
    index_cols = [
        "region", "region_label",
        "htp_contribution_index", "htp_index_100",
        "n_components_used", "components_used",
    ]
    save_csv(
        df_htp[[c for c in index_cols if c in df_htp.columns]],
        "htp_index.csv", log
    )

    # ── 8b. Corrected / additional analytical indices ────────────────────────
    log.info("=== Step 8b: Corrected & additional indices ===")

    # TASK 1: HTP national-norm-normalized profile (replaces share-based reading)
    df_htp_profile = build_htp_profile_normalized(
        df_census, df_meti2, df_meti4, df_patents, df_meti2_raw
    )
    outputs["df_htp_profile"] = df_htp_profile
    save_csv(df_htp_profile, "htp_profile_normalized.csv", log)
    write_provenance(
        "htp_profile_normalized.csv",
        """
HTP normalized profile — provenance
===================================
Formula (per component k, region r):
    intensity_region = X[k,r] / B[k,r]
    intensity_japan  = X[k,Japan] / B[k,Japan]
    normalized_value = intensity_region / intensity_japan
A value of 1.0 means the region is exactly at the national average on that
dimension (this is location-quotient logic). This deliberately replaces the old
"share of national total", which measured size rather than specialization.

Components, numerator (X) and base (B):
  hightech_employment      X = Economic Census parent service employment (G,L,
                               hierarchy 2) + METI 2-digit high-tech mfg
                               employment;  B = total employment (all industries)
  hightech_establishments  X = same service+mfg high-tech establishments;
                               B = total establishments (all industries)
  hightech_mfg_value_added X = METI 2-digit high-tech mfg value added (mn yen);
                               B = total manufacturing value added (mn yen)
  hightech_mfg_shipments   X = METI 2-digit high-tech mfg shipments (mn yen);
                               B = total manufacturing shipments (mn yen)
  patent_applications      X = JPO patent applications (latest year, first-
                               applicant address);  B = total employment

Construct labels (to control conceptual double counting):
  hq_corporate_innovation = {employment, establishments, patents}
       (all three proxy where corporate HQs sit)
  industrial_production    = {mfg value added, mfg shipments}

row_type: "component" rows give per-component values; "construct_mean" rows give
the mean normalized value within each construct.

Caveats: employment/establishments numerators mix Economic Census (all
establishments) with METI manufacturing (4+ employees); see README. The
"note" column flags where a metro value relies >=90% on a single prefecture.
        """,
        log,
    )
    summarize_output(
        df_htp_profile[df_htp_profile["row_type"] == "component"],
        "htp_profile_normalized", ["normalized_value"], log,
    )

    df_htp_sens = build_htp_composite_sensitivity(df_htp_profile)
    save_csv(df_htp_sens, "htp_composite_sensitivity.csv", log)
    write_provenance(
        "htp_composite_sensitivity.csv",
        """
HTP composite sensitivity — provenance
======================================
Optional scalar composite = weighted mean of available normalized component
values (national norm = 1.0). Three weight schemes test ranking stability:
  equal               : every component weighted equally
  hq_weighted         : hq_corporate_innovation construct weighted 2x
  production_weighted : industrial_production construct weighted 2x
ranking_stable_across_schemes = True means the same region leads under all three
schemes. The composite is a SUMMARY of the profile, not the headline result.
        """,
        log,
    )
    summarize_output(
        df_htp_sens, "htp_composite_sensitivity",
        ["tokyo_ma_composite", "osaka_kansai_composite"], log,
    )

    # TASK 2: Krugman specialization index
    df_krugman = build_krugman_specialization_index(df_meti2)
    save_csv(df_krugman, "krugman_specialization_index.csv", log)
    write_provenance(
        "krugman_specialization_index.csv",
        """
Krugman specialization (dissimilarity) index — provenance
=========================================================
s_i(region) = industry_i value / region manufacturing total (2-digit, METI).
KSI = 0.5 * sum_i | s_i(Tokyo) - s_i(Osaka) |, computed separately on
employment shares and value-added shares. KSI = 0 means identical industrial
structure; KSI = 1 means completely different. The TOTAL_KSI row holds the
index; other rows hold per-industry absolute share differences (the drivers of
divergence). Monetary base: value_added in million yen.
        """,
        log,
    )
    summarize_output(
        df_krugman, "krugman_specialization_index",
        ["abs_diff_employment", "abs_diff_value_added"], log,
    )

    # TASK 3: Value-added per worker by sector
    df_vapw = build_va_per_worker_by_sector(df_meti2, df_meti4, df_meti2_raw)
    save_csv(df_vapw, "va_per_worker_by_sector.csv", log)
    write_provenance(
        "va_per_worker_by_sector.csv",
        """
Value added per worker by sector — provenance
=============================================
va_per_worker_mn_yen = value_added_mn_yen / employment (million yen per worker).
va_per_worker_vs_japan = (region VA/worker) / (Japan VA/worker) for the same
industry; >1 means the region's slice of that industry is higher-value than the
national average for that industry. Covers 2-digit and 4-digit high-tech
industries (see 'level'). single_prefecture_note flags 2-digit metro values that
rely >=90% on one prefecture. Monetary unit: million yen.
        """,
        log,
    )
    summarize_output(
        df_vapw, "va_per_worker_by_sector",
        ["va_per_worker_mn_yen", "va_per_worker_vs_japan"], log,
    )

    # TASK 4: LQ employment vs value-added divergence
    df_lq_div = build_lq_emp_vs_va_divergence(
        lq_mfg2_emp, lq_mfg2_va, lq_mfg4_emp, lq_mfg4_va
    )
    save_csv(df_lq_div, "lq_emp_vs_va_divergence.csv", log)
    write_provenance(
        "lq_emp_vs_va_divergence.csv",
        """
LQ employment vs value-added divergence — provenance
====================================================
lq_va_minus_lq_emp = LQ(value added) - LQ(employment) for each high-tech
industry/region. Where this is positive (holds_high_value_end = True), the
region is more specialized in that industry's VALUE than in its headcount, i.e.
it holds the high-value end. is_top3_high_value_divergence flags the three
largest positive divergences per region and level. LQ uses the standard
location-quotient formula (unchanged).
        """,
        log,
    )
    summarize_output(
        df_lq_div, "lq_emp_vs_va_divergence", ["lq_va_minus_lq_emp"], log,
    )

    # TASK 5: LQ robustness across bases
    df_lq_robust = build_lq_robustness_across_bases(
        lq_mfg2_emp, lq_mfg2_est, lq_mfg2_va
    )
    save_csv(df_lq_robust, "lq_robustness_across_bases.csv", log)
    write_provenance(
        "lq_robustness_across_bases.csv",
        """
LQ robustness across bases — provenance
=======================================
For 2-digit high-tech industries, LQ is computed on three bases: employment,
establishments and value added. rank_* gives each industry's LQ rank within the
region under each base (1 = highest). classification_stable = True when all three
bases agree on the LQ>1 (specialized) vs LQ<1 (under-represented) classification,
i.e. the specialization signal is robust to the choice of base.
        """,
        log,
    )
    summarize_output(
        df_lq_robust, "lq_robustness_across_bases",
        ["lq_employment", "lq_establishments", "lq_value_added"], log,
    )

    # ── 9. Figures ──────────────────────────────────────────────────────────
    log.info("=== Step 9: Generating figures ===")

    plot_lq_mfg_2digit(lq_mfg2_emp, "employment")
    plot_lq_mfg_4digit_selected(lq_mfg4_emp)
    plot_htp_components(df_htp)
    plot_htp_profile_normalized(df_htp_profile)
    plot_wb_exports_usd(df_wb_usd)
    plot_wb_exports_share(df_wb_share)
    plot_tech_trade_balance(df_tech_balance)
    plot_selected_hightech_technology_balance(df_selected_tech)

    # ── 10. Quality checks ──────────────────────────────────────────────────
    log.info("=== Step 10: Quality checks ===")
    run_quality_checks(outputs, log)

    log.info("=== Pipeline complete ===")
    log.info(f"Tables → {OUT_TABLES}")
    log.info(f"Figures → {OUT_FIGS}")
    log.info(f"Log    → {OUT_LOGS / 'run_log.txt'}")


if __name__ == "__main__":
    main()
