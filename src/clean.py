"""
Data cleaning and region tagging.
Operates on DataFrames returned by loaders.
"""

import logging
import pandas as pd

from src.config import (
    JAPAN_AREA_CODE,
    TOKYO_MA_AREA_CODES,
    OSAKA_KANSAI_AREA_CODES,
    TOKYO_MA_PREF_CODES,
    OSAKA_KANSAI_PREF_CODES,
    JAPAN_PREF_CODE,
    REGION_LABELS,
    CENSUS_TOTAL_CODE,
    BROAD_HIGHTECH_DIV_CODES,
    MFG_2DIGIT_HIGHTECH,
    MFG_4DIGIT_HIGHTECH,
    ALL_2DIGIT_HIGHTECH_CODES,
    ALL_4DIGIT_HIGHTECH_CODES,
    SELECTED_HIGHTECH_TECHNOLOGY_ROWS,
)

log = logging.getLogger("coursework")


def _tag_region_census(area_code: str) -> str | None:
    if area_code == JAPAN_AREA_CODE:
        return "japan"
    if area_code in TOKYO_MA_AREA_CODES:
        return "tokyo_ma"
    if area_code in OSAKA_KANSAI_AREA_CODES:
        return "osaka_kansai"
    return None


def _tag_region_meti(pref_code: str) -> str | None:
    if pref_code == JAPAN_PREF_CODE:
        return "japan"
    if pref_code in TOKYO_MA_PREF_CODES:
        return "tokyo_ma"
    if pref_code in OSAKA_KANSAI_PREF_CODES:
        return "osaka_kansai"
    return None


# ── Economic Census cleaning ──────────────────────────────────────────────────

def clean_economic_census(df: pd.DataFrame) -> pd.DataFrame:
    """Tag regions; keep only rows relevant to LQ."""
    df = df.copy()
    df["region"] = df["area_code"].map(_tag_region_census)
    df = df[df["region"].notna()].copy()

    # Aggregate Tokyo MA and Osaka/Kansai by summing prefecture rows
    japan_rows = df[df["region"] == "japan"].copy()

    metro_sum = (
        df[df["region"] != "japan"]
        .groupby(["region", "div_code", "industry_label", "hierarchy"], as_index=False)
        .agg(establishments=("establishments", "sum"), employment=("employment", "sum"))
    )
    # Add area_code column (synthetic label for metro aggregates)
    metro_sum["area_code"] = metro_sum["region"]

    combined = pd.concat([japan_rows, metro_sum], ignore_index=True)
    log.info(f"  Economic Census cleaned: {len(combined)} rows (Japan + 2 metro aggregates)")
    return combined


def clean_meti_2digit(df: pd.DataFrame) -> pd.DataFrame:
    """Tag regions; aggregate metro prefectures; convert '00' industry = all-mfg total."""
    df = df.copy()
    df["region"] = df["pref_code"].map(_tag_region_meti)
    df = df[df["region"].notna()].copy()

    japan_rows = df[df["region"] == "japan"].copy()

    metro_sum = (
        df[df["region"] != "japan"]
        .groupby(
            ["region", "industry_code_2d", "industry_name_jp"], as_index=False
        )
        .agg(
            establishments=("establishments", "sum"),
            employment=("employment", "sum"),
            shipments_mn_yen=("shipments_mn_yen", "sum"),
            value_added_mn_yen=("value_added_mn_yen", "sum"),
        )
    )
    metro_sum["pref_code"] = metro_sum["region"]
    metro_sum["pref_name"] = metro_sum["region"].map(REGION_LABELS)

    combined = pd.concat([japan_rows, metro_sum], ignore_index=True)

    # Make sure region column is carried through
    combined["region"] = combined["pref_code"].map(
        lambda x: _tag_region_meti(x) if x in (
            [JAPAN_PREF_CODE] + TOKYO_MA_PREF_CODES + OSAKA_KANSAI_PREF_CODES
        ) else x
    )
    # For metro aggregate rows, region == pref_code already set above
    combined.loc[combined["pref_code"].isin(["tokyo_ma", "osaka_kansai"]), "region"] = \
        combined.loc[combined["pref_code"].isin(["tokyo_ma", "osaka_kansai"]), "pref_code"]

    log.info(f"  METI 2-digit cleaned: {len(combined)} rows")
    return combined


def clean_meti_4digit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate 4-digit manufacturing data for Japan, Tokyo MA, and Osaka/Kansai.
    Japan totals are derived by summing all 47 prefectures (national rows absent
    for individual 4-digit codes in 第7表).
    """
    df = df.copy()

    # Japan total = sum of all prefecture rows
    japan_sum = (
        df[df["pref_code"] != "00"]
        .groupby(["industry_code_4d", "industry_name_jp"], as_index=False)
        .agg(
            establishments=("establishments", "sum"),
            employment=("employment", "sum"),
            shipments_mn_yen=("shipments_mn_yen", "sum"),
            value_added_mn_yen=("value_added_mn_yen", "sum"),
        )
    )
    japan_sum["region"] = "japan"
    japan_sum["pref_code"] = "00"
    japan_sum["pref_name"] = "Japan"

    # Metro aggregates
    for pref_codes, region_name in [
        (TOKYO_MA_PREF_CODES, "tokyo_ma"),
        (OSAKA_KANSAI_PREF_CODES, "osaka_kansai"),
    ]:
        metro_sum = (
            df[df["pref_code"].isin(pref_codes)]
            .groupby(["industry_code_4d", "industry_name_jp"], as_index=False)
            .agg(
                establishments=("establishments", "sum"),
                employment=("employment", "sum"),
                shipments_mn_yen=("shipments_mn_yen", "sum"),
                value_added_mn_yen=("value_added_mn_yen", "sum"),
            )
        )
        metro_sum["region"] = region_name
        metro_sum["pref_code"] = region_name
        metro_sum["pref_name"] = REGION_LABELS[region_name]
        japan_sum = pd.concat([japan_sum, metro_sum], ignore_index=True)

    combined = japan_sum
    log.info(f"  METI 4-digit cleaned: {len(combined)} rows")
    return combined


# ── Technology exchange cleaning ──────────────────────────────────────────────

def clean_tech_exchange(df_exports: pd.DataFrame,
                        df_imports: pd.DataFrame) -> pd.DataFrame:
    """
    Merge exports and imports on industry row number, compute trade balance
    and export/import ratio.
    """
    # Use row_no as join key (same table structure for 210/211)
    merged = pd.merge(
        df_exports[["row_no", "industry_jp", "hierarchy",
                    "tech_export_receipts_mn_yen"]],
        df_imports[["row_no", "industry_jp",
                    "tech_import_payments_mn_yen"]],
        on="row_no",
        how="outer",
        suffixes=("_exp", "_imp"),
    )
    # Prefer export industry name (fill from import if missing)
    merged["industry_jp"] = merged["industry_jp_exp"].combine_first(
        merged.get("industry_jp_imp", pd.Series(dtype=str))
    )

    exp = merged["tech_export_receipts_mn_yen"]
    imp = merged["tech_import_payments_mn_yen"]

    merged["tech_balance_mn_yen"] = exp - imp
    merged["export_import_ratio"] = exp / imp.where(imp != 0)

    drop_cols = [c for c in ["industry_jp_exp", "industry_jp_imp"] if c in merged.columns]
    merged = merged.drop(columns=drop_cols)

    log.info(f"  Tech exchange merged: {len(merged)} rows")
    return merged


def build_selected_hightech_technology_balance(df_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Select high-tech technology exchange rows without summing parent and child
    industry rows together.  All monetary columns are in million yen.
    """
    df = df_balance.copy()
    selected_names = list(SELECTED_HIGHTECH_TECHNOLOGY_ROWS.keys())
    selected = df[df["industry_jp"].isin(selected_names)].copy()

    if selected.empty:
        return pd.DataFrame(columns=[
            "industry_jp",
            "industry_en",
            "hierarchy",
            "tech_export_receipts_mn_yen",
            "tech_import_payments_mn_yen",
            "tech_balance_mn_yen",
            "export_import_ratio",
            "interpretation_group",
        ])

    selected["industry_en"] = selected["industry_jp"].map(
        lambda name: SELECTED_HIGHTECH_TECHNOLOGY_ROWS[name][0]
    )
    selected["interpretation_group"] = selected["industry_jp"].map(
        lambda name: SELECTED_HIGHTECH_TECHNOLOGY_ROWS[name][1]
    )

    out_cols = [
        "industry_jp",
        "industry_en",
        "hierarchy",
        "tech_export_receipts_mn_yen",
        "tech_import_payments_mn_yen",
        "tech_balance_mn_yen",
        "export_import_ratio",
        "interpretation_group",
    ]
    selected = selected[out_cols].sort_values(
        ["interpretation_group", "hierarchy", "industry_jp"]
    )
    log.info(f"  Selected high-tech technology balance: {len(selected)} rows")
    return selected
