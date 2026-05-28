"""
LQ calculation and HTP Contribution Index assembly.
"""

import logging
import pandas as pd
import numpy as np

from src.config import (
    REGION_LABELS,
    CENSUS_TOTAL_CODE,
    USE_BROAD_ICT_PARENT,
    BROAD_HIGHTECH_CENSUS_INDUSTRIES,
    CENSUS_PARENT_HIERARCHY,
    MFG_2DIGIT_ENGLISH,
    ALL_2DIGIT_HIGHTECH_CODES,
    ALL_4DIGIT_HIGHTECH_CODES,
    MFG_2DIGIT_HIGHTECH,
    MFG_4DIGIT_HIGHTECH,
)
from src.utils import safe_divide

log = logging.getLogger("coursework")


# ── Generic LQ engine ─────────────────────────────────────────────────────────

def calculate_lq_from_totals(
    region_ind: float | None,
    region_tot: float | None,
    japan_ind: float | None,
    japan_tot: float | None,
) -> float | None:
    """LQ = (reg_ind / reg_tot) / (jpn_ind / jpn_tot)."""
    reg_share = safe_divide(region_ind, region_tot)
    jpn_share = safe_divide(japan_ind, japan_tot)
    return safe_divide(reg_share, jpn_share)


def _build_lq_table(df: pd.DataFrame,
                    industry_col: str,
                    value_col: str,
                    total_industry_code: str,
                    region_col: str = "region") -> pd.DataFrame:
    """
    Generic LQ builder.
    df must already be aggregated (one row per region × industry).
    """
    regions = ["tokyo_ma", "osaka_kansai"]

    # Japan totals
    japan_total = df.loc[
        (df[region_col] == "japan") & (df[industry_col] == total_industry_code),
        value_col
    ].sum()

    records = []
    for region in regions:
        region_total = df.loc[
            (df[region_col] == region) & (df[industry_col] == total_industry_code),
            value_col
        ].sum()

        industries = df[
            (df[region_col] == "japan") & (df[industry_col] != total_industry_code)
        ][industry_col].unique()

        for ind in industries:
            japan_ind = df.loc[
                (df[region_col] == "japan") & (df[industry_col] == ind),
                value_col
            ].sum()
            region_ind = df.loc[
                (df[region_col] == region) & (df[industry_col] == ind),
                value_col
            ].sum()

            lq = calculate_lq_from_totals(region_ind, region_total,
                                          japan_ind, japan_total)
            records.append({
                "region": region,
                "region_label": REGION_LABELS[region],
                industry_col: ind,
                value_col: region_ind,
                f"{value_col}_japan": japan_ind,
                "lq": lq,
            })

    return pd.DataFrame(records)


# ── LQ: Economic Census (broad sectors) ──────────────────────────────────────

def lq_census_employment(df_census: pd.DataFrame) -> pd.DataFrame:
    """LQ by employment for broad industry divisions (Economic Census)."""
    log.info("Calculating LQ: census employment")

    agg = df_census.groupby(["region", "div_code"], as_index=False).agg(
        employment=("employment", "sum")
    )

    result = _build_lq_table(agg, "div_code", "employment", CENSUS_TOTAL_CODE)

    # Add human-readable label
    label_map = {
        "E": "Manufacturing",
        "G": "Information & Communications",
        "G1": "ICT (Communications & Broadcasting)",
        "G2": "ICT (Software & Internet)",
        "L": "Scientific Research & Technical Services",
        "D": "Construction",
        "F": "Utilities",
        "H": "Transport & Postal",
        "I": "Wholesale & Retail",
        "J": "Finance & Insurance",
        "K": "Real Estate & Leasing",
        "M": "Accommodations & Food Services",
        "O": "Education",
        "P": "Healthcare & Welfare",
    }
    result["industry_label"] = result["div_code"].map(label_map).fillna(result["div_code"])
    log.info(f"  census employment LQ: {len(result)} rows")
    return result


def lq_census_establishments(df_census: pd.DataFrame) -> pd.DataFrame:
    """LQ by establishments for broad industry divisions (Economic Census)."""
    log.info("Calculating LQ: census establishments")

    agg = df_census.groupby(["region", "div_code"], as_index=False).agg(
        establishments=("establishments", "sum")
    )

    result = _build_lq_table(agg, "div_code", "establishments", CENSUS_TOTAL_CODE)
    log.info(f"  census establishments LQ: {len(result)} rows")
    return result


# ── LQ: METI 2-digit manufacturing ───────────────────────────────────────────

def lq_mfg_2digit(df_meti2: pd.DataFrame,
                  value_col: str = "employment") -> pd.DataFrame:
    """LQ for 2-digit manufacturing industries."""
    log.info(f"Calculating LQ: METI 2-digit by {value_col}")

    # Determine region column
    df = df_meti2.copy()
    if "region" not in df.columns:
        raise ValueError("region column missing")

    agg = df.groupby(["region", "industry_code_2d"], as_index=False).agg(
        **{value_col: (value_col, "sum")}
    )

    result = _build_lq_table(agg, "industry_code_2d", value_col, "00")

    # Add English labels
    result["industry_en"] = result["industry_code_2d"].map(MFG_2DIGIT_ENGLISH)
    result["is_hightech"] = result["industry_code_2d"].isin(ALL_2DIGIT_HIGHTECH_CODES)

    log.info(f"  METI 2-digit {value_col} LQ: {len(result)} rows")
    return result


# ── LQ: METI 4-digit manufacturing ───────────────────────────────────────────

def lq_mfg_4digit(df_meti4: pd.DataFrame,
                  value_col: str = "employment") -> pd.DataFrame:
    """LQ for 4-digit manufacturing industries."""
    log.info(f"Calculating LQ: METI 4-digit by {value_col}")

    df = df_meti4.copy()

    agg = df.groupby(["region", "industry_code_4d"], as_index=False).agg(
        **{value_col: (value_col, "sum")}
    )

    # Build a pseudo-total row: sum of all 4-digit industries per region
    agg_total = agg.groupby("region")[value_col].sum().reset_index()
    agg_total["industry_code_4d"] = "0000"
    agg = pd.concat([agg, agg_total], ignore_index=True)

    result = _build_lq_table(agg, "industry_code_4d", value_col, "0000")

    result["is_hightech"] = result["industry_code_4d"].isin(ALL_4DIGIT_HIGHTECH_CODES)

    # Add category label
    code_to_cat = {}
    for cat, codes in MFG_4DIGIT_HIGHTECH.items():
        for c in codes:
            code_to_cat[c] = cat
    result["hightech_category"] = result["industry_code_4d"].map(code_to_cat)

    log.info(f"  METI 4-digit {value_col} LQ: {len(result)} rows")
    return result


# ── HTP Contribution Index ────────────────────────────────────────────────────

def _as_int_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def build_htp_census_category_diagnostic(df_census: pd.DataFrame) -> pd.DataFrame:
    """
    Explain which Economic Census rows enter HTP employment/establishment
    service components. Selected high-tech manufacturing is sourced from METI,
    so Economic Census manufacturing rows are listed as excluded.
    """
    service_codes = {
        code
        for codes in BROAD_HIGHTECH_CENSUS_INDUSTRIES.values()
        for code in codes
    }
    candidates = (
        df_census[["div_code", "industry_label", "hierarchy"]]
        .drop_duplicates()
        .copy()
    )
    candidates["hierarchy_num"] = _as_int_series(candidates["hierarchy"])
    candidates = candidates[
        candidates["div_code"].astype(str).str.startswith(("E", "G", "L"))
    ].copy()

    records = []
    for component in ["hightech_employment_share", "hightech_establishments_share"]:
        for _, row in candidates.sort_values(
            ["div_code", "hierarchy_num", "industry_label"]
        ).iterrows():
            code = str(row["div_code"])
            label = str(row["industry_label"])
            display_code = label.split("_", 1)[0] if "_" in label else code
            hierarchy = row["hierarchy_num"]
            included = bool(code in service_codes and hierarchy == CENSUS_PARENT_HIERARCHY)

            if included:
                reason = "included parent service category"
            elif code == "E":
                reason = (
                    "excluded; selected high-tech manufacturing is sourced from "
                    "METI 2-digit industries"
                )
            elif code.startswith("G") and USE_BROAD_ICT_PARENT:
                reason = "excluded; parent ICT category G is used to avoid double counting"
            elif code == "L":
                reason = (
                    "excluded; parent scientific/professional category L is used "
                    "to avoid double counting"
                )
            else:
                reason = "excluded; not part of HTP broad service component"

            records.append({
                "component": component,
                "industry_code": display_code,
                "industry_name": label,
                "included": included,
                "reason": reason,
            })

    return pd.DataFrame(records)


def aggregate_jpo_patents_by_region(
    df_patents: pd.DataFrame,
    patent_col: str | None = None,
) -> pd.DataFrame:
    """Aggregate manual JPO prefecture patent applications to coursework regions."""
    if df_patents is None or df_patents.empty:
        return pd.DataFrame()

    available = sorted(
        [c for c in df_patents.columns if c.startswith("patent_applications_")]
    )
    if not available:
        raise ValueError("No patent_applications_* column found in JPO patent data")
    patent_col = patent_col or available[-1]

    df = df_patents.copy()
    df["prefecture_code"] = df["prefecture_code"].astype(str).str.zfill(2)
    df[patent_col] = pd.to_numeric(df[patent_col], errors="coerce")
    japan_total = df[patent_col].sum()

    region_codes = {
        "tokyo_ma": ["11", "12", "13", "14"],
        "osaka_kansai": ["26", "27", "28", "29"],
        "japan": sorted(df["prefecture_code"].unique().tolist()),
    }

    records = []
    for region, codes in region_codes.items():
        value = df.loc[df["prefecture_code"].isin(codes), patent_col].sum()
        records.append({
            "region": region,
            "region_label": REGION_LABELS[region],
            patent_col: value,
            "japan_patent_applications": japan_total,
            "patent_applications_share": safe_divide(value, japan_total),
        })

    return pd.DataFrame(records)


def _census_parent_service_value(
    df_census: pd.DataFrame,
    region: str,
    value_col: str,
) -> float:
    service_codes = {
        code
        for codes in BROAD_HIGHTECH_CENSUS_INDUSTRIES.values()
        for code in codes
    }
    hierarchy = _as_int_series(df_census["hierarchy"])
    mask = (
        (df_census["region"] == region) &
        (df_census["div_code"].isin(service_codes)) &
        (hierarchy == CENSUS_PARENT_HIERARCHY)
    )
    return df_census.loc[mask, value_col].sum()


def _meti_hightech_value(df_meti2: pd.DataFrame, region: str, value_col: str) -> float:
    mask = (
        (df_meti2["region"] == region) &
        (df_meti2["industry_code_2d"].isin(ALL_2DIGIT_HIGHTECH_CODES))
    )
    return df_meti2.loc[mask, value_col].sum()

def build_htp_components(
    df_census: pd.DataFrame,
    df_meti2: pd.DataFrame,
    df_meti4: pd.DataFrame,
    df_patents: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Assemble HTP component shares:  metro_value / japan_value.
    Returns one row per region with each component share.
    """
    log.info("Building HTP components")
    if USE_BROAD_ICT_PARENT:
        log.info(
            "HTP broad employment uses parent ICT category G only; "
            "subcategories G1/G2 excluded to avoid double counting."
        )

    records = {}
    for region in ["tokyo_ma", "osaka_kansai"]:
        records[region] = {"region": region, "region_label": REGION_LABELS[region]}

    def _add_component(region: str, name: str, metro_val, japan_val):
        if metro_val is None or japan_val is None or japan_val == 0:
            records[region][name] = None
            log.info(f"  HTP component {name} [{region}]: MISSING (no data)")
        else:
            records[region][name] = metro_val / japan_val
            log.info(f"  HTP component {name} [{region}]: {metro_val/japan_val:.4f}")

    # ── Components 1-2: service parents from Economic Census + selected METI HT manufacturing ──
    for region in ["tokyo_ma", "osaka_kansai"]:
        metro_ht = (
            _census_parent_service_value(df_census, region, "employment") +
            _meti_hightech_value(df_meti2, region, "employment")
        )
        japan_ht = (
            _census_parent_service_value(df_census, "japan", "employment") +
            _meti_hightech_value(df_meti2, "japan", "employment")
        )
        _add_component(region, "hightech_employment_share", metro_ht or None, japan_ht or None)

    for region in ["tokyo_ma", "osaka_kansai"]:
        metro_est = (
            _census_parent_service_value(df_census, region, "establishments") +
            _meti_hightech_value(df_meti2, region, "establishments")
        )
        japan_est = (
            _census_parent_service_value(df_census, "japan", "establishments") +
            _meti_hightech_value(df_meti2, "japan", "establishments")
        )
        _add_component(region, "hightech_establishments_share",
                       metro_est or None, japan_est or None)

    # ── Component 3: High-tech manufacturing value added share (2-digit) ───────
    ht_2d_codes = list(ALL_2DIGIT_HIGHTECH_CODES)
    for region in ["tokyo_ma", "osaka_kansai"]:
        metro_va = df_meti2.loc[
            (df_meti2["region"] == region) &
            (df_meti2["industry_code_2d"].isin(ht_2d_codes)),
            "value_added_mn_yen"
        ].sum()
        japan_va = df_meti2.loc[
            (df_meti2["region"] == "japan") &
            (df_meti2["industry_code_2d"].isin(ht_2d_codes)),
            "value_added_mn_yen"
        ].sum()
        _add_component(region, "hightech_mfg_va_share", metro_va or None, japan_va or None)

    # ── Component 4: High-tech manufacturing shipments share (2-digit) ─────────
    for region in ["tokyo_ma", "osaka_kansai"]:
        metro_sh = df_meti2.loc[
            (df_meti2["region"] == region) &
            (df_meti2["industry_code_2d"].isin(ht_2d_codes)),
            "shipments_mn_yen"
        ].sum()
        japan_sh = df_meti2.loc[
            (df_meti2["region"] == "japan") &
            (df_meti2["industry_code_2d"].isin(ht_2d_codes)),
            "shipments_mn_yen"
        ].sum()
        _add_component(region, "hightech_mfg_shipments_share",
                       metro_sh or None, japan_sh or None)

    # ── Component 5: Patent applications share (if available) ─────────────────
    if df_patents is not None and not df_patents.empty:
        df_pat_region = aggregate_jpo_patents_by_region(df_patents)
        for region in ["tokyo_ma", "osaka_kansai"]:
            row = df_pat_region[df_pat_region["region"] == region]
            if row.empty:
                records[region]["patent_applications_share"] = None
                continue
            records[region]["patent_applications_share"] = row.iloc[0]["patent_applications_share"]
            log.info(
                f"  HTP component patent_applications_share [{region}]: "
                f"{records[region]['patent_applications_share']:.4f}"
            )
    else:
        for region in ["tokyo_ma", "osaka_kansai"]:
            records[region]["patent_applications_share"] = None
        log.warning("  HTP component patent_applications_share: SKIPPED (no JPO data)")

    # ── Assemble and compute index ─────────────────────────────────────────────
    component_cols = [
        "hightech_employment_share",
        "hightech_establishments_share",
        "hightech_mfg_va_share",
        "hightech_mfg_shipments_share",
        "patent_applications_share",
    ]

    result_rows = []
    for region, rec in records.items():
        available = [c for c in component_cols if rec.get(c) is not None]
        values = [rec[c] for c in available]
        mean_share = np.mean(values) if values else None
        rec["htp_contribution_index"] = mean_share
        rec["htp_index_100"] = mean_share * 100 if mean_share is not None else None
        rec["n_components_used"] = len(available)
        rec["components_used"] = "|".join(available)
        rec["components_missing"] = "|".join(
            [c for c in component_cols if rec.get(c) is None]
        )
        result_rows.append(rec)

    df_out = pd.DataFrame(result_rows)
    log.info(f"  HTP index built: {len(df_out)} regions")
    return df_out


# ── World Bank summary stats ──────────────────────────────────────────────────

def worldbank_summary(df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Add annual growth to a World Bank time series."""
    df = df.dropna(subset=["value"]).copy()
    if df.empty:
        log.warning(f"  WB {label}: no non-null data")
        return df

    df = df.sort_values("year")
    first = df.iloc[0]
    last = df.iloc[-1]

    n_years = last["year"] - first["year"]
    cagr = ((last["value"] / first["value"]) ** (1 / n_years) - 1) * 100 if n_years > 0 else None
    change = last["value"] - first["value"]

    log.info(
        f"  WB {label}: {first['year']}={first['value']:.2f} → "
        f"{last['year']}={last['value']:.2f} | CAGR={cagr:.2f}% | change={change:.2f}"
    )
    df["annual_growth_pct"] = df["value"].pct_change() * 100
    return df


def worldbank_period_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Summarize 2010-2024 change, percent change, CAGR, and the years of the
    minimum and maximum values.

    NOTE on min/max year semantics (TASK 6 fix / clarification):
    ``min_value_year`` is the YEAR IN WHICH THE SERIES TAKES ITS MINIMUM VALUE
    (and likewise ``max_value_year``).  These are not the first/last year of the
    window.  Paired ``min_value`` / ``max_value`` columns are emitted so the
    pairing is verifiable at a glance.  The values are taken directly from the
    time series via idxmin/idxmax, so they cannot be swapped.
    """
    series = df.dropna(subset=["value"]).sort_values("year").copy()
    cols = [
        "value_2010", "value_2024",
        "absolute_change_2010_2024", "percent_change_2010_2024", "CAGR_2010_2024",
        "min_value", "min_value_year", "max_value", "max_value_year",
        # legacy aliases (kept for backward compatibility)
        "min_year", "max_year",
    ]
    if series.empty:
        return pd.DataFrame(columns=cols)

    first = series.iloc[0]
    last = series.iloc[-1]
    n_years = last["year"] - first["year"]
    absolute_change = last["value"] - first["value"]
    percent_change = safe_divide(absolute_change, first["value"])
    cagr = (
        ((last["value"] / first["value"]) ** (1 / n_years) - 1) * 100
        if n_years > 0 and first["value"] > 0
        else None
    )

    idx_min = series["value"].idxmin()
    idx_max = series["value"].idxmax()
    min_value = float(series.loc[idx_min, "value"])
    max_value = float(series.loc[idx_max, "value"])
    min_value_year = int(series.loc[idx_min, "year"])
    max_value_year = int(series.loc[idx_max, "year"])

    return pd.DataFrame([{
        "value_2010": first["value"],
        "value_2024": last["value"],
        "absolute_change_2010_2024": absolute_change,
        "percent_change_2010_2024": percent_change * 100 if percent_change is not None else None,
        "CAGR_2010_2024": cagr,
        "min_value": min_value,
        "min_value_year": min_value_year,
        "max_value": max_value,
        "max_value_year": max_value_year,
        # legacy aliases
        "min_year": min_value_year,
        "max_year": max_value_year,
    }])
