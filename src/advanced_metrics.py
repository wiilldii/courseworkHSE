"""
Advanced / corrective metrics added after methodological review.

Contents
--------
TASK 1  HTP national-norm-normalized profile + composite sensitivity
TASK 2  Krugman specialization (dissimilarity) index between the two regions
TASK 3  Value-added per worker by sector and region
TASK 4  LQ employment vs LQ value-added divergence
TASK 5  LQ robustness across bases (employment / establishments / value added)

Design principles
-----------------
* Every normalized index is national-norm-normalized:
      x_norm = (X_region / B_region) / (X_japan / B_japan)
  A value of 1.0 means the region sits exactly at the national average on that
  dimension; >1 is above average, <1 below.  This is the SAME logic as a location
  quotient and deliberately replaces the old "share of national total" measure,
  which conflated specialization with sheer economic size.
* The deliverable is the PROFILE (the shape across components), not a single
  ranking.  A scalar composite is provided only as a labelled summary, with an
  explicit weight-sensitivity check.
* All monetary values are in million yen (``*_mn_yen``) unless a column name says
  otherwise (MIC sales columns are in 億円 / 100mn yen and are not used here).

These functions do NOT modify the existing LQ computations; they only add new
outputs.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from src.config import (
    REGION_LABELS,
    CENSUS_TOTAL_CODE,
    ALL_2DIGIT_HIGHTECH_CODES,
    MFG_2DIGIT_ENGLISH,
    MFG_2DIGIT_HIGHTECH,
    MFG_4DIGIT_HIGHTECH,
    TOKYO_MA_PREF_CODES,
    OSAKA_KANSAI_PREF_CODES,
)
from src.metrics import (
    _census_parent_service_value,
    _meti_hightech_value,
    aggregate_jpo_patents_by_region,
)
from src.utils import safe_divide

log = logging.getLogger("coursework")

METRO_REGIONS = ["tokyo_ma", "osaka_kansai"]

# Construct grouping: which conceptual thing each component actually measures.
#   hq_corporate_innovation -> driven by where corporate HQs sit (services live
#       near HQs; patents are filed at the first applicant's HQ address).
#   industrial_production    -> physical manufacturing output on the ground.
COMPONENT_CONSTRUCT = {
    "hightech_employment": "hq_corporate_innovation",
    "hightech_establishments": "hq_corporate_innovation",
    "patent_applications": "hq_corporate_innovation",
    "hightech_mfg_value_added": "industrial_production",
    "hightech_mfg_shipments": "industrial_production",
}

SINGLE_PREF_THRESHOLD = 0.90  # flag if one prefecture supplies >= 90% of a metro value


# ── shared accessors ──────────────────────────────────────────────────────────

def _census_total(df_census: pd.DataFrame, region: str, value_col: str) -> float:
    """Region total across all industries (Economic Census 'AR' row)."""
    return df_census.loc[
        (df_census["region"] == region) & (df_census["div_code"] == CENSUS_TOTAL_CODE),
        value_col,
    ].sum()


def _meti2_total(df_meti2: pd.DataFrame, region: str, value_col: str) -> float:
    """Region all-manufacturing total (METI 2-digit '00' row)."""
    return df_meti2.loc[
        (df_meti2["region"] == region) & (df_meti2["industry_code_2d"] == "00"),
        value_col,
    ].sum()


def _region_pref_codes(region: str) -> list[str]:
    return TOKYO_MA_PREF_CODES if region == "tokyo_ma" else OSAKA_KANSAI_PREF_CODES


def _dominant_pref_share(
    df_raw: pd.DataFrame,
    region: str,
    value_col: str,
    code_col: str,
    code_filter,
) -> tuple[float | None, str | None]:
    """
    Within a metro region, what share of `value_col` (summed over `code_filter`
    industries) comes from the single largest prefecture?  Used to flag values
    that rely on one prefecture rather than the whole agglomeration.
    Returns (dominant_share, dominant_pref_code).
    """
    codes = _region_pref_codes(region)
    subset = df_raw[df_raw["pref_code"].isin(codes)]
    if code_filter is not None:
        subset = subset[subset[code_col].isin(list(code_filter))]
    total = subset[value_col].sum()
    if not total or total <= 0:
        return None, None
    by_pref = subset.groupby("pref_code")[value_col].sum()
    if by_pref.empty:
        return None, None
    dom_code = by_pref.idxmax()
    return by_pref.max() / total, dom_code


def _dominant_pref_share_patents(
    df_patents: pd.DataFrame, region: str
) -> tuple[float | None, str | None]:
    """Largest single-prefecture share of a metro region's patent applications."""
    pat_col = [c for c in df_patents.columns if c.startswith("patent_applications_")]
    if not pat_col:
        return None, None
    pat_col = sorted(pat_col)[-1]  # latest year
    codes = _region_pref_codes(region)
    sub = df_patents.copy()
    sub["prefecture_code"] = sub["prefecture_code"].astype(str).str.zfill(2)
    sub = sub[sub["prefecture_code"].isin(codes)]
    total = pd.to_numeric(sub[pat_col], errors="coerce").sum()
    if not total or total <= 0:
        return None, None
    by_pref = pd.to_numeric(sub.set_index("prefecture_code")[pat_col], errors="coerce")
    return by_pref.max() / total, by_pref.idxmax()


# ============================================================================
# TASK 1 — HTP national-norm-normalized profile
# ============================================================================

def _patent_region_value(df_patents: pd.DataFrame | None, region: str) -> tuple[float | None, float | None]:
    """Return (region_patents, japan_patents) for the latest patent year."""
    if df_patents is None or df_patents.empty:
        return None, None
    agg = aggregate_jpo_patents_by_region(df_patents)
    pat_col = [c for c in agg.columns if c.startswith("patent_applications_")]
    if not pat_col:
        return None, None
    pat_col = pat_col[0]
    reg_row = agg[agg["region"] == region]
    if reg_row.empty:
        return None, None
    region_val = float(reg_row.iloc[0][pat_col])
    japan_val = float(reg_row.iloc[0]["japan_patent_applications"])
    return region_val, japan_val


def build_htp_profile_normalized(
    df_census: pd.DataFrame,
    df_meti2: pd.DataFrame,
    df_meti4: pd.DataFrame,
    df_patents: pd.DataFrame | None = None,
    df_meti2_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    TASK 1.  National-norm-normalized HTP profile (long format).

    For each component k and region r:
        intensity_r      = X[k,r] / B[k,r]
        intensity_japan  = X[k,Japan] / B[k,Japan]
        normalized_value = intensity_r / intensity_japan

    `normalized_value` = 1 means the region is at the national average.
    Includes per-component rows AND per-construct mean rows (row_type column).
    """
    log.info("TASK 1: Building HTP national-norm-normalized profile")

    # Component definitions: (component, base_name, value_unit, numerator_fn, base_fn)
    # numerator_fn / base_fn take (region) and return a float (in the stated unit).
    def num_ht_emp(region):
        return (_census_parent_service_value(df_census, region, "employment")
                + _meti_hightech_value(df_meti2, region, "employment"))

    def num_ht_est(region):
        return (_census_parent_service_value(df_census, region, "establishments")
                + _meti_hightech_value(df_meti2, region, "establishments"))

    def num_ht_va(region):
        return _meti_hightech_value(df_meti2, region, "value_added_mn_yen")

    def num_ht_ship(region):
        return _meti_hightech_value(df_meti2, region, "shipments_mn_yen")

    components = [
        {
            "component": "hightech_employment",
            "base_name": "total_employment_all_industries",
            "value_unit": "persons",
            "num_fn": num_ht_emp,
            "base_fn": lambda r: _census_total(df_census, r, "employment"),
        },
        {
            "component": "hightech_establishments",
            "base_name": "total_establishments_all_industries",
            "value_unit": "establishments",
            "num_fn": num_ht_est,
            "base_fn": lambda r: _census_total(df_census, r, "establishments"),
        },
        {
            "component": "hightech_mfg_value_added",
            "base_name": "total_manufacturing_value_added_mn_yen",
            "value_unit": "mn_yen",
            "num_fn": num_ht_va,
            "base_fn": lambda r: _meti2_total(df_meti2, r, "value_added_mn_yen"),
        },
        {
            "component": "hightech_mfg_shipments",
            "base_name": "total_manufacturing_shipments_mn_yen",
            "value_unit": "mn_yen",
            "num_fn": num_ht_ship,
            "base_fn": lambda r: _meti2_total(df_meti2, r, "shipments_mn_yen"),
        },
        {
            "component": "patent_applications",
            "base_name": "total_employment_all_industries",
            "value_unit": "applications",
            "num_fn": None,  # handled specially (patents)
            "base_fn": lambda r: _census_total(df_census, r, "employment"),
        },
    ]

    rows = []
    norm_lookup: dict[tuple[str, str], float | None] = {}

    for comp in components:
        name = comp["component"]
        construct = COMPONENT_CONSTRUCT[name]

        # Japan-level numerator
        if name == "patent_applications":
            # patents need region + japan together
            pass

        for region in METRO_REGIONS:
            if name == "patent_applications":
                region_num, japan_num = _patent_region_value(df_patents, region)
            else:
                region_num = comp["num_fn"](region)
                japan_num = comp["num_fn"]("japan")

            region_base = comp["base_fn"](region)
            japan_base = comp["base_fn"]("japan")

            intensity_r = safe_divide(region_num, region_base)
            intensity_j = safe_divide(japan_num, japan_base)
            normalized = safe_divide(intensity_r, intensity_j)
            norm_lookup[(region, name)] = normalized

            # single-prefecture reliance flag (manufacturing-side proxy / patents)
            note = ""
            if df_meti2_raw is not None and name in (
                "hightech_employment", "hightech_establishments",
                "hightech_mfg_value_added", "hightech_mfg_shipments",
            ):
                raw_col = {
                    "hightech_employment": "employment",
                    "hightech_establishments": "establishments",
                    "hightech_mfg_value_added": "value_added_mn_yen",
                    "hightech_mfg_shipments": "shipments_mn_yen",
                }[name]
                share, dom = _dominant_pref_share(
                    df_meti2_raw, region, raw_col, "industry_code_2d",
                    ALL_2DIGIT_HIGHTECH_CODES,
                )
                if share is not None and share >= SINGLE_PREF_THRESHOLD:
                    note = (f"high-tech mfg part {share:.0%} concentrated in "
                            f"prefecture {dom}")
            if name == "patent_applications" and df_patents is not None and not df_patents.empty:
                share, dom = _dominant_pref_share_patents(df_patents, region)
                if share is not None and share >= SINGLE_PREF_THRESHOLD:
                    note = f"{share:.0%} of metro patents from prefecture {dom}"

            rows.append({
                "row_type": "component",
                "region": region,
                "region_label": REGION_LABELS[region],
                "component_or_construct": name,
                "construct": construct,
                "base_name": comp["base_name"],
                "value_unit": comp["value_unit"],
                "hightech_value_region": region_num,
                "base_value_region": region_base,
                "intensity_region": intensity_r,
                "hightech_value_japan": japan_num,
                "base_value_japan": japan_base,
                "intensity_japan": intensity_j,
                "normalized_value": normalized,
                "note": note,
            })

    # per-construct mean rows
    constructs = sorted(set(COMPONENT_CONSTRUCT.values()))
    for region in METRO_REGIONS:
        for construct in constructs:
            members = [c for c, cc in COMPONENT_CONSTRUCT.items() if cc == construct]
            vals = [norm_lookup.get((region, m)) for m in members]
            vals = [v for v in vals if v is not None]
            mean_norm = float(np.mean(vals)) if vals else None
            rows.append({
                "row_type": "construct_mean",
                "region": region,
                "region_label": REGION_LABELS[region],
                "component_or_construct": construct,
                "construct": construct,
                "base_name": "",
                "value_unit": "ratio_vs_national_norm",
                "hightech_value_region": None,
                "base_value_region": None,
                "intensity_region": None,
                "hightech_value_japan": None,
                "base_value_japan": None,
                "intensity_japan": None,
                "normalized_value": mean_norm,
                "note": f"mean of {len(vals)} normalized components",
            })

    df = pd.DataFrame(rows)
    log.info(f"  HTP normalized profile: {len(df)} rows "
             f"({df['row_type'].eq('component').sum()} components + "
             f"{df['row_type'].eq('construct_mean').sum()} construct means)")
    return df


def build_htp_composite_sensitivity(df_profile: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 1.  Optional scalar composite under 3 weight schemes, with a stability
    check.  Composite = weighted mean of available normalized component values.
    Reports whether the region ranking is stable across schemes.
    """
    log.info("TASK 1: Building HTP composite sensitivity")

    comp = df_profile[df_profile["row_type"] == "component"].copy()

    # weight schemes: weight per construct
    schemes = {
        "equal": {
            "hq_corporate_innovation": 1.0,
            "industrial_production": 1.0,
            "description": "equal weight on every normalized component",
        },
        "hq_weighted": {
            "hq_corporate_innovation": 2.0,
            "industrial_production": 1.0,
            "description": "HQ/corporate-innovation construct weighted 2x",
        },
        "production_weighted": {
            "hq_corporate_innovation": 1.0,
            "industrial_production": 2.0,
            "description": "industrial-production construct weighted 2x",
        },
    }

    rows = []
    composites = {}  # scheme -> {region: composite}
    for scheme, cfg in schemes.items():
        composites[scheme] = {}
        for region in METRO_REGIONS:
            sub = comp[(comp["region"] == region) & comp["normalized_value"].notna()]
            if sub.empty:
                composites[scheme][region] = None
                continue
            weights = sub["construct"].map(
                {k: v for k, v in cfg.items() if k != "description"}
            ).astype(float)
            vals = sub["normalized_value"].astype(float)
            composite = float(np.average(vals, weights=weights))
            composites[scheme][region] = composite

    # determine stability of ordering
    higher_each = []
    for scheme in schemes:
        t = composites[scheme]["tokyo_ma"]
        o = composites[scheme]["osaka_kansai"]
        if t is None or o is None:
            higher_each.append(None)
        else:
            higher_each.append("tokyo_ma" if t >= o else "osaka_kansai")
    ranking_stable = len(set(h for h in higher_each if h is not None)) <= 1

    for scheme, cfg in schemes.items():
        t = composites[scheme]["tokyo_ma"]
        o = composites[scheme]["osaka_kansai"]
        higher = ("tokyo_ma" if (t is not None and o is not None and t >= o)
                  else "osaka_kansai" if (t is not None and o is not None) else None)
        rows.append({
            "weight_scheme": scheme,
            "description": cfg["description"],
            "tokyo_ma_composite": t,
            "osaka_kansai_composite": o,
            "tokyo_over_osaka_ratio": safe_divide(t, o),
            "higher_region": higher,
            "ranking_stable_across_schemes": ranking_stable,
        })

    df = pd.DataFrame(rows)
    log.info(f"  HTP composite sensitivity: ranking_stable={ranking_stable}")
    return df


# ============================================================================
# TASK 2 — Krugman specialization (dissimilarity) index
# ============================================================================

def build_krugman_specialization_index(df_meti2: pd.DataFrame) -> pd.DataFrame:
    """
    TASK 2.  Krugman specialization (dissimilarity) index between Tokyo MA and
    Osaka/Kansai, computed on 2-digit manufacturing employment shares and on
    value-added shares.

        s_i(region) = industry_i value / region manufacturing total
        KSI         = 0.5 * sum_i | s_i(Tokyo) - s_i(Osaka) |

    Output: one row per 2-digit industry with shares and absolute contribution
    under both bases, plus a final TOTAL_KSI row.
    """
    log.info("TASK 2: Building Krugman specialization index")

    df = df_meti2[df_meti2["industry_code_2d"] != "00"].copy()

    def _shares(region, value_col):
        sub = df[df["region"] == region]
        total = _meti2_total(df_meti2, region, value_col)
        s = sub.set_index("industry_code_2d")[value_col] / total if total else sub.set_index("industry_code_2d")[value_col] * np.nan
        return s

    industries = sorted(df["industry_code_2d"].unique())

    # Japanese name fallback for non-high-tech 2-digit industries
    jp_names = (
        df.dropna(subset=["industry_name_jp"])
        .drop_duplicates("industry_code_2d")
        .set_index("industry_code_2d")["industry_name_jp"]
        .to_dict()
    )

    emp_tokyo = _shares("tokyo_ma", "employment")
    emp_osaka = _shares("osaka_kansai", "employment")
    va_tokyo = _shares("tokyo_ma", "value_added_mn_yen")
    va_osaka = _shares("osaka_kansai", "value_added_mn_yen")

    rows = []
    for code in industries:
        s_t_e = float(emp_tokyo.get(code, np.nan))
        s_o_e = float(emp_osaka.get(code, np.nan))
        s_t_v = float(va_tokyo.get(code, np.nan))
        s_o_v = float(va_osaka.get(code, np.nan))
        rows.append({
            "industry_code_2d": code,
            "industry_en": MFG_2DIGIT_ENGLISH.get(code, jp_names.get(code, code)),
            "is_hightech": code in ALL_2DIGIT_HIGHTECH_CODES,
            "share_tokyo_employment": s_t_e,
            "share_osaka_employment": s_o_e,
            "abs_diff_employment": abs(s_t_e - s_o_e),
            "share_tokyo_value_added": s_t_v,
            "share_osaka_value_added": s_o_v,
            "abs_diff_value_added": abs(s_t_v - s_o_v),
        })

    df_out = pd.DataFrame(rows).sort_values("abs_diff_employment", ascending=False)

    ksi_emp = 0.5 * df_out["abs_diff_employment"].sum()
    ksi_va = 0.5 * df_out["abs_diff_value_added"].sum()

    total_row = {
        "industry_code_2d": "TOTAL_KSI",
        "industry_en": "Krugman index = 0.5 * sum(|share diff|)",
        "is_hightech": "",
        "share_tokyo_employment": "",
        "share_osaka_employment": "",
        "abs_diff_employment": ksi_emp,
        "share_tokyo_value_added": "",
        "share_osaka_value_added": "",
        "abs_diff_value_added": ksi_va,
    }
    df_out = pd.concat([df_out, pd.DataFrame([total_row])], ignore_index=True)

    log.info(f"  Krugman index: employment={ksi_emp:.4f}, value_added={ksi_va:.4f} "
             "(0 = identical structure, 1 = completely different)")
    return df_out


# ============================================================================
# TASK 3 — Value-added per worker by sector and region
# ============================================================================

def _va_per_worker_level(
    df_meti: pd.DataFrame,
    code_col: str,
    label_map: dict,
    hightech_codes: set,
    level_name: str,
    df_raw: pd.DataFrame | None,
) -> pd.DataFrame:
    df = df_meti[df_meti[code_col].isin(list(hightech_codes))].copy()

    # Japan VA/worker per industry (denominator for the ratio)
    japan = df[df["region"] == "japan"].set_index(code_col)
    japan_vapw = (japan["value_added_mn_yen"] / japan["employment"]).to_dict()

    rows = []
    for region in ["japan"] + METRO_REGIONS:
        sub = df[df["region"] == region]
        for _, r in sub.iterrows():
            code = r[code_col]
            emp = r["employment"]
            va = r["value_added_mn_yen"]
            vapw = safe_divide(va, emp)
            ratio = safe_divide(vapw, japan_vapw.get(code))
            note = ""
            if region in METRO_REGIONS and df_raw is not None and code_col == "industry_code_2d":
                share, dom = _dominant_pref_share(
                    df_raw, region, "employment", code_col, [code]
                )
                if share is not None and share >= SINGLE_PREF_THRESHOLD:
                    note = f"{share:.0%} of metro employment in prefecture {dom}"
            rows.append({
                "level": level_name,
                "industry_code": code,
                "industry_label": label_map.get(code, r.get("industry_name_jp", code)),
                "region": region,
                "region_label": REGION_LABELS[region],
                "employment": emp,
                "value_added_mn_yen": va,
                "va_per_worker_mn_yen": vapw,
                "va_per_worker_vs_japan": ratio if region != "japan" else 1.0,
                "single_prefecture_note": note,
            })
    return pd.DataFrame(rows)


def build_va_per_worker_by_sector(
    df_meti2: pd.DataFrame,
    df_meti4: pd.DataFrame,
    df_meti2_raw: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    TASK 3.  Value added per worker (million yen per worker) for each high-tech
    industry, for Tokyo MA, Osaka/Kansai and Japan, with a ratio versus the
    national VA/worker.  Separates "many low-value jobs" from "high-value niche".
    Covers 2-digit and 4-digit levels (see `level` column).
    """
    log.info("TASK 3: Building value-added per worker by sector")

    code_to_label_4d = {}
    for cat, codes in MFG_4DIGIT_HIGHTECH.items():
        for c in codes:
            code_to_label_4d[c] = cat

    df2 = _va_per_worker_level(
        df_meti2, "industry_code_2d", MFG_2DIGIT_ENGLISH,
        ALL_2DIGIT_HIGHTECH_CODES, "2-digit", df_meti2_raw,
    )
    from src.config import ALL_4DIGIT_HIGHTECH_CODES
    df4 = _va_per_worker_level(
        df_meti4, "industry_code_4d", code_to_label_4d,
        ALL_4DIGIT_HIGHTECH_CODES, "4-digit", None,
    )

    out = pd.concat([df2, df4], ignore_index=True)
    log.info(f"  VA/worker table: {len(out)} rows "
             f"({len(df2)} at 2-digit, {len(df4)} at 4-digit)")
    return out


# ============================================================================
# TASK 4 — LQ employment vs LQ value-added divergence
# ============================================================================

def _merge_lq_pair(lq_emp: pd.DataFrame, lq_va: pd.DataFrame,
                   code_col: str, level_name: str,
                   label_col: str | None) -> pd.DataFrame:
    e = lq_emp[lq_emp["is_hightech"]][[
        "region", "region_label", code_col, "lq"
    ]].rename(columns={"lq": "lq_employment"})
    v = lq_va[lq_va["is_hightech"]][["region", code_col, "lq"]].rename(
        columns={"lq": "lq_value_added"}
    )
    merged = pd.merge(e, v, on=["region", code_col], how="outer")
    if label_col and label_col in lq_emp.columns:
        labels = lq_emp[[code_col, label_col]].drop_duplicates()
        merged = merged.merge(labels, on=code_col, how="left")
        merged = merged.rename(columns={label_col: "industry_label"})
    merged["level"] = level_name
    merged["lq_va_minus_lq_emp"] = (
        merged["lq_value_added"] - merged["lq_employment"]
    )
    merged["holds_high_value_end"] = merged["lq_va_minus_lq_emp"] > 0
    return merged


def build_lq_emp_vs_va_divergence(
    lq_mfg2_emp: pd.DataFrame,
    lq_mfg2_va: pd.DataFrame,
    lq_mfg4_emp: pd.DataFrame,
    lq_mfg4_va: pd.DataFrame,
) -> pd.DataFrame:
    """
    TASK 4.  Side-by-side LQ on employment and value added per high-tech
    industry/region, with their difference.  lq_va > lq_emp means the region
    holds the high-value end of that industry.  Flags the largest positive
    divergences per region.
    """
    log.info("TASK 4: Building LQ employment-vs-value-added divergence")

    d2 = _merge_lq_pair(lq_mfg2_emp, lq_mfg2_va, "industry_code_2d",
                        "2-digit", "industry_en")
    d2 = d2.rename(columns={"industry_code_2d": "industry_code"})
    d4 = _merge_lq_pair(lq_mfg4_emp, lq_mfg4_va, "industry_code_4d",
                        "4-digit", "hightech_category")
    d4 = d4.rename(columns={"industry_code_4d": "industry_code"})

    out = pd.concat([d2, d4], ignore_index=True)

    # rank top positive divergences per region (within each level)
    out["divergence_rank_in_region"] = (
        out.groupby(["region", "level"])["lq_va_minus_lq_emp"]
        .rank(ascending=False, method="min")
    )
    out["is_top3_high_value_divergence"] = (
        (out["divergence_rank_in_region"] <= 3) & (out["lq_va_minus_lq_emp"] > 0)
    )

    out = out.sort_values(
        ["level", "region", "lq_va_minus_lq_emp"], ascending=[True, True, False]
    )
    log.info(f"  LQ divergence table: {len(out)} rows")
    return out


# ============================================================================
# TASK 5 — LQ robustness across bases
# ============================================================================

def build_lq_robustness_across_bases(
    lq_mfg2_emp: pd.DataFrame,
    lq_mfg2_est: pd.DataFrame,
    lq_mfg2_va: pd.DataFrame,
) -> pd.DataFrame:
    """
    TASK 5.  For 2-digit high-tech industries, put LQ computed on employment,
    establishments and value added side by side, rank each industry under each
    base (within region), and flag whether the high/low classification
    (LQ>1 vs LQ<1) is stable across all three bases.
    """
    log.info("TASK 5: Building LQ robustness across bases")

    def _ht(df, name):
        return df[df["is_hightech"]][["region", "region_label",
                                      "industry_code_2d", "industry_en", "lq"]].rename(
            columns={"lq": name}
        )

    e = _ht(lq_mfg2_emp, "lq_employment")
    s = _ht(lq_mfg2_est, "lq_establishments")[["region", "industry_code_2d", "lq_establishments"]]
    v = _ht(lq_mfg2_va, "lq_value_added")[["region", "industry_code_2d", "lq_value_added"]]

    out = e.merge(s, on=["region", "industry_code_2d"], how="outer")
    out = out.merge(v, on=["region", "industry_code_2d"], how="outer")

    for base in ["lq_employment", "lq_establishments", "lq_value_added"]:
        out[f"rank_{base.split('_', 1)[1]}"] = (
            out.groupby("region")[base].rank(ascending=False, method="min")
        )

    lq_cols = ["lq_employment", "lq_establishments", "lq_value_added"]
    out["all_above_1"] = (out[lq_cols] > 1).all(axis=1)
    out["all_below_1"] = (out[lq_cols] < 1).all(axis=1)
    out["classification_stable"] = out["all_above_1"] | out["all_below_1"]

    out = out.sort_values(["region", "lq_employment"], ascending=[True, False])
    n_stable = int(out["classification_stable"].sum())
    log.info(f"  LQ robustness: {n_stable}/{len(out)} region-industry rows have "
             "a stable LQ>1 vs <1 classification across all three bases")
    return out
