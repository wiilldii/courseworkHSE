"""
Chart generation. All plots saved as PNG at 300 dpi.
No seaborn — matplotlib only.
"""

import logging
import warnings
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

from src.config import OUT_FIGS, MFG_2DIGIT_ENGLISH, REGION_LABELS

# Use Meiryo for CJK support on Windows
try:
    plt.rcParams["font.family"] = ["Meiryo", "DejaVu Sans"]
except Exception:
    pass

log = logging.getLogger("coursework")

COLORS = {
    "tokyo_ma": "#1f4e79",
    "osaka_kansai": "#c00000",
    "neutral": "#404040",
}

REGION_ORDER = ["tokyo_ma", "osaka_kansai"]
REGION_COLOR = [COLORS["tokyo_ma"], COLORS["osaka_kansai"]]


def _save(fig, name: str):
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    path = OUT_FIGS / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    log.info(f"  Saved figure: {path.name}")


# ── LQ 2-digit manufacturing profile ─────────────────────────────────────────

def plot_lq_mfg_2digit(df_lq: pd.DataFrame, value_col: str = "employment"):
    """
    Horizontal bar chart: LQ for high-tech 2-digit manufacturing,
    Tokyo MA vs Osaka/Kansai side-by-side.
    """
    from src.config import ALL_2DIGIT_HIGHTECH_CODES

    df = df_lq[df_lq["industry_code_2d"].isin(ALL_2DIGIT_HIGHTECH_CODES)].copy()
    df["industry_en"] = df["industry_code_2d"].map(MFG_2DIGIT_ENGLISH).fillna(
        df["industry_code_2d"]
    )

    industries = df["industry_en"].unique().tolist()
    x = np.arange(len(industries))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (region, color) in enumerate(zip(REGION_ORDER, REGION_COLOR)):
        vals = [
            df.loc[(df["region"] == region) & (df["industry_en"] == ind), "lq"].values
            for ind in industries
        ]
        vals = [v[0] if len(v) > 0 and v[0] is not None else 0 for v in vals]
        ax.bar(x + i * width - width / 2, vals, width, label=REGION_LABELS[region],
               color=color, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--", label="LQ = 1")
    ax.set_xticks(x)
    ax.set_xticklabels(industries, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Location Quotient (LQ)")
    ax.set_title(f"Location Quotient – High-Tech Manufacturing (2-digit)\nby {value_col.replace('_', ' ').title()}")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    _save(fig, "lq_mfg_2digit_profile.png")


# ── LQ 4-digit selected high-tech ─────────────────────────────────────────────

def plot_lq_mfg_4digit_selected(df_lq: pd.DataFrame):
    """
    LQ for selected high-tech 4-digit categories, aggregated by category.
    """
    from src.config import MFG_4DIGIT_HIGHTECH, ALL_4DIGIT_HIGHTECH_CODES

    df = df_lq[df_lq["is_hightech"]].copy()
    if df.empty:
        log.warning("  plot_lq_mfg_4digit_selected: no high-tech rows")
        return

    # Average LQ within each high-tech category
    df_cat = (
        df.groupby(["region", "hightech_category"])["lq"]
        .mean()
        .reset_index()
        .dropna(subset=["lq"])
    )

    categories = [c for c in MFG_4DIGIT_HIGHTECH.keys() if c in df_cat["hightech_category"].values]
    if not categories:
        log.warning("  plot_lq_mfg_4digit_selected: no categories found")
        return

    cat_labels = {
        "pharmaceuticals": "Pharmaceuticals",
        "semiconductor_equipment_robotics": "Semi. Equip. & Robotics",
        "measurement_medical_precision": "Measurement & Medical",
        "electronic_components_semiconductors": "Electronic Components",
        "electrical_industrial_equipment": "Electrical Equipment",
        "communication_computer_equipment": "Comm. & Computer Equip.",
    }

    x = np.arange(len(categories))
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))

    for i, (region, color) in enumerate(zip(REGION_ORDER, REGION_COLOR)):
        vals = [
            df_cat.loc[
                (df_cat["region"] == region) &
                (df_cat["hightech_category"] == cat), "lq"
            ].values
            for cat in categories
        ]
        vals = [v[0] if len(v) > 0 else 0 for v in vals]
        ax.bar(x + i * width - width / 2, vals, width,
               label=REGION_LABELS[region], color=color, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [cat_labels.get(c, c) for c in categories],
        rotation=30, ha="right", fontsize=8
    )
    ax.set_ylabel("Location Quotient (LQ, mean within category)")
    ax.set_title("Location Quotient – High-Tech Manufacturing (4-digit categories)\nby Employment")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    _save(fig, "lq_mfg_4digit_selected_profile.png")


# ── HTP components bar chart ──────────────────────────────────────────────────

def plot_htp_components(df_htp: pd.DataFrame):
    """Grouped bar chart of HTP component shares (% of Japan)."""
    component_cols = [
        "hightech_employment_share",
        "hightech_establishments_share",
        "hightech_mfg_va_share",
        "hightech_mfg_shipments_share",
        "patent_applications_share",
    ]
    labels = {
        "hightech_employment_share": "HT Employment\nShare",
        "hightech_establishments_share": "HT Establishments\nShare",
        "hightech_mfg_va_share": "HT Mfg. Value\nAdded Share",
        "hightech_mfg_shipments_share": "HT Mfg.\nShipments Share",
        "patent_applications_share": "Patent Apps.\nShare",
    }

    present = [c for c in component_cols if c in df_htp.columns and
               df_htp[c].notna().any()]
    if not present:
        log.warning("  plot_htp_components: no component data")
        return

    x = np.arange(len(present))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (region, color) in enumerate(zip(REGION_ORDER, REGION_COLOR)):
        row = df_htp[df_htp["region"] == region]
        if row.empty:
            continue
        vals = [row[c].values[0] * 100 if row[c].values[0] is not None else 0
                for c in present]
        ax.bar(x + i * width - width / 2, vals, width,
               label=REGION_LABELS[region], color=color, alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in present], fontsize=8)
    ax.set_ylabel("Share of Japan Total (%)")
    ax.set_title("HTP Index Components: Share of Japan Total")
    ax.legend(fontsize=9)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    _save(fig, "htp_components_bar.png")


# ── World Bank high-tech exports ──────────────────────────────────────────────

def plot_wb_exports_usd(df: pd.DataFrame):
    """Line chart: Japan high-tech exports in current USD."""
    df = df.dropna(subset=["value"]).sort_values("year")
    if df.empty:
        log.warning("  plot_wb_exports_usd: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["year"], df["value"] / 1e9, marker="o", markersize=4,
            color=COLORS["neutral"], linewidth=1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("USD billion")
    ax.set_title("Japan High-Technology Exports (current USD)\nSource: World Bank WDI")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, "japan_hightech_exports_usd.png")


def plot_wb_exports_share(df: pd.DataFrame):
    """Line chart: Japan high-tech exports as % of manufactured exports."""
    df = df.dropna(subset=["value"]).sort_values("year")
    if df.empty:
        log.warning("  plot_wb_exports_share: no data")
        return

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(df["year"], df["value"], marker="o", markersize=4,
            color=COLORS["neutral"], linewidth=1.8)
    ax.set_xlabel("Year")
    ax.set_ylabel("% of manufactured exports")
    ax.set_title("Japan High-Technology Exports (% of Manufactured Exports)\nSource: World Bank WDI")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, "japan_hightech_exports_share_manufactured.png")


# ── Technology export/import balance ─────────────────────────────────────────

def plot_tech_trade_balance(df: pd.DataFrame):
    """
    Bar chart of technology trade balance by industry.
    Shows top industries by absolute balance.
    """
    df = df.dropna(subset=["tech_balance_mn_yen"]).copy()
    df = df[df["hierarchy"].isin([2, 3, "2", "3"])].copy()

    # Top 12 by absolute balance
    df["abs_balance"] = df["tech_balance_mn_yen"].abs()
    df = df.nlargest(12, "abs_balance").sort_values("tech_balance_mn_yen")

    if df.empty:
        log.warning("  plot_tech_trade_balance: no data after filtering")
        return

    colors = ["#c00000" if v < 0 else "#1f4e79" for v in df["tech_balance_mn_yen"]]

    fig, ax = plt.subplots(figsize=(9, 6))
    y = np.arange(len(df))
    ax.barh(y, df["tech_balance_mn_yen"] / 1000, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(df["industry_jp"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Technology Trade Balance (billion yen)")
    ax.set_title("Japan Technology Trade Balance by Industry (2024)\n"
                 "Source: MIC Survey of R&D")
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, "technology_export_import_balance.png")


def plot_htp_profile_normalized(df_profile: pd.DataFrame):
    """
    Normalized HTP profile (national norm = 1.0).  Grouped bars per component
    with a reference line at 1.0 so the chart reads as 'different profiles',
    not 'who is bigger'.
    """
    comp = df_profile[df_profile["row_type"] == "component"].copy()
    if comp.empty:
        log.warning("  plot_htp_profile_normalized: no component rows")
        return

    order = [
        "hightech_employment", "hightech_establishments",
        "patent_applications",
        "hightech_mfg_value_added", "hightech_mfg_shipments",
    ]
    labels = {
        "hightech_employment": "HT\nEmployment",
        "hightech_establishments": "HT\nEstablishments",
        "patent_applications": "Patent\nApplications",
        "hightech_mfg_value_added": "HT Mfg.\nValue Added",
        "hightech_mfg_shipments": "HT Mfg.\nShipments",
    }
    order = [c for c in order if c in comp["component_or_construct"].unique()]
    x = np.arange(len(order))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    for i, (region, color) in enumerate(zip(REGION_ORDER, REGION_COLOR)):
        vals = []
        for c in order:
            row = comp[(comp["region"] == region) &
                       (comp["component_or_construct"] == c)]
            v = row["normalized_value"].values
            vals.append(v[0] if len(v) and pd.notna(v[0]) else 0)
        ax.bar(x + i * width - width / 2, vals, width,
               label=REGION_LABELS[region], color=color, alpha=0.85)

    ax.axhline(1.0, color="black", linewidth=1.0, linestyle="--",
               label="National norm = 1.0")
    ax.set_xticks(x)
    ax.set_xticklabels([labels.get(c, c) for c in order], fontsize=8)
    ax.set_ylabel("Normalized intensity (Japan = 1.0)")
    ax.set_title("HTP Normalized Profile: specialization relative to the national average\n"
                 "(values >1 = above national norm; this shows PROFILE, not size)")
    ax.legend(fontsize=8)
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, "htp_profile_normalized.png")


def plot_selected_hightech_technology_balance(df: pd.DataFrame):
    """Optional focused bar chart for selected high-tech technology balance."""
    if df is None or df.empty:
        log.warning("  plot_selected_hightech_technology_balance: no data")
        return

    plot_df = df.dropna(subset=["tech_balance_mn_yen"]).copy()
    if plot_df.empty:
        log.warning("  plot_selected_hightech_technology_balance: no non-null balance")
        return

    plot_df = plot_df.sort_values("tech_balance_mn_yen")
    colors = ["#c00000" if v < 0 else "#1f4e79" for v in plot_df["tech_balance_mn_yen"]]

    fig, ax = plt.subplots(figsize=(9, 5))
    y = np.arange(len(plot_df))
    ax.barh(y, plot_df["tech_balance_mn_yen"] / 1000, color=colors, alpha=0.85)
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["industry_en"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Technology Trade Balance (billion yen)")
    ax.set_title("Selected High-Tech Technology Trade Balance (2024)\n"
                 "Source: MIC Survey of R&D")
    ax.xaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)
    _save(fig, "selected_hightech_technology_balance.png")
