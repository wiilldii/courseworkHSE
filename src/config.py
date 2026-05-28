"""
Project-wide constants for Tokyo/Osaka high-tech industry analysis.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROC = ROOT / "data" / "processed"
OUT_TABLES = ROOT / "outputs" / "tables"
OUT_FIGS = ROOT / "outputs" / "figures"
OUT_LOGS = ROOT / "outputs" / "logs"

# ── Geography ─────────────────────────────────────────────────────────────────

# Economic Census (b1_006_1e): area classification codes
JAPAN_AREA_CODE = "00000_Japan"

TOKYO_MA_AREA_CODES = [
    "11000_Saitama-ken",
    "12000_Chiba-ken",
    "13000_Tokyo-to",
    "14000_Kanagawa-ken",
]

OSAKA_KANSAI_AREA_CODES = [
    "26000_Kyoto-fu",
    "27000_Osaka-fu",
    "28000_Hyogo-ken",
    "29000_Nara-ken",
]

# METI manufacturing data: 2-digit prefecture codes
JAPAN_PREF_CODE = "00"
TOKYO_MA_PREF_CODES = ["11", "12", "13", "14"]
OSAKA_KANSAI_PREF_CODES = ["26", "27", "28", "29"]

# Prefecture names in Japanese (for METI data)
TOKYO_MA_PREF_NAMES_JP = ["埼玉県", "千葉県", "東京都", "神奈川県"]
OSAKA_KANSAI_PREF_NAMES_JP = ["京都府", "大阪府", "兵庫県", "奈良県"]

REGION_LABELS = {
    "tokyo_ma": "Tokyo Metropolitan Area",
    "osaka_kansai": "Osaka/Kansai Metropolitan Area",
    "japan": "Japan",
}

# ── Industry codes – Economic Census (b1_006_1e) ─────────────────────────────

# Division codes (col 3) and group labels (col 4) in b1_006_1e
CENSUS_TOTAL_CODE = "AR"  # All industries total

BROAD_HIGHTECH_DIV_CODES = {
    "manufacturing": "E",
    "ict_services": "G",
    "scientific_professional": "L",
}

# HTP census-service component policy.  Economic Census rows contain parent
# categories and child rows; use parent service rows only to avoid double
# counting, and source selected high-tech manufacturing from METI instead.
USE_BROAD_ICT_PARENT = True
BROAD_HIGHTECH_CENSUS_INDUSTRIES = {
    "ict_services": ["G"],
    "scientific_professional_services": ["L"],
}
CENSUS_PARENT_HIERARCHY = 2

# Sub-divisions for ICT (for optional breakdown)
ICT_SUB_CODES = {"ict_comm": "G1", "ict_software_internet": "G2"}

# ── Industry codes – METI manufacturing 2-digit ───────────────────────────────

MFG_2DIGIT_HIGHTECH = {
    "chemicals_advanced_materials": ["16"],
    "general_purpose_machinery": ["25"],
    "production_machinery_robotics": ["26"],
    "business_machinery_medical_precision": ["27"],
    "electronic_parts_devices_circuits": ["28"],
    "electrical_machinery": ["29"],
    "information_communication_equipment": ["30"],
}

MFG_2DIGIT_ENGLISH = {
    "16": "Chemicals & Adv. Materials",
    "25": "General-Purpose Machinery",
    "26": "Production Machinery & Robotics",
    "27": "Business Machinery & Medical Devices",
    "28": "Electronic Parts & Circuits",
    "29": "Electrical Machinery",
    "30": "ICT Equipment",
    "00": "All Manufacturing",
}

# ── Industry codes – METI manufacturing 4-digit ───────────────────────────────

MFG_4DIGIT_HIGHTECH = {
    "pharmaceuticals": ["1651", "1652", "1655"],
    "semiconductor_equipment_robotics": ["2671", "2694"],
    "measurement_medical_precision": ["2738", "2739", "2741", "2743", "2753"],
    "electronic_components_semiconductors": ["2811", "2813", "2831", "2841", "2842", "2899"],
    "electrical_industrial_equipment": [
        "2911", "2912", "2921", "2929", "2962",
        "2969", "2971", "2973", "2999",
    ],
    "communication_computer_equipment": ["3011", "3013", "3019", "3023", "3031"],
}

# Flat set of all high-tech 4-digit codes (for filtering)
ALL_4DIGIT_HIGHTECH_CODES = {
    code
    for codes in MFG_4DIGIT_HIGHTECH.values()
    for code in codes
}

# Flat set of all high-tech 2-digit codes
ALL_2DIGIT_HIGHTECH_CODES = {
    code
    for codes in MFG_2DIGIT_HIGHTECH.values()
    for code in codes
}

# ── Data quality ──────────────────────────────────────────────────────────────

MISSING_MARKERS = {"X", "...", "-", "***", "x", "…"}

# ── World Bank ────────────────────────────────────────────────────────────────

WB_YEARS = list(range(2010, 2025))
WB_COUNTRY_CODE = "JPN"

# ── JPO patent data ──────────────────────────────────────────────────────────

JPO_PATENT_MANUAL_FILENAME = "jpo_patent_applications_prefecture_manual.csv"
JPO_LEGACY_PATENT_MANUAL_FILENAME = "jpo_patents_prefecture_manual.csv"
JPO_INVENTORS_MANUAL_FILENAME = "jpo_inventors_prefecture_manual.csv"

# ── Technology exchange selected high-tech sectors ───────────────────────────

SELECTED_HIGHTECH_TECHNOLOGY_ROWS = {
    "医薬品製造業": ("Pharmaceuticals", "pharma_life_sciences"),
    "化学工業": ("Chemicals", "pharma_life_sciences"),
    "はん用機械器具製造業": ("General-purpose machinery", "machinery_automation"),
    "生産用機械器具製造業": ("Production machinery", "machinery_automation"),
    "業務用機械器具製造業": (
        "Business-oriented machinery / medical devices",
        "machinery_automation",
    ),
    "電子部品・デバイス・電子回路製造業": (
        "Electronic parts, devices and circuits",
        "electronics_components",
    ),
    "電気機械器具製造業": ("Electrical machinery", "electronics_components"),
    "情報通信機械器具製造業": (
        "Information and communication equipment",
        "electronics_components",
    ),
    "情報サービス業": ("Information services", "ict_services"),
    "学術・開発研究機関": (
        "Scientific research and development institutes",
        "rd_technical_services",
    ),
    "技術サービス業(他に分類されないもの)": (
        "Technical services, n.e.c.",
        "rd_technical_services",
    ),
}

# ── R&D / technology exchange (eStat column positions, 0-indexed) ─────────────

ESTAT_COL_HIERARCHY = 11
ESTAT_COL_INDUSTRY = 12
ESTAT_COL_MGMT_NO = 13
ESTAT_DATA_START = 14  # first numeric data column

MIC_TABLE_SPECS = {
    "2024ka101.xlsx": {
        "sheet": "A101",
        "header_row": 14,
        "unit_row": 13,
        "data_start_row": 15,
    },
    "2024ka102.xlsx": {
        "sheet": "A102",
        "header_row": 11,
        "unit_row": 10,
        "data_start_row": 12,
    },
    "2024ka201.xlsx": {
        "sheet": "A201",
        "header_row": 12,
        "unit_row": 11,
        "data_start_row": 13,
    },
    "2024ka210.xlsx": {
        "sheet": "A210",
        "header_row": 11,
        "unit_row": 10,
        "data_start_row": 12,
    },
    "2024ka211.xlsx": {
        "sheet": "A211",
        "header_row": 11,
        "unit_row": 10,
        "data_start_row": 12,
    },
    "2024ka212.xlsx": {
        "sheet": "A212",
        "header_row": 11,
        "unit_row": 10,
        "data_start_row": 12,
    },
}
