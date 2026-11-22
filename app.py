"""JewelCast AI - application shell.

This milestone-level page reads computed artefacts only. The full six-page
forecasting and inventory dashboard is planned for milestone M7.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.utils.config import load_config
from src.utils.paths import resolve_path

st.set_page_config(
    page_title="JewelCast AI",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    :root {
        --ivory: #faf7f0;
        --navy: #12213a;
        --gold: #b08d57;
        --muted: #6b7280;
        --card: #ffffff;
    }
    .stApp { background: var(--ivory); color: var(--navy); }
    h1, h2, h3 { color: var(--navy); letter-spacing: -0.02em; }
    [data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid #e7dfcf;
        border-radius: 14px;
        padding: 16px 18px;
        box-shadow: 0 1px 2px rgba(18, 33, 58, 0.04);
    }
    [data-testid="stMetricLabel"] { color: var(--muted); }
    [data-testid="stMetricValue"] { color: var(--navy); }
    .jc-kicker {
        color: var(--gold);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.14em;
        text-transform: uppercase;
    }
    .jc-subtitle { color: var(--muted); font-size: 1.02rem; margin-top: -0.4rem; }
    .jc-panel {
        background: var(--card);
        border: 1px solid #e7dfcf;
        border-radius: 14px;
        padding: 18px 20px;
        margin-bottom: 14px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

config = load_config()
processed = resolve_path(config["paths"]["data_processed"])
MODE_LABELS = {
    "synthetic": "Synthetic jewellery (INR, simulated)",
    "uci": "UCI Online Retail II (GBP, UK)",
    "m5": "M5 Walmart benchmark (USD)",
}
CURRENCY = {"synthetic": "₹", "uci": "£", "m5": "$"}

available = [
    mode
    for mode in ("synthetic", "uci", "m5")
    if (processed / mode / "dataset_summary.json").exists()
]

st.markdown('<div class="jc-kicker">Retail intelligence</div>', unsafe_allow_html=True)
st.title("JewelCast AI")
st.markdown(
    '<div class="jc-subtitle">SKU-level demand forecasting and inventory optimization for fashion jewellery.</div>',
    unsafe_allow_html=True,
)

if not available:
    st.warning(
        "No processed datasets found. Run:\n\n"
        "```powershell\npython scripts/build_datasets.py --all\n```"
    )
    st.stop()

with st.sidebar:
    st.header("Data mode")
    mode = st.selectbox("Processed dataset", available, format_func=lambda value: MODE_LABELS[value])
    st.caption("Modes are kept separate. Real and synthetic data are never mixed.")
    st.divider()
    st.markdown("**Milestone status**")
    st.write("M0–M1 complete: ingestion, validation, quality, metadata, tests.")
    st.write("M2–M7 planned: EDA, features, models, inventory, explainability, dashboard.")

summary = json.loads((processed / mode / "dataset_summary.json").read_text(encoding="utf-8"))
quality_path = processed / mode / "quality_report.json"
quality = json.loads(quality_path.read_text(encoding="utf-8")) if quality_path.exists() else {}
provenance_path = processed / mode / "provenance.json"
provenance = json.loads(provenance_path.read_text(encoding="utf-8")) if provenance_path.exists() else {}

currency = CURRENCY.get(mode, "")
units = summary.get("units_sold") or 0
revenue = summary.get("revenue") or 0
date_min = str(summary.get("date_min", ""))[:10]
date_max = str(summary.get("date_max", ""))[:10]

st.markdown('<div class="jc-panel">', unsafe_allow_html=True)
columns = st.columns(5)
columns[0].metric("SKU-day rows", f"{int(summary.get('rows', 0)):,}")
columns[1].metric("Active SKUs", f"{int(summary.get('skus', 0)):,}")
columns[2].metric("Categories", f"{int(summary.get('categories', 0)):,}")
columns[3].metric("Units sold", f"{units:,.0f}")
columns[4].metric("Revenue", f"{currency}{revenue:,.0f}")
st.caption(f"Coverage: {date_min} to {date_max} · Inventory available: {summary.get('inventory_available')}")
st.markdown("</div>", unsafe_allow_html=True)

tab_quality, tab_provenance, tab_limitations = st.tabs(
    ["Data quality", "Provenance", "Limitations"]
)

with tab_quality:
    left, right = st.columns([1, 1])
    with left:
        st.subheader("Integrity checks")
        checks = pd.DataFrame(
            [
                ["Duplicate SKU-date rows", quality.get("duplicate_sku_date_rows", 0)],
                ["Negative units rows", quality.get("negative_units_rows", 0)],
                ["Negative revenue rows", quality.get("negative_revenue_rows", 0)],
                ["Null price rows", quality.get("null_price_rows", 0)],
                ["Null inventory rows", quality.get("null_inventory_rows", 0)],
                ["Date-imputed rows", quality.get("date_imputed_rows", 0)],
                ["Stockout-flagged rows", quality.get("stockout_flagged_rows", 0)],
            ],
            columns=["Check", "Rows"],
        )
        st.dataframe(checks, hide_index=True, use_container_width=True)
        st.caption(
            f"Zero-sales share: {quality.get('totals', {}).get('zero_sales_share', 0):.1%} · "
            f"Outliers above extreme bound: {quality.get('outliers', {}).get('rows_above_extreme_bound', 0):,}"
        )
    with right:
        st.subheader("Category mix")
        categories = quality.get("category_summary", [])
        if categories:
            frame = pd.DataFrame(categories)
            st.dataframe(frame, hide_index=True, use_container_width=True)
        else:
            st.info("No category summary available.")

with tab_provenance:
    st.subheader("Source and treatment")
    st.write(provenance.get("provenance", "Provenance not recorded."))
    st.json(provenance, expanded=False)

with tab_limitations:
    st.subheader("Known limitations")
    for item in provenance.get("limitations", []):
        st.write(f"- {item}")
    st.info(
        "Forecasting, inventory optimization, SHAP explanations and the full dashboard "
        "are planned for later milestones. No model accuracy is claimed yet."
    )
