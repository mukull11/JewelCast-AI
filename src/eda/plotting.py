"""Static (Matplotlib) and interactive (Plotly) EDA charts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

GOLD = "#b08d57"
NAVY = "#12213a"
TEAL = "#2f7f8f"
RED = "#b23a48"
GREEN = "#3f7d4e"
GRID = "#e7dfcf"


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def save_matplotlib(fig: plt.Figure, path: Path) -> Path:
    _ensure_parent(path)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def save_plotly(fig: go.Figure, path: Path) -> Path:
    _ensure_parent(path)
    fig.write_html(path, include_plotlyjs="cdn", full_html=True)
    return path


def _style_axes(ax: plt.Axes) -> None:
    ax.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def daily_trend_chart(
    daily: pd.DataFrame,
    out_dir: Path,
    currency: str,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if daily.empty:
        return {}
    frame = daily.sort_values("period_start")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax1 = plt.subplots(figsize=(11, 4.6))
        ax1.plot(frame["period_start"], frame["units_sold"], color=GOLD, linewidth=1.4, label="Units sold")
        ax1.set_ylabel("Units sold")
        ax1.set_xlabel("Date")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2 = ax1.twinx()
        ax2.plot(frame["period_start"], frame["revenue"], color=NAVY, linewidth=1.2, alpha=0.8, label="Revenue")
        ax2.set_ylabel(f"Revenue ({currency})")
        _style_axes(ax1)
        ax1.set_title("Daily demand and revenue")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
        fig.autofmt_xdate()
        paths["png"] = save_matplotlib(fig, out_dir / "daily_demand.png")
    if make_html:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(
            go.Scatter(x=frame["period_start"], y=frame["units_sold"], name="Units sold", line=dict(color=GOLD, width=1.6)),
            secondary_y=False,
        )
        fig.add_trace(
            go.Scatter(x=frame["period_start"], y=frame["revenue"], name="Revenue", line=dict(color=NAVY, width=1.2), opacity=0.75),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Units sold", secondary_y=False)
        fig.update_yaxes(title_text=f"Revenue ({currency})", secondary_y=True)
        fig.update_layout(title="Daily demand and revenue", hovermode="x unified", template="plotly_white", height=420)
        paths["html"] = save_plotly(fig, out_dir / "daily_demand.html")
    return paths


def weekly_trend_chart(
    weekly: pd.DataFrame,
    out_dir: Path,
    currency: str,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if weekly.empty:
        return {}
    frame = weekly.sort_values("period_start")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(11, 4.2))
        ax.bar(frame["period_start"], frame["units_sold"], color=GOLD, width=6.0, label="Units sold")
        ax.set_ylabel("Units sold")
        ax.set_xlabel("Week starting (Monday)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        _style_axes(ax)
        ax.set_title("Weekly units sold")
        fig.autofmt_xdate()
        paths["png"] = save_matplotlib(fig, out_dir / "weekly_demand.png")
    if make_html:
        fig = go.Figure(
            go.Bar(x=frame["period_start"], y=frame["units_sold"], name="Units sold", marker_color=GOLD)
        )
        fig.update_layout(title="Weekly units sold", xaxis_title="Week starting", yaxis_title="Units sold", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "weekly_demand.html")
    return paths


def monthly_trend_chart(
    monthly: pd.DataFrame,
    out_dir: Path,
    currency: str,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if monthly.empty:
        return {}
    frame = monthly.sort_values("period_start")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax1 = plt.subplots(figsize=(11, 4.4))
        ax1.bar(frame["period_start"], frame["units_sold"], color=GOLD, width=20.0, label="Units sold")
        ax1.set_ylabel("Units sold")
        ax1.set_xlabel("Month")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax2 = ax1.twinx()
        ax2.plot(frame["period_start"], frame["revenue"], color=NAVY, linewidth=1.8, marker="o", markersize=3, label="Revenue")
        ax2.set_ylabel(f"Revenue ({currency})")
        _style_axes(ax1)
        ax1.set_title("Monthly demand and revenue")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left", frameon=False)
        fig.autofmt_xdate()
        paths["png"] = save_matplotlib(fig, out_dir / "monthly_demand.png")
    if make_html:
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        fig.add_trace(go.Bar(x=frame["period_start"], y=frame["units_sold"], name="Units sold", marker_color=GOLD), secondary_y=False)
        fig.add_trace(
            go.Scatter(x=frame["period_start"], y=frame["revenue"], name="Revenue", line=dict(color=NAVY, width=1.8)),
            secondary_y=True,
        )
        fig.update_yaxes(title_text="Units sold", secondary_y=False)
        fig.update_yaxes(title_text=f"Revenue ({currency})", secondary_y=True)
        fig.update_layout(title="Monthly demand and revenue", hovermode="x unified", template="plotly_white", height=420)
        paths["html"] = save_plotly(fig, out_dir / "monthly_demand.html")
    return paths


def weekday_chart(
    weekday: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if weekday.empty:
        return {}
    frame = weekday.sort_values("weekday_number")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        colors = [GOLD if value < 1 else TEAL for value in frame["units_index"].fillna(1)]
        ax.bar(frame["weekday"], frame["avg_daily_units"], color=colors)
        ax.set_ylabel("Average units per day")
        ax.set_xlabel("Weekday")
        ax.axhline(frame["avg_daily_units"].mean(), color=NAVY, linestyle="--", linewidth=1, label="Overall average")
        _style_axes(ax)
        ax.set_title("Weekday demand profile")
        ax.legend(frameon=False)
        plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
        paths["png"] = save_matplotlib(fig, out_dir / "weekday_profile.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["weekday"], y=frame["avg_daily_units"], marker_color=GOLD))
        fig.update_layout(title="Weekday demand profile", xaxis_title="Weekday", yaxis_title="Average units per day", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "weekday_profile.html")
    return paths


def month_chart(
    months: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if months.empty:
        return {}
    frame = months.sort_values("month")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(10, 4.2))
        colors = [TEAL if value >= 1 else GOLD for value in frame["units_index"].fillna(1)]
        ax.bar(frame["month_name"], frame["avg_daily_units"], color=colors)
        ax.set_ylabel("Average units per day")
        ax.set_xlabel("Month")
        ax.axhline(frame["avg_daily_units"].mean(), color=NAVY, linestyle="--", linewidth=1)
        _style_axes(ax)
        ax.set_title("Month-of-year demand profile")
        paths["png"] = save_matplotlib(fig, out_dir / "month_profile.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["month_name"], y=frame["avg_daily_units"], marker_color=GOLD))
        fig.update_layout(title="Month-of-year demand profile", xaxis_title="Month", yaxis_title="Average units per day", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "month_profile.html")
    return paths


def category_revenue_chart(
    categories: pd.DataFrame,
    out_dir: Path,
    top_n: int = 20,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if categories.empty:
        return {}
    frame = categories.sort_values("revenue", ascending=False).head(top_n)
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, max(3.5, 0.38 * len(frame) + 1.4)))
        ax.barh(frame["category"][::-1], frame["revenue"][::-1], color=GOLD)
        ax.set_xlabel("Revenue")
        _style_axes(ax)
        ax.set_title("Category revenue contribution")
        paths["png"] = save_matplotlib(fig, out_dir / "category_revenue.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["revenue"], y=frame["category"], orientation="h", marker_color=GOLD))
        fig.update_layout(title="Category revenue contribution", xaxis_title="Revenue", yaxis_title="Category", template="plotly_white", height=max(360, 28 * len(frame)))
        paths["html"] = save_plotly(fig, out_dir / "category_revenue.html")
    return paths


def category_month_heatmap(
    category_monthly: pd.DataFrame,
    out_dir: Path,
    top_n: int = 12,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if category_monthly.empty:
        return {}
    top_categories = (
        category_monthly.groupby("category", observed=True)["units_sold"].sum().nlargest(top_n).index
    )
    subset = category_monthly.loc[category_monthly["category"].isin(top_categories)].copy()
    subset["month_label"] = subset["month_start"].dt.strftime("%Y-%m")
    pivot = subset.pivot_table(index="category", columns="month_label", values="units_sold", aggfunc="sum", fill_value=0)
    if pivot.empty:
        return {}
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(max(10, 0.35 * pivot.shape[1] + 3), max(3.5, 0.45 * pivot.shape[0] + 1.5)))
        image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlOrBr")
        ax.set_xticks(range(pivot.shape[1]))
        ax.set_xticklabels(pivot.columns, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(pivot.shape[0]))
        ax.set_yticklabels(pivot.index, fontsize=8)
        ax.set_title("Category monthly units heatmap")
        fig.colorbar(image, ax=ax, label="Units sold")
        paths["png"] = save_matplotlib(fig, out_dir / "category_month_heatmap.png")
    if make_html:
        fig = go.Figure(
            go.Heatmap(
                z=pivot.to_numpy(),
                x=list(pivot.columns),
                y=list(pivot.index),
                colorscale="YlOrBr",
                colorbar=dict(title="Units"),
            )
        )
        fig.update_layout(title="Category monthly units heatmap", template="plotly_white", height=max(400, 26 * pivot.shape[0]))
        paths["html"] = save_plotly(fig, out_dir / "category_month_heatmap.html")
    return paths


def abc_chart(
    sku_metrics: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if sku_metrics.empty or "abc_class" not in sku_metrics.columns:
        return {}
    frame = sku_metrics.sort_values("revenue", ascending=False).reset_index(drop=True)
    frame["rank"] = np.arange(1, len(frame) + 1)
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(10, 4.6))
        colors = frame["abc_class"].map({"A": GOLD, "B": TEAL, "C": NAVY}).fillna(NAVY)
        ax.bar(frame["rank"], frame["revenue_share"], color=colors, width=1.0, label="Revenue share")
        ax.plot(frame["rank"], frame["cumulative_revenue_share"], color=RED, linewidth=1.6, label="Cumulative share")
        ax.axhline(0.8, color=NAVY, linestyle="--", linewidth=1, alpha=0.6)
        ax.axhline(0.95, color=NAVY, linestyle=":", linewidth=1, alpha=0.6)
        ax.set_xlabel("SKU rank by revenue")
        ax.set_ylabel("Share of revenue")
        _style_axes(ax)
        ax.set_title("ABC revenue concentration (Pareto curve)")
        ax.legend(frameon=False)
        paths["png"] = save_matplotlib(fig, out_dir / "abc_pareto.png")
    if make_html:
        fig = go.Figure()
        fig.add_trace(go.Bar(x=frame["rank"], y=frame["revenue_share"], name="Revenue share", marker_color=GOLD))
        fig.add_trace(go.Scatter(x=frame["rank"], y=frame["cumulative_revenue_share"], name="Cumulative share", line=dict(color=RED, width=2)))
        fig.add_hline(y=0.8, line_dash="dash", line_color=NAVY)
        fig.add_hline(y=0.95, line_dash="dot", line_color=NAVY)
        fig.update_layout(title="ABC revenue concentration (Pareto curve)", xaxis_title="SKU rank by revenue", yaxis_title="Share of revenue", template="plotly_white", height=420)
        paths["html"] = save_plotly(fig, out_dir / "abc_pareto.html")
    return paths


def sku_distribution_chart(
    sku_metrics: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if sku_metrics.empty:
        return {}
    units = sku_metrics["units_sold"].astype(float)
    positive = units[units > 0]
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        if not positive.empty:
            bins = np.logspace(np.log10(max(positive.min(), 0.5)), np.log10(max(positive.max(), 1.0)), 30)
            ax.hist(positive, bins=bins, color=GOLD, edgecolor="white")
            ax.set_xscale("log")
        ax.set_xlabel("Total units sold per SKU (log scale)")
        ax.set_ylabel("Number of SKUs")
        _style_axes(ax)
        ax.set_title("SKU demand distribution (positive-demand SKUs)")
        paths["png"] = save_matplotlib(fig, out_dir / "sku_demand_distribution.png")
    if make_html:
        fig = go.Figure()
        if not positive.empty:
            fig.add_trace(go.Histogram(x=positive, nbinsx=30, marker_color=GOLD))
            fig.update_xaxes(type="log", title_text="Total units sold per SKU (log scale)")
        fig.update_layout(title="SKU demand distribution (positive-demand SKUs)", yaxis_title="Number of SKUs", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "sku_demand_distribution.html")
    return paths


def intermittent_chart(
    intermittent: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if intermittent.empty:
        return {}
    frame = intermittent.copy()
    order = ["smooth", "erratic", "intermittent", "lumpy", "no_demand"]
    frame["intermittent_class"] = pd.Categorical(frame["intermittent_class"], categories=order, ordered=True)
    frame = frame.sort_values("intermittent_class")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, 4.0))
        ax.bar(frame["intermittent_class"].astype(str), frame["skus"], color=[GOLD, TEAL, RED, NAVY, "#9aa4b2"][: len(frame)])
        ax.set_ylabel("Number of SKUs")
        ax.set_xlabel("Intermittent demand class")
        _style_axes(ax)
        ax.set_title("Intermittent demand classification")
        paths["png"] = save_matplotlib(fig, out_dir / "intermittent_classes.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["intermittent_class"].astype(str), y=frame["skus"], marker_color=GOLD))
        fig.update_layout(title="Intermittent demand classification", xaxis_title="Class", yaxis_title="Number of SKUs", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "intermittent_classes.html")
    return paths


def price_distribution_chart(
    daily: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    price = pd.to_numeric(daily.get("selling_price"), errors="coerce").dropna()
    price = price[price > 0]
    if price.empty:
        return {}
    upper = price.quantile(0.99)
    clipped = price[price <= upper] if upper > 0 else price
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.hist(clipped, bins=40, color=GOLD, edgecolor="white")
        ax.set_xlabel("Selling price (clipped at 99th percentile)")
        ax.set_ylabel("SKU-day rows")
        _style_axes(ax)
        ax.set_title("Selling price distribution")
        paths["png"] = save_matplotlib(fig, out_dir / "price_distribution.png")
    if make_html:
        fig = go.Figure(go.Histogram(x=clipped, nbinsx=40, marker_color=GOLD))
        fig.update_layout(title="Selling price distribution", xaxis_title="Selling price", yaxis_title="SKU-day rows", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "price_distribution.html")
    return paths


def price_demand_chart(
    price_bins: pd.DataFrame,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if price_bins.empty:
        return {}
    frame = price_bins.sort_values("avg_price")
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(9, 4.2))
        ax.plot(frame["avg_price"], frame["avg_units_per_row"], color=GOLD, marker="o", linewidth=1.8)
        ax.set_xlabel("Average selling price in bin")
        ax.set_ylabel("Average units per SKU-day")
        _style_axes(ax)
        ax.set_title("Pooled price-demand association (not causal)")
        paths["png"] = save_matplotlib(fig, out_dir / "price_demand.png")
    if make_html:
        fig = go.Figure(
            go.Scatter(
                x=frame["avg_price"],
                y=frame["avg_units_per_row"],
                mode="lines+markers",
                line=dict(color=GOLD, width=2),
                marker=dict(size=7),
            )
        )
        fig.update_layout(title="Pooled price-demand association (not causal)", xaxis_title="Average price in bin", yaxis_title="Average units per SKU-day", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "price_demand.html")
    return paths


def promotion_chart(
    promotion: dict[str, Any],
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if not promotion.get("available") or "segments" not in promotion:
        return {}
    frame = promotion["segments"].copy()
    if frame.empty:
        return {}
    paths: dict[str, Path] = {}
    if make_png:
        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.bar(frame["segment"], frame["avg_daily_units"], color=[GOLD, TEAL][: len(frame)])
        ax.set_ylabel("Average units per SKU-day")
        ax.set_xlabel("Segment")
        _style_axes(ax)
        ax.set_title("Promotion versus non-promotion demand")
        paths["png"] = save_matplotlib(fig, out_dir / "promotion_demand.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["segment"], y=frame["avg_daily_units"], marker_color=GOLD))
        fig.update_layout(title="Promotion versus non-promotion demand", xaxis_title="Segment", yaxis_title="Average units per SKU-day", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "promotion_demand.html")
    return paths


def festival_chart(
    festival: dict[str, Any],
    mode: str,
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    key = {"synthetic": "festival_segments", "uci": "holiday_segments", "m5": "event_segments"}.get(mode)
    if not key or key not in festival:
        return {}
    frame = festival[key]
    if frame is None or frame.empty:
        return {}
    paths: dict[str, Path] = {}
    title = {
        "synthetic": "Festival versus non-festival demand",
        "uci": "UK bank holiday versus non-holiday demand",
        "m5": "M5 event versus non-event demand",
    }.get(mode, "Calendar-event demand")
    if make_png:
        fig, ax = plt.subplots(figsize=(8, 4.0))
        ax.bar(frame["segment"], frame["avg_daily_units"], color=[GOLD, TEAL][: len(frame)])
        ax.set_ylabel("Average units per day")
        ax.set_xlabel("Segment")
        _style_axes(ax)
        ax.set_title(title)
        paths["png"] = save_matplotlib(fig, out_dir / "calendar_event_demand.png")
    if make_html:
        fig = go.Figure(go.Bar(x=frame["segment"], y=frame["avg_daily_units"], marker_color=GOLD))
        fig.update_layout(title=title, xaxis_title="Segment", yaxis_title="Average units per day", template="plotly_white", height=400)
        paths["html"] = save_plotly(fig, out_dir / "calendar_event_demand.html")
    return paths


def inventory_charts(
    inventory: dict[str, Any],
    out_dir: Path,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, Path]:
    if not inventory.get("available"):
        return {}
    paths: dict[str, Path] = {}
    by_sku = inventory.get("by_sku", pd.DataFrame())
    if not by_sku.empty:
        if make_png:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.scatter(by_sku["avg_stock"], by_sku["inventory_turnover"], color=GOLD, alpha=0.75)
            ax.set_xlabel("Average on-hand stock")
            ax.set_ylabel("Inventory turnover (period units / mean stock)")
            _style_axes(ax)
            ax.set_title("Stock level versus inventory turnover")
            paths["png"] = save_matplotlib(fig, out_dir / "inventory_turnover.png")
        if make_html:
            fig = go.Figure(
                go.Scatter(
                    x=by_sku["avg_stock"],
                    y=by_sku["inventory_turnover"],
                    mode="markers",
                    marker=dict(color=GOLD, size=7, opacity=0.75),
                )
            )
            fig.update_layout(title="Stock level versus inventory turnover", xaxis_title="Average on-hand stock", yaxis_title="Inventory turnover", template="plotly_white", height=400)
            paths["html"] = save_plotly(fig, out_dir / "inventory_turnover.html")

        if make_png:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.hist(by_sku["days_of_cover"].replace([np.inf, -np.inf], np.nan).dropna(), bins=30, color=TEAL, edgecolor="white")
            ax.set_xlabel("Days of inventory cover")
            ax.set_ylabel("Number of SKUs")
            _style_axes(ax)
            ax.set_title("Days of inventory cover by SKU")
            paths["png"] = save_matplotlib(fig, out_dir / "inventory_days_cover.png")
        if make_html:
            cover = by_sku["days_of_cover"].replace([np.inf, -np.inf], np.nan).dropna()
            fig = go.Figure(go.Histogram(x=cover, nbinsx=30, marker_color=TEAL))
            fig.update_layout(title="Days of inventory cover by SKU", xaxis_title="Days of cover", yaxis_title="Number of SKUs", template="plotly_white", height=400)
            paths["html"] = save_plotly(fig, out_dir / "inventory_days_cover.html")

        if make_png:
            fig, ax = plt.subplots(figsize=(9, 4.2))
            ax.hist(by_sku["stockout_frequency"], bins=25, color=RED, edgecolor="white")
            ax.set_xlabel("Stockout frequency (stockout days / observed days)")
            ax.set_ylabel("Number of SKUs")
            _style_axes(ax)
            ax.set_title("Stockout frequency by SKU")
            paths["png"] = save_matplotlib(fig, out_dir / "inventory_stockout_frequency.png")
        if make_html:
            fig = go.Figure(go.Histogram(x=by_sku["stockout_frequency"], nbinsx=25, marker_color=RED))
            fig.update_layout(title="Stockout frequency by SKU", xaxis_title="Stockout frequency", yaxis_title="Number of SKUs", template="plotly_white", height=400)
            paths["html"] = save_plotly(fig, out_dir / "inventory_stockout_frequency.html")
    return paths


def generate_charts(
    context: dict[str, Any],
    out_dir: Path,
    mode: str,
    currency: str,
    top_n: int = 20,
    make_png: bool = True,
    make_html: bool = True,
) -> dict[str, dict[str, Path]]:
    """Generate the standard EDA chart set for one dataset mode."""
    out_dir.mkdir(parents=True, exist_ok=True)
    charts: dict[str, dict[str, Path]] = {}
    trends = context["trends"]
    charts["daily"] = daily_trend_chart(trends["daily"], out_dir, currency, make_png, make_html)
    charts["weekly"] = weekly_trend_chart(trends["weekly"], out_dir, currency, make_png, make_html)
    charts["monthly"] = monthly_trend_chart(trends["monthly"], out_dir, currency, make_png, make_html)
    charts["weekday"] = weekday_chart(context["weekday"]["by_weekday"], out_dir, make_png, make_html)
    charts["month"] = month_chart(context["month_profile"], out_dir, make_png, make_html)
    charts["category_revenue"] = category_revenue_chart(context["category_metrics"], out_dir, top_n, make_png, make_html)
    charts["category_month"] = category_month_heatmap(context["category_monthly"], out_dir, top_n, make_png, make_html)
    charts["abc"] = abc_chart(context["sku_metrics"], out_dir, make_png, make_html)
    charts["sku_distribution"] = sku_distribution_chart(context["sku_metrics"], out_dir, make_png, make_html)
    charts["intermittent"] = intermittent_chart(context["intermittent_summary"], out_dir, make_png, make_html)
    charts["price_distribution"] = price_distribution_chart(context["daily"], out_dir, make_png, make_html)
    charts["price_demand"] = price_demand_chart(context["price_demand_bins"], out_dir, make_png, make_html)
    charts["promotion"] = promotion_chart(context["promotion"], out_dir, make_png, make_html)
    charts["calendar_event"] = festival_chart(context["festival"], mode, out_dir, make_png, make_html)
    charts["inventory"] = inventory_charts(context["inventory"], out_dir, make_png, make_html)
    return {key: value for key, value in charts.items() if value}
