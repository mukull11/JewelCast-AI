"""Tests for EDA summary calculations, ABC, intermittent demand and missing data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.eda.metrics import (
    abc_classify,
    abc_summary,
    category_metrics,
    classify_intermittent,
    compute_sku_metrics,
    demand_concentration,
    demand_trends,
    festival_analysis,
    inventory_analysis,
    intermittent_summary,
    missing_date_summary,
    overview_metrics,
    price_summary,
    promotion_analysis,
    sku_date_coverage,
)


def _row(date, sku, units, price, category="Earrings", stock=10.0, stockout=0, imputed=0):
    return {
        "date": pd.Timestamp(date),
        "sku": sku,
        "product_name": f"Product {sku}",
        "category": category,
        "units_sold": units,
        "selling_price": price,
        "discount_percentage": 0.0,
        "revenue": units * price,
        "festival_flag": 0,
        "festival_name": None,
        "promotion_flag": 0,
        "available_stock": stock,
        "stockout_flag": stockout,
        "event_name": None,
        "date_imputed": imputed,
        "data_source": "synthetic",
    }


@pytest.fixture()
def eda_daily() -> pd.DataFrame:
    rows = [
        _row("2024-01-01", "A", 2, 10.0, stock=10.0),
        _row("2024-01-02", "A", 0, 10.0, stock=8.0),
        _row("2024-01-03", "A", 1, 12.0, stock=7.0),
        _row("2024-01-04", "A", 3, 12.0, stock=4.0),
        _row("2024-01-01", "B", 0, 20.0, category="Rings", stock=5.0),
        _row("2024-01-02", "B", 0, 20.0, category="Rings", stock=5.0),
        _row("2024-01-03", "B", 0, 20.0, category="Rings", stock=5.0),
        _row("2024-01-04", "B", 5, 20.0, category="Rings", stock=0.0, stockout=1),
        _row("2024-01-01", "C", 1, 10.0, category="Bangles", stock=2.0),
        _row("2024-01-03", "C", 0, 10.0, category="Bangles", stock=1.0, imputed=1),
    ]
    frame = pd.DataFrame(rows)
    frame.loc[(frame["sku"] == "A") & (frame["date"] == pd.Timestamp("2024-01-03")), "festival_flag"] = 1
    frame.loc[(frame["sku"] == "A") & (frame["date"] == pd.Timestamp("2024-01-03")), "festival_name"] = "FestivalX"
    frame.loc[(frame["sku"] == "B") & (frame["date"] == pd.Timestamp("2024-01-03")), "festival_flag"] = 1
    frame.loc[(frame["sku"] == "B") & (frame["date"] == pd.Timestamp("2024-01-03")), "festival_name"] = "FestivalX"
    frame.loc[(frame["sku"].isin(["A", "B"])) & (frame["date"] == pd.Timestamp("2024-01-04")), "promotion_flag"] = 1
    frame.loc[(frame["sku"] == "A") & (frame["date"] == pd.Timestamp("2024-01-04")), "discount_percentage"] = 20.0
    return frame


def test_overview_metrics(eda_daily):
    overview = overview_metrics(eda_daily)
    assert overview["rows"] == 10
    assert overview["skus"] == 3
    assert overview["active_skus"] == 3
    assert overview["inactive_skus"] == 0
    assert overview["categories"] == 3
    assert overview["units_sold"] == 12
    assert overview["revenue"] == pytest.approx(178.0)
    assert overview["zero_sales_rows"] == 5
    assert overview["zero_sales_share"] == pytest.approx(0.5)
    assert overview["coverage"]["observed_dates"] == 4
    assert overview["coverage"]["imputed_rows"] == 1
    assert overview["coverage"]["missing_sku_days"] == 1
    assert overview["inventory_available"] is True


def test_demand_trends_sum_to_total(eda_daily):
    trends = demand_trends(eda_daily)
    assert len(trends["daily"]) == 4
    assert trends["daily"]["units_sold"].sum() == 12
    assert trends["weekly"]["units_sold"].sum() == 12
    assert trends["monthly"]["units_sold"].sum() == 12
    assert trends["daily"].loc[trends["daily"]["period_start"] == pd.Timestamp("2024-01-04"), "units_sold"].iloc[0] == 8
    assert trends["daily"]["active_skus"].tolist() == [2, 0, 1, 2]


def test_abc_classification_boundaries():
    frame = pd.DataFrame(
        {
            "sku": ["S1", "S2", "S3", "S4", "S5"],
            "revenue": [50.0, 30.0, 10.0, 5.0, 5.0],
        }
    )
    classified = abc_classify(frame)
    assert classified["abc_class"].tolist() == ["A", "A", "B", "B", "C"]
    assert classified["cumulative_revenue_share"].iloc[-1] == pytest.approx(1.0)
    assert classified["revenue_share"].sum() == pytest.approx(1.0)


def test_abc_classification_zero_revenue():
    frame = pd.DataFrame({"sku": ["A", "B"], "revenue": [0.0, 0.0]})
    classified = abc_classify(frame)
    assert classified["abc_class"].tolist() == ["C", "C"]
    assert classified["revenue_share"].sum() == 0.0


def test_intermittent_classification_thresholds():
    assert classify_intermittent(1.0, 0.1) == "smooth"
    assert classify_intermittent(1.0, 0.6) == "erratic"
    assert classify_intermittent(2.0, 0.1) == "intermittent"
    assert classify_intermittent(2.0, 0.6) == "lumpy"
    assert classify_intermittent(None, 0.1) == "no_demand"
    assert classify_intermittent(np.inf, 0.1) == "no_demand"


def test_sku_metrics_adi_cv_and_classes(eda_daily):
    metrics = compute_sku_metrics(eda_daily).set_index("sku")
    assert metrics.loc["A", "units_sold"] == 6
    assert metrics.loc["A", "days_observed"] == 4
    assert metrics.loc["A", "non_zero_days"] == 3
    assert metrics.loc["A", "adi"] == pytest.approx(4 / 3)
    assert metrics.loc["A", "cv2_nonzero"] == pytest.approx(0.25)
    assert metrics.loc["A", "intermittent_class"] == "intermittent"
    assert metrics.loc["B", "adi"] == pytest.approx(4.0)
    assert metrics.loc["B", "intermittent_class"] == "intermittent"
    assert metrics.loc["C", "days_observed"] == 2
    assert metrics.loc["C", "zero_sales_share"] == pytest.approx(0.5)
    assert metrics["abc_class"].notna().all()
    assert metrics["units_share"].sum() == pytest.approx(1.0)
    assert metrics["revenue_share"].sum() == pytest.approx(1.0)


def test_abc_and_intermittent_summaries(eda_daily):
    metrics = compute_sku_metrics(eda_daily)
    abc = abc_summary(metrics)
    assert abc["skus"].sum() == 3
    assert abc["revenue"].sum() == pytest.approx(178.0)
    intermittent = intermittent_summary(metrics)
    assert intermittent["skus"].sum() == 3
    assert intermittent["skus"].dtype.kind in "iu"


def test_missing_date_summary_and_coverage(eda_daily):
    summary = missing_date_summary(eda_daily)
    assert summary["calendar_span_days"] == 4
    assert summary["observed_dates"] == 4
    assert summary["missing_calendar_days"] == 0
    assert summary["imputed_rows"] == 1
    assert summary["imputed_share"] == pytest.approx(0.1)
    assert summary["expected_sku_days"] == 11
    assert summary["observed_sku_days"] == 10
    assert summary["missing_sku_days"] == 1
    coverage = sku_date_coverage(eda_daily).set_index("sku")
    assert coverage.loc["C", "expected_days"] == 3
    assert coverage.loc["C", "missing_days"] == 1
    assert coverage.loc["C", "imputed_days"] == 1


def test_demand_concentration(eda_daily):
    metrics = compute_sku_metrics(eda_daily)
    concentration = demand_concentration(metrics)
    assert concentration["units_hhi"] == pytest.approx((6 / 12) ** 2 + (5 / 12) ** 2 + (1 / 12) ** 2)
    assert concentration["top10_units_share"] == pytest.approx(1.0)
    assert concentration["units_gini"] > 0


def test_category_metrics(eda_daily):
    categories = category_metrics(eda_daily).set_index("category")
    assert categories.loc["Earrings", "units_sold"] == 6
    assert categories.loc["Rings", "units_sold"] == 5
    assert categories.loc["Bangles", "units_sold"] == 1
    assert categories["units_share"].sum() == pytest.approx(1.0)


def test_price_summary_with_and_without_discount(eda_daily):
    summary = price_summary(eda_daily)
    assert summary["discount_available"] is True
    assert summary["price"]["rows"] == 10
    assert summary["positive_discount_rows"] == 1
    without_discount = eda_daily.drop(columns=["discount_percentage"])
    missing = price_summary(without_discount)
    assert missing["discount_available"] is False
    assert missing["discount_null_share"] == 1.0


def test_promotion_analysis(eda_daily):
    promotion = promotion_analysis(eda_daily)
    assert promotion["available"] is True
    assert promotion["summary"]["skus_compared"] == 1
    assert promotion["summary"]["promotion_rows"] == 2
    assert promotion["summary"]["non_promotion_rows"] == 8
    assert promotion["summary"]["active_avg_daily_units"] == pytest.approx(3.0)
    assert promotion["summary"]["inactive_avg_daily_units"] == pytest.approx(1.0)
    assert promotion["summary"]["uplift"] == pytest.approx(2.0)
    assert promotion["summary"]["median_sku_uplift"] == pytest.approx(2.0)
    assert set(promotion["by_category"]["segment"]) == {"promotion", "non_promotion"}


def test_festival_analysis_synthetic(eda_daily):
    festival = festival_analysis(eda_daily, mode="synthetic")
    summary = festival["festival_vs_non"]
    assert summary["active_days"] == 1
    assert summary["inactive_days"] == 3
    assert summary["active_avg_daily_units"] == pytest.approx(1.0)
    assert summary["inactive_avg_daily_units"] == pytest.approx(11.0 / 3.0)
    by_festival = festival["by_festival"].set_index("festival_name")
    assert by_festival.loc["FestivalX", "units_sold"] == 1


def test_inventory_analysis(eda_daily):
    inventory = inventory_analysis(eda_daily)
    assert inventory["available"] is True
    summary = inventory["summary"]
    assert summary["rows"] == 10
    assert summary["stockout_rows"] == 1
    assert summary["skus_with_stockout"] == 1
    by_sku = inventory["by_sku"].set_index("sku")
    assert by_sku.loc["A", "avg_stock"] == pytest.approx(7.25)
    assert by_sku.loc["A", "inventory_turnover"] == pytest.approx(6 / 7.25)
    assert by_sku.loc["A", "days_of_cover"] == pytest.approx(7.25 / 1.5)
    assert by_sku.loc["B", "stockout_frequency"] == pytest.approx(0.25)


def test_inventory_analysis_unavailable_for_uci_like_frame(eda_daily):
    frame = eda_daily.copy()
    frame["available_stock"] = np.nan
    inventory = inventory_analysis(frame)
    assert inventory["available"] is False
    assert "not fabricated" in inventory["reason"]
