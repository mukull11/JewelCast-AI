"""Forecast artefact writing and Markdown reporting."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd

from .evaluation import comparison_table
from src.utils.io import save_parquet, write_json
from src.utils.paths import resolve_path

FORECAST_COLUMNS = (
    "dataset",
    "sku",
    "category",
    "forecast_origin_date",
    "forecast_date",
    "horizon_day",
    "forecast_horizon",
    "predicted_units",
    "model",
    "training_cutoff_date",
    "forecast_start_date",
    "forecast_end_date",
    "generated_at",
)

MODE_LABELS = {
    "synthetic": "Synthetic Indian fashion jewellery (simulated, INR)",
    "uci": "UCI Online Retail II (UK benchmark, GBP)",
    "m5": "M5 Walmart benchmark (USD)",
}

MODE_PROVENANCE = {
    "synthetic": "Fully simulated. Not Voylla or any real company's sales, inventory or pricing.",
    "uci": "UK online-retail transactions in GBP. Not Indian retail data and not jewellery-only.",
    "m5": "Walmart M5 benchmark in USD. Not jewellery data and not Voylla data.",
}


def _fmt(value: Any, digits: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not np.isfinite(number):
        return "n/a"
    return f"{number:,.{digits}f}"


def _md_table(frame: pd.DataFrame, columns: list[str], headers: list[str] | None = None, max_rows: int | None = None) -> str:
    if frame is None or frame.empty:
        return "_No data available._"
    available = [column for column in columns if column in frame.columns]
    if not available:
        return "_No data available._"
    view = frame.loc[:, available]
    if max_rows is not None:
        view = view.head(max_rows)
    headers = headers or available
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in view.iterrows():
        values = []
        for column in available:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                values.append(_fmt(value))
            elif isinstance(value, (int, np.integer)):
                values.append(f"{int(value):,}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def _prepare_forecast_frame(predictions: pd.DataFrame, mode: str, horizon: int, generated_at: str) -> pd.DataFrame:
    view = predictions.loc[predictions["horizon_day"] <= int(horizon)].copy()
    view = view.rename(columns={"date": "forecast_date", "origin_date": "forecast_origin_date"})
    view["dataset"] = mode
    view["forecast_horizon"] = int(horizon)
    view["training_cutoff_date"] = view["forecast_origin_date"]
    view["forecast_start_date"] = view["forecast_origin_date"] + pd.Timedelta(days=1)
    view["forecast_end_date"] = view["forecast_origin_date"] + pd.Timedelta(days=int(horizon))
    view["generated_at"] = generated_at
    for column in FORECAST_COLUMNS:
        if column not in view.columns:
            view[column] = pd.NA
    view = view.loc[:, list(FORECAST_COLUMNS)]
    view["predicted_units"] = pd.to_numeric(view["predicted_units"], errors="coerce").clip(lower=0.0)
    return view.sort_values(["forecast_origin_date", "sku", "forecast_date"], kind="mergesort").reset_index(drop=True)


def write_forecast_files(
    mode: str,
    predictions: pd.DataFrame,
    horizons: Iterable[int],
    output_dir: str | Path,
    generated_at: str | None = None,
    model_name: str | None = None,
) -> dict[int, dict[str, Path]]:
    """Write one Parquet forecast file and CSV summary per horizon."""
    generated_at = generated_at or datetime.now(timezone.utc).isoformat()
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    view = predictions.copy()
    if model_name is not None:
        view = view.loc[view["model"].astype(str) == str(model_name)].copy()
    if view.empty:
        return {}

    paths: dict[int, dict[str, Path]] = {}
    for horizon in sorted({int(value) for value in horizons}):
        frame = _prepare_forecast_frame(view, mode, horizon, generated_at)
        parquet_path = save_parquet(frame, target / f"forecast_h{horizon}.parquet")
        summary = (
            frame.groupby(["dataset", "model", "forecast_origin_date", "category"], observed=True)
            .agg(
                series=("sku", "nunique"),
                forecast_days=("forecast_date", "nunique"),
                total_predicted_units=("predicted_units", "sum"),
                mean_daily_units=("predicted_units", "mean"),
            )
            .reset_index()
            .sort_values(["forecast_origin_date", "category"])
        )
        summary_path = target / f"forecast_h{horizon}_summary.csv"
        summary.to_csv(summary_path, index=False, encoding="utf-8")
        head_path = target / f"forecast_h{horizon}_head.csv"
        frame.head(200).to_csv(head_path, index=False, encoding="utf-8")
        paths[horizon] = {"parquet": parquet_path, "summary": summary_path, "head": head_path}
    return paths


def write_actuals(mode: str, actuals: pd.DataFrame, output_dir: str | Path, name: str = "actuals.parquet") -> Path:
    frame = actuals.copy()
    frame["dataset"] = mode
    frame = frame.rename(columns={"date": "actual_date", "units_sold": "actual_units"})
    frame = frame.sort_values(["sku", "actual_date"], kind="mergesort").reset_index(drop=True)
    return save_parquet(frame, resolve_path(output_dir) / name)


def write_metrics(mode: str, run: Any, output_dir: str | Path) -> dict[str, Path]:
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    validation_horizon = run.validation_metrics.get("horizon_metrics", pd.DataFrame())
    test_horizon = run.test_metrics.get("horizon_metrics", pd.DataFrame())
    validation_daily = run.validation_metrics.get("daily_metrics", pd.DataFrame())
    test_daily = run.test_metrics.get("daily_metrics", pd.DataFrame())

    for name, frame in (
        ("validation_metrics.csv", validation_horizon),
        ("test_metrics.csv", test_horizon),
        ("validation_daily_metrics.csv", validation_daily),
        ("test_daily_metrics.csv", test_daily),
    ):
        if frame is not None and not frame.empty:
            path = target / name
            frame.to_csv(path, index=False, encoding="utf-8")
            paths[name] = path

    comparison = comparison_table(test_horizon, metric="wape", level="overall")
    if not comparison.empty:
        path = target / "model_comparison.csv"
        comparison.to_csv(path, index=False, encoding="utf-8")
        paths["model_comparison.csv"] = path

    validation_comparison = comparison_table(validation_horizon, metric="wape", level="overall")
    if not validation_comparison.empty:
        path = target / "validation_model_comparison.csv"
        validation_comparison.to_csv(path, index=False, encoding="utf-8")
        paths["validation_model_comparison.csv"] = path

    if not run.feature_importance.empty:
        path = target / "feature_importance.csv"
        run.feature_importance.to_csv(path, index=False, encoding="utf-8")
        paths["feature_importance.csv"] = path
    return paths


def _selection_table(selection: Mapping[str, Any]) -> pd.DataFrame:
    scores = selection.get("scores")
    if isinstance(scores, pd.DataFrame):
        return scores
    return pd.DataFrame()


def write_report(mode: str, run: Any, output_dir: str | Path) -> Path:
    target = resolve_path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    metadata = run.metadata
    split = metadata["split"]
    selection = run.selection
    test_metrics = run.test_metrics.get("horizon_metrics", pd.DataFrame())
    validation_metrics = run.validation_metrics.get("horizon_metrics", pd.DataFrame())

    test_overall = test_metrics.loc[test_metrics["level"] == "overall"] if not test_metrics.empty else pd.DataFrame()
    validation_overall = (
        validation_metrics.loc[validation_metrics["level"] == "overall"]
        if not validation_metrics.empty
        else pd.DataFrame()
    )
    selected_test = (
        test_overall.loc[test_overall["model"].astype(str) == run.selected_model_name]
        if not test_overall.empty
        else pd.DataFrame()
    )
    selected_category = (
        test_metrics.loc[
            (test_metrics["level"] == "category")
            & (test_metrics["model"].astype(str) == run.selected_model_name)
        ]
        if not test_metrics.empty
        else pd.DataFrame()
    )

    lines: list[str] = []
    lines.append(f"# Forecasting Report - {MODE_LABELS.get(mode, mode)}")
    lines.append("")
    lines.append(f"Generated: {metadata.get('generated_at')}")
    lines.append("")
    lines.append(f"**Provenance:** {MODE_PROVENANCE.get(mode, metadata.get('provenance', ''))}")
    lines.append("")
    lines.append(
        "This report is produced independently for one dataset mode. Synthetic, UCI and M5 "
        "series are never merged or trained together."
    )
    lines.append("")
    lines.append("## 1. Target and data")
    lines.append("")
    lines.append(f"- Target: {metadata.get('target_definition')}")
    lines.append(f"- Rows: {metadata['panel']['rows']:,}")
    lines.append(f"- Series: {metadata['panel']['skus']:,}")
    lines.append(f"- Categories: {metadata['panel']['categories']:,}")
    lines.append(f"- Date range: {metadata['panel']['date_min']} to {metadata['panel']['date_max']}")
    lines.append(f"- Calendar gap rows filled: {metadata['panel']['calendar_gap_rows_filled']:,}")
    lines.append(f"- Stockout-censoring note: {metadata.get('stockout_censoring')}")
    lines.append("")
    lines.append("### Chronological split")
    lines.append("")
    lines.append(
        f"- Train: {split['train_start']} to {split['train_end']} ({split['train_days']} days)"
    )
    lines.append(
        f"- Validation: {split['validation_start']} to {split['validation_end']} ({split['validation_days']} days)"
    )
    lines.append(f"- Test: {split['test_start']} to {split['test_end']} ({split['test_days']} days)")
    lines.append(f"- Validation forecast origins: {', '.join(metadata['validation_origins'])}")
    lines.append(f"- Test forecast origins: {', '.join(metadata['test_origins'])}")
    lines.append("")
    lines.append("## 2. Features")
    lines.append("")
    lines.append(
        "All historical features are causal. Lags are grouped by SKU. Rolling statistics are "
        "shifted by one day, so the target day is never included in its own feature. Price, "
        "discount and promotion features are lagged because future commercial decisions are "
        "not known at forecast time. Calendar features are known in advance."
    )
    lines.append("")
    lines.append(", ".join(f"`{name}`" for name in metadata.get("feature_names", [])))
    lines.append("")
    lines.append("## 3. Model selection")
    lines.append("")
    lines.append(
        f"Selection metric: mean validation **{selection.get('metric', 'wape').upper()}** across horizons "
        f"(level: {selection.get('level', 'overall')})."
    )
    lines.append("")
    selection_table = _selection_table(selection)
    if not selection_table.empty:
        lines.append(_md_table(selection_table, list(selection_table.columns)))
        lines.append("")
    lines.append(f"**Selected model:** `{run.selected_model_name}`")
    lines.append("")
    lines.append("Validation comparison (WAPE by horizon):")
    lines.append("")
    validation_comparison = comparison_table(validation_overall, metric="wape", level="overall")
    if not validation_comparison.empty:
        lines.append(_md_table(validation_comparison, list(validation_comparison.columns)))
    else:
        lines.append("_No validation comparison available._")
    lines.append("")
    lines.append("## 4. Test performance")
    lines.append("")
    lines.append("Test metrics are reported only after model selection. Actual values are kept separate from predictions.")
    lines.append("")
    if not test_overall.empty:
        view = test_overall.loc[
            :,
            ["model", "horizon", "n", "actual_sum", "pred_sum", "mae", "rmse", "wape", "smape"],
        ].sort_values(["horizon", "model"])
        lines.append(_md_table(view, list(view.columns)))
    else:
        lines.append("_No test metrics available._")
    lines.append("")
    if not selected_test.empty:
        lines.append("Selected model by horizon:")
        lines.append("")
        view = selected_test.loc[:, ["horizon", "n", "actual_sum", "pred_sum", "mae", "rmse", "wape", "smape"]]
        lines.append(_md_table(view, list(view.columns)))
        lines.append("")
    if not selected_category.empty:
        lines.append("Selected model by category and horizon (top 20 by actual units):")
        lines.append("")
        view = selected_category.loc[
            :, ["horizon", "category", "n", "actual_sum", "pred_sum", "mae", "rmse", "wape", "smape"]
        ].sort_values(["horizon", "actual_sum"], ascending=[True, False]).head(20)
        lines.append(_md_table(view, list(view.columns)))
        lines.append("")
    lines.append("## 5. Feature importance")
    lines.append("")
    if not run.feature_importance.empty:
        view = run.feature_importance.head(15).loc[:, ["feature", "importance", "importance_share"]]
        lines.append(_md_table(view, list(view.columns)))
        lines.append("")
        lines.append(
            "Feature importance describes how the model uses features. It is not evidence of "
            "causal effects."
        )
    else:
        lines.append("_Feature importance is not available for the selected model._")
    lines.append("")
    lines.append("## 6. Limitations")
    lines.append("")
    lines.append(
        "- Synthetic results measure recovery of the simulator, not real market accuracy. "
        "Synthetic records are not Voylla data."
    )
    lines.append(
        "- UCI is UK online-retail data in GBP and is not jewellery-only. M5 is a Walmart "
        "benchmark in USD. Neither is Indian jewellery data."
    )
    lines.append(
        "- Synthetic observed sales are stockout-censored. The model forecasts observed sales, "
        "not latent unconstrained demand."
    )
    lines.append(
        "- UCI and M5 have no inventory records, so lost sales and stockout-censored demand "
        "cannot be identified."
    )
    lines.append(
        "- Recursive forecasts accumulate error over longer horizons. Future prices, promotions "
        "and business decisions are held at their last observed values or set to zero where "
        "they are unknown."
    )
    lines.append(
        "- New or short-history SKUs use a category-level fallback; the model has not learned "
        "their individual history."
    )
    lines.append("")
    lines.append("## 7. Reproduce")
    lines.append("")
    lines.append("```powershell")
    lines.append(f"python scripts/train_forecasting.py --modes {mode}")
    lines.append(f"python scripts/evaluate_forecasting.py --modes {mode}")
    lines.append(f"python scripts/generate_forecasts.py --modes {mode}")
    lines.append("```")
    lines.append("")

    path = target / "FORECASTING_REPORT.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def save_model_artifacts(run: Any, config: Mapping[str, Any], model_dir: str | Path) -> dict[str, Any]:
    target = resolve_path(model_dir)
    target.mkdir(parents=True, exist_ok=True)
    selection_path = write_json(
        {
            "mode": run.mode,
            "selected_model": run.selected_model_name,
            "selection_metric": run.selection.get("metric"),
            "selection_level": run.selection.get("level"),
            "validation_origins": [str(origin.date()) for origin in run.validation_origins],
            "test_origins": [str(origin.date()) for origin in run.test_origins],
            "model_type": run.selected_model.model_type if run.selected_model is not None else None,
            "feature_names": run.feature_names,
            "scores": _selection_table(run.selection).to_dict(orient="records"),
        },
        target / "model_selection.json",
    )
    result: dict[str, Any] = {"selection": selection_path}
    if run.selected_model is not None and run.selected_model.model_type == run.selected_model_name:
        result.update(run.selected_model.save(model_dir))
    return result


def persist_run(
    run: Any,
    config: Mapping[str, Any],
    output_dir: str | Path | None = None,
    model_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write every M3 artefact for one mode and return the paths."""
    base = resolve_path(output_dir or (Path(config["paths"]["reports"]) / "forecasting" / run.mode))
    base.mkdir(parents=True, exist_ok=True)
    generated_at = str(run.metadata.get("generated_at"))

    paths: dict[str, Any] = {"output_dir": base}
    paths["validation_predictions"] = save_parquet(
        run.validation_predictions, base / "validation_predictions.parquet"
    )
    paths["test_predictions"] = save_parquet(run.test_predictions, base / "test_predictions.parquet")
    paths["validation_actuals"] = write_actuals(
        run.mode, run.validation_actuals, base, name="validation_actuals.parquet"
    )
    paths["test_actuals"] = write_actuals(run.mode, run.test_actuals, base, name="test_actuals.parquet")
    paths["metrics"] = write_metrics(run.mode, run, base)
    paths["forecasts"] = write_forecast_files(
        run.mode,
        run.test_predictions,
        run.metadata["horizons"],
        base / "forecasts",
        generated_at=generated_at,
        model_name=run.selected_model_name,
    )
    paths["report"] = write_report(run.mode, run, base)
    model_root = resolve_path(model_dir or (Path(config["paths"]["models"]) / "forecasting" / run.mode))
    paths["models"] = save_model_artifacts(run, config, model_root)
    paths["run_metadata"] = write_json(run.metadata, base / "run_metadata.json")
    return paths
