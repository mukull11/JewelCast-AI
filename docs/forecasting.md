# Forecasting Engine (M3)

## Scope

Milestone M3 implements the forecasting engine only. Inventory optimization,
reorder points, safety stock, purchase orders, the Streamlit dashboard, agentic
features and deep learning are explicitly out of scope.

## Problem and target definition

The forecasting target is **daily unit demand at SKU level** for horizons of
7, 15 and 30 days.

| Mode | Series identifier |
|---|---|
| `synthetic` | SKU id (simulated Indian fashion jewellery) |
| `uci` | StockCode (UK online retail benchmark) |
| `m5` | item-store combination |

Datasets are never merged. Each mode is trained, validated and evaluated with
its own features, split and artefacts.

For UCI, a day without a recorded transaction and a day with unavailable
information are distinguished through the loader audit flags; missing daily
series rows are completed with `date_imputed` flags. For the synthetic dataset,
observed sales are stockout-censored. The model therefore forecasts **observed
historical sales**, not latent unconstrained customer demand; this limitation is
documented and must be revisited before using forecasts for inventory policy.

## Feature engineering

- Lags: 1, 7, 14, 28, 30 days (grouped by series, shifted past observations only).
- Rolling statistics: 7/14/28-day rolling mean, 7/28-day rolling standard
  deviation, 28-day zero-demand share; shifted by one period so the target day
  is never part of its own feature.
- Calendar: day of week, day of month, month, quarter, week of year, day of year,
  weekend flag, month-end flag, days to month end, and mode-specific event flags
  (synthetic: Indian festivals/wedding seasons; UCI: UK bank holidays; M5: its own
  calendar events). No Indian festival features are injected into other modes.
- Product: category code, series age in days, historical mean and variability.
- Commercial: lagged price, discount and promotion features; category-relative
  price ratio; promotion share over a rolling window.

All features are historical-information only. Future prices, promotions and
business decisions are held at observed values or set to zero where unknown.

## Baselines and machine learning models

Baselines: naive, seasonal naive (lag 7), moving average (window 7). Short or
empty series fall back to category-level or global historical means.

Machine learning: global model trained per mode. Preferred LightGBM, with XGBoost
and scikit-learn HistGradientBoosting as fallbacks. Target transformed with
`log1p`, predictions inverted and clipped at zero.

## Temporal validation

Chronological split (default 70/15/15 of unique dates) into train, validation and
test periods. Validation uses rolling-origin backtests; the test period is held
out and evaluated once the best model is selected on validation only.
## Evaluation

Metrics: MAE, RMSE, WAPE (primary) and sMAPE. WAPE handles zero-sales days;
sMAPE guards against zero denominators. Reported overall, category and SKU level,
per horizon (7/15/30). Horizon metrics sum actual and predicted demand per series
over the forecast window.

## Forecast outputs

One Parquet forecast file per horizon plus CSV summary/head under
`reports/forecasting/<mode>/forecasts/`. Schema: dataset, sku, category,
forecast_origin_date, forecast_date, horizon_day, forecast_horizon, predicted_units,
model, training_cutoff_date, forecast_start_date, forecast_end_date, generated_at.

## Reproducibility and efficiency

Config-driven, fixed seeds, efficient dtypes, bounded training rows per mode,
CPU-first libraries only.

## Known limitations

- Synthetic results are simulator recovery, not business performance.
- UCI is UK retail in GBP; M5 is a Walmart benchmark in USD.
- Stockout censoring and unknown future commercial decisions.
- Recursive forecasts accumulate error.

## How to run

```powershell
python scripts/train_forecasting.py --modes synthetic
python scripts/evaluate_forecasting.py --modes synthetic
python scripts/generate_forecasts.py --modes synthetic
```