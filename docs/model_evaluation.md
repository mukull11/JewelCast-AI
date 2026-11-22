# Model Evaluation Report

## Status

**Not evaluated yet.** No forecasting model has been trained or scored. This file
will be updated with actual results after milestone M4. No numbers are fabricated.

## Planned evaluation protocol

- **Splits:** chronological train/validation/test only. No random shuffling.
- **Backtesting:** rolling-origin evaluation across multiple forecast origins.
- **Horizons:** 7, 15 and 30 days.
- **Levels:** individual SKU, category and overall business.
- **Metrics:**
  - MAE — mean absolute error.
  - RMSE — root mean squared error.
  - WAPE — weighted absolute percentage error, robust to zero-sales days.
  - sMAPE — symmetric mean absolute percentage error.
  - MAPE is not the primary metric because retail demand contains zeros.
- **Model selection:** validation/backtest performance, never training accuracy.
- **Uncertainty:** quantile forecasts (p10/p50/p90) where feasible; point
  forecasts and intervals will be reported separately.

## Planned comparison

| Model | Type | Status |
|---|---|---|
| Naive | Baseline | Not Evaluated |
| Seasonal Naive (lag 7) | Baseline | Not Evaluated |
| Moving Average | Baseline | Not Evaluated |
| SARIMA | Statistical | Not Evaluated |
| LightGBM | Global ML, direct horizons | Not Evaluated |
| XGBoost | Global ML, direct horizons | Not Evaluated |
| LSTM / GRU | Optional deep learning | Not Evaluated |

## Reporting requirements

When results are produced, the report must include:

- the exact dataset mode, date range and SKU subset;
- the split dates and backtest origins;
- per-horizon and per-level metrics;
- actual-vs-predicted charts and residual distributions;
- model artefact paths and training metadata;
- explicit disclosure when a model failed, was skipped or used a fallback.

## Limitations

- Metrics on synthetic data measure recovery of the simulator, not real market
  accuracy.
- UCI and M5 results are benchmark experiments, not jewellery business results.
- No financial impact will be claimed without measured evidence.
