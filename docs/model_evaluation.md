# Model Evaluation Report

## Status

**Implemented in M3.** The forecasting engine, baselines, chronological splits,
rolling-origin backtests, evaluation metrics and recursive multi-horizon forecasts
are implemented under `src/forecasting/` with CLI entry points
`scripts/train_forecasting.py`, `scripts/evaluate_forecasting.py` and
`scripts/generate_forecasts.py`.

Actual numbers are produced by running the training, evaluation and forecast
scripts. No numbers are fabricated. Results are written to
`reports/forecasting/<mode>/`.

## Implemented evaluation protocol

- **Splits:** chronological train/validation/test only. No random shuffling.
- **Backtesting:** rolling-origin evaluation across multiple forecast origins.
- **Horizons:** 7, 15 and 30 days.
- **Levels:** individual SKU, category and overall business.
- **Metrics:** MAE, RMSE, WAPE and sMAPE. WAPE is primary because retail demand
  contains zero-sales days; sMAPE guards against zero denominators. MAPE is not
  used as the primary metric.
- **Model selection:** validation/backtest performance, never training accuracy.
- **Uncertainty:** quantile forecasts (p10/p50/p90) remain planned for a later
  milestone.
