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

## Inventory evaluation (M4, synthetic only)

Inventory evaluation is separate from forecast accuracy. It measures how a
replenishment policy behaves in the causal historical simulation, not how accurate
a forecast is.

- **Inputs:** synthetic daily stock, observed sales (stockout-censored demand
  proxy), simulated purchase orders and per-SKU supplier lead times.
- **Protocol:** at each review date the decision uses start-of-day stock and a
  forecast whose origin is the previous day. Orders arrive after the supplier lead
  time. Every policy starts from the same on-hand stock (and, by default, an empty
  pipeline) and faces the same demand proxy and lead times.
- **Policies compared:** forecast-driven (M3 model) and simple reorder point
  (trailing-mean forecast). Historical replenishment behaviour is not reconstructed
  because latent demand is not stored.
- **Metrics:** fill rate, SKU-day cycle service level, stockout frequency, lost-sales
  proxy, average inventory, inventory turnover, days of inventory, excess units and
  replenishment count/units. Holding cost is not calculated because no validated
  cost input exists.
- **No-look-ahead:** forecast origins are always `decision_date - 1`; tests assert
  this. Model providers refuse a saved artefact whose training cutoff post-dates the
  requested origin.
- **Scenarios:** demand ±10/20%, lead time +7 days and service level 97/99% are
  model-based, one-assumption-at-a-time comparisons with all other inputs fixed.

Actual numbers are produced only by running:

```powershell
python scripts/simulate_inventory.py --mode synthetic
python scripts/evaluate_inventory.py
```

Outputs are written to `reports/inventory/`. See
`docs/inventory_optimization.md` for full formulas, assumptions and limitations.
No service-level guarantee, cost saving or real-world performance improvement is
claimed.
