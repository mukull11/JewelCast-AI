# Project Limitations

## Data limitations

1. **No real jewellery sales data.** UCI Online Retail II is UK general retail in
   GBP; M5 is a Walmart benchmark in USD; synthetic data is simulated. None is
   Voylla data.
2. **UCI is not jewellery-only.** Jewellery categories are derived from keyword
   rules and validated by SKU count. The taxonomy is heuristic, not ground truth.
3. **M5 is subsetted.** Only 30 items × 3 stores × 730 days are processed to keep
   memory bounded. Results do not represent the full competition dataset.
4. **No inventory in UCI/M5.** `available_stock` is null, not zero. Lost sales and
   stockout-censored demand cannot be identified for these modes.
5. **Synthetic assumptions.** Festival effects, promotions, price elasticity,
   lifecycle, inventory policy and purchase orders are generated from documented
   assumptions. They demonstrate methodology, not market reality.
6. **Historical periods.** UCI covers 2009–2011 and M5 covers 2014–2016. They
   should not be used to infer current Indian market behaviour.
7. **Calendar completion.** Missing SKU-days are added with zero sales and
   `date_imputed=1`. This is not invented demand, but it does treat an open day
   with no recorded transaction as a zero-sales observation.
8. **Outliers are flagged, not removed.** Extreme values remain in the data and
   must be handled by modelling choices or robust metrics.

## Methodological limitations (current status)

- Forecasting models, backtesting and multi-horizon forecasts are implemented in
  M3. Inventory optimization, historical inventory simulation, policy comparison
  and inventory scenarios are implemented in M4 for the synthetic dataset only.
  SHAP and the dashboard remain planned. No accuracy, savings or real-world
  service-level claims are made.
- Inventory optimization requires real or explicitly simulated inventory and
  purchase-order records. UCI and M5 have neither, so the inventory engine rejects
  those modes rather than fabricating stock.
- Synthetic inventory records are generated with an `(s, S)`-style policy and
  deterministic supplier lead times. They are not observed supplier, warehouse or
  purchasing behaviour.
- Synthetic observed sales are stockout-censored. The simulation uses them as a
  common demand proxy; it does not reconstruct latent demand, so absolute fill
  rates may be optimistic and true lost sales are unknown.
- The generator does not persist latent demand. Historical replenishment behaviour
  cannot be reconstructed fairly, so M4 compares a forecast-driven policy with a
  simple reorder-point policy instead of claiming a historical baseline.
- The committed quantity in weekly snapshots was derived from the following seven
  days of demand. It is excluded from decisions by default to avoid look-ahead.
- Safety stock uses a documented demand-volatility approximation with category
  fallbacks. A configured service level is a planning target, not proof that the
  service level is achieved.
- No purchase cost, holding cost, supplier MOQ, pack size or warehouse capacity
  exists in the synthetic dataset. These inputs are not invented; only units and
  cover days are reported.
- Inventory scenario results are model-based illustrations, not causal estimates.
- EDA findings are descriptive. Price, discount, festival and promotion
  associations are correlations, not causal elasticities, and may be confounded by
  category mix, seasonality, lifecycle and availability.
- UCI non-product audit codes are excluded from product EDA by default. Including
  them changes SKU counts and price/revenue concentration.
- M5 event analysis depends on the sparse `event_name` calendar field; event names
  are not forward-filled.
- The multi-horizon model uses recursive forecasting, which may accumulate error
  over longer horizons. Future prices, promotions and business decisions are held
  at observed values or set to zero where unknown.
- Cold-start forecasting relies on category/attribute similarity; it cannot
  recover true new-product demand without early sales.
- Scenario simulation produces model-based estimates, not causal effects.
- Deep learning (LSTM/GRU) and Prophet are optional and may be skipped on
  CPU-only machines.

## Operational limitations

- CPU-first, single-machine design. No GPU or distributed training is assumed.
- Parquet and SQLite are local-storage choices; PostgreSQL is a future extension.
- No paid APIs are required.
- Real dataset downloads are manual; M5 requires a Kaggle account.

## Ethical and communication constraints

- Synthetic data must always be labelled synthetic.
- UCI and M5 provenance must be stated whenever results are shown.
- Null inventory must never be presented as zero stock.
- Forecasts are estimates with uncertainty; they should support, not replace,
  business judgement.
