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

- Forecasting models, backtesting, inventory optimization, SHAP and the dashboard
  are **not implemented yet**. No accuracy or savings claims are made.
- EDA findings are descriptive. Price, discount, festival and promotion
  associations are correlations, not causal elasticities, and may be confounded by
  category mix, seasonality, lifecycle and availability.
- UCI non-product audit codes are excluded from product EDA by default. Including
  them changes SKU counts and price/revenue concentration.
- M5 event analysis depends on the sparse `event_name` calendar field; event names
  are not forward-filled.
- The planned direct multi-horizon models may not capture complex cross-SKU
  substitution or cannibalisation.
- Cold-start forecasting will rely on category/attribute similarity; it cannot
  recover true new-product demand without early sales.
- Scenario simulation will produce model-based estimates, not causal effects.
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
