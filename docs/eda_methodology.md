# EDA Methodology

This document defines every metric produced by the M2 exploratory data analysis
pipeline. The EDA is **descriptive only**: it does not create lags, rolling
windows, future targets or any other forecasting feature, so it introduces no
leakage into later modelling stages.

## Scope and separation

- The three modes (`synthetic`, `uci`, `m5`) are analysed independently and are
  never combined.
- Synthetic uses the configured Indian festival and wedding calendar.
- UCI uses only the documented UK bank-holiday calendar.
- M5 uses its own `event_name` calendar feature.
- UCI and M5 have no inventory records, so inventory metrics are reported as
  unavailable rather than fabricated.
- UCI postage, manual, bad-debt, fee and similar non-product codes are excluded
  from EDA by default. They remain in the processed audit table. Use
  `--include-non-products` to include them.

## Inputs and outputs

Inputs are the processed Parquet artefacts under
`data/processed/<mode>/daily_sales.parquet`. Synthetic additionally reads
`inventory_snapshots.parquet` and `purchase_orders.parquet`.

For each mode, `scripts/run_eda.py` writes:

```text
reports/eda/<mode>/
├── EDA_REPORT.md
├── summary.json
├── tables/*.csv
└── charts/*.png and *.html
```

A top-level `reports/eda/manifest.json` records the modes analysed, output paths
and headline results. PNG charts use Matplotlib; HTML charts use Plotly with a
CDN script include.

## Overview metrics

| Metric | Definition |
|---|---|
| `rows` | Number of SKU-day rows analysed. |
| `skus` | Distinct SKU count. |
| `active_skus` | SKUs with total units > 0. |
| `inactive_skus` | SKUs with total units <= 0. |
| `active_sku_share` | `active_skus / skus`. |
| `categories` | Distinct non-null categories. |
| `days` | Distinct observed dates. |
| `units_sold` | Sum of `units_sold`. |
| `revenue` | Sum of `revenue`. |
| `zero_sales_rows` | Rows with `units_sold == 0`. |
| `zero_sales_share` | `zero_sales_rows / rows`. |
| `mean_units_per_sku_day` | Arithmetic mean of row-level units. |
| `median_units_per_sku_day` | Median of row-level units. |
| `price_available` | True when at least one non-null selling price exists. |
| `price_null_share` | Share of rows with null selling price. |
| `inventory_available` | True when at least one non-null available-stock value exists. |

## Date coverage and missing data

- `calendar_span_days = max(date) - min(date) + 1`.
- `observed_dates` is the number of distinct dates in the dataset.
- `missing_calendar_days = calendar_span_days - observed_dates`.
- `imputed_rows` is the sum of `date_imputed`.
- `imputed_share = imputed_rows / rows`.
- For each SKU, `expected_days = last_date - first_date + 1`.
- `expected_sku_days = sum(expected_days)`.
- `observed_sku_days = sum(observed rows per SKU)`.
- `missing_sku_days = expected_sku_days - observed_sku_days`.

After successful date completion, `missing_sku_days` should be zero. A non-zero
value indicates a gap in the processed dataset. `date_imputed=1` means the row was
created to complete an open calendar day; it is a zero-sales observation, not
invented demand.

## Demand trends

Daily, weekly (`W-SUN`, weeks starting Monday) and monthly aggregates contain:

- `units_sold` and `revenue` summed within the period;
- `rows` (SKU-day observations);
- `zero_sales_rows` and `zero_sales_share`;
- `date_imputed_rows`;
- `active_skus` = distinct SKUs with positive units in the period;
- `days` = distinct observed dates in the period.

## SKU-level metrics

For each SKU:

- `days_observed` = number of SKU-day rows.
- `units_sold` and `revenue` = period totals.
- `zero_days` / `non_zero_days` = count of zero / positive unit days.
- `mean_units`, `std_units` = mean and sample standard deviation of all daily
  units.
- `mean_price`, `min_price`, `max_price` = selling-price statistics.
- `promotion_days` = sum of `promotion_flag`.

Derived metrics:

- `zero_sales_share = zero_days / days_observed`.
- `ADI` (average inter-demand interval) = `days_observed / non_zero_days`.
  Undefined when there is no demand.
- `CV_all = std_units / mean_units` (all days).
- `CV²_nonzero = (std(non-zero units) / mean(non-zero units))²`. This is set to
  zero when a SKU has a single non-zero demand observation.
- `units_share = SKU units / total units`.
- `revenue_share = SKU revenue / total revenue`.

## ABC classification

SKUs are sorted by revenue descending with a stable sort. With total revenue
`T`, each SKU has:

- `revenue_share = revenue / T`;
- `cumulative_revenue_share = cumsum(revenue_share)`;
- `previous_share` = cumulative share before the current SKU.

Class assignment:

- **A** when `previous_share < 0.80`;
- **B** when `0.80 <= previous_share < 0.95`;
- **C** otherwise.

The item that crosses the 80% boundary is included in A, and the item that
crosses 95% is included in B. If total revenue is zero, every SKU is assigned to
C. The summary table reports SKU count, SKU share, units, revenue and revenue
share per class.

## Intermittent demand (Syntetos-Boylan)

The four-way Syntetos-Boylan classification uses:

- `ADI` threshold = 1.32;
- `CV²` threshold = 0.49.

| ADI | CV² | Class |
|---|---|---|
| `< 1.32` | `< 0.49` | smooth |
| `< 1.32` | `>= 0.49` | erratic |
| `>= 1.32` | `< 0.49` | intermittent |
| `>= 1.32` | `>= 0.49` | lumpy |
| undefined | undefined | no_demand |

The summary reports SKU count, SKU share, units, revenue, mean ADI, median ADI and
mean CV² per class.

## Demand concentration

Using SKU totals `x` sorted ascending:

- **Gini** = `(n + 1 - 2 * sum(cumsum(x)) / sum(x)) / n`, with 0 for zero totals.
- **HHI** = `sum((x / sum(x))²)` on the 0–1 scale.
- **Top-N share** = sum of the N largest SKU values divided by the total.

Gini and HHI are reported separately for units and revenue, along with top-10 and
top-20 shares.

## Category analysis

Category metrics include:

- SKU count and active SKU count;
- rows, units, revenue and zero-sales share;
- average selling price;
- promotion row share;
- unit share and revenue share.

`category_monthly` aggregates units, revenue and distinct days per category-month.
`category_seasonality` computes:

- `avg_daily_units = category-month units / category-month observed days`;
- `category_avg_daily_units = category units / category observed days`;
- `seasonality_index = avg_daily_units / category_avg_daily_units`.

An index of 1.0 is the category baseline.

## Weekday and month profiles

Demand is first aggregated to date level, then grouped by weekday or month:

- `avg_daily_units = period units / period observed days`;
- `avg_daily_revenue = period revenue / period observed days`;
- `units_index = avg_daily_units / overall average daily units`.

Weekend uplift is:

```text
weekend_avg_daily_units / weekday_avg_daily_units - 1
```

## Calendar-event analysis

For all modes, event uplift is:

```text
active_avg_daily_units / inactive_avg_daily_units - 1
```

- **Synthetic:** `festival_flag=1` defines festival days; configured wedding
  seasons define wedding days. `by_festival` also reports average daily units and
  uplift versus non-festival days.
- **UCI:** `festival_flag=1` marks documented UK bank holidays only.
- **M5:** `event_flag=1` when the sparse `event_name` value is present. Events are
  not forward-filled.

Calendar labels are never transferred between modes.

## Pricing and discount analysis

- Price statistics: count, mean, standard deviation, min, p25, median, p75, p95
  and max over non-null selling prices.
- Discount statistics use the same distribution summary over non-null
  `discount_percentage`.
- `price_null_share` and `discount_null_share` describe missingness.
- `positive_discount_rows` / `zero_discount_rows` count known discounts.
- Price-demand bins use pooled price quantiles (`qcut`, up to 10 bins). Each bin
  reports rows, units, revenue, average price, average units per row and
  zero-sales share.
- Within-SKU price-demand association uses Spearman correlation between selling
  price and units, requiring at least 30 priced observations and at least two
  distinct prices.
- Category price sensitivity summarises the mean, median and negative share of
  those within-SKU correlations.
- Discount-demand association uses the same within-SKU Spearman approach.

These are **exploratory associations, not causal price elasticities**. Price,
discount, festival and promotion effects can be confounded by category mix,
seasonality, product lifecycle and inventory availability.

## Promotion analysis

Promotions in the synthetic dataset are SKU-specific, so a date is **not**
classified as promoted merely because one SKU is on promotion. For each SKU, the
pipeline computes:

- `promotion_avg = mean units on promoted days`;
- `non_promotion_avg = mean units on non-promoted days`;
- `sku_uplift = promotion_avg / non_promotion_avg - 1`.

Only SKUs with both promoted and non-promoted observations and a positive
non-promotion baseline enter the paired comparison. The summary reports the mean
promoted and non-promoted SKU-day averages, the ratio-of-means uplift, the median
per-SKU uplift, the number of SKUs compared, and row counts. Category tables use
the same paired approach. UCI and M5 have no promotion records, so the analysis is
reported as unavailable.

## Inventory analysis (synthetic only)

Inventory metrics are computed only when `available_stock` has observed values.
For each SKU:

- `avg_daily_units = total_units / days_observed`;
- `inventory_turnover = total_units / mean_on_hand_stock`;
- `annualised_turnover = inventory_turnover * 365 / days_observed`;
- `days_of_cover = mean_on_hand_stock / avg_daily_units`;
- `stockout_frequency = stockout_days / days_observed`.

The summary also reports mean/median stock, zero-stock share, stockout rows,
SKUs with any stockout, and mean/median turnover and days of cover. Weekly
inventory snapshots and purchase orders are summarised separately, including
mean stock, incoming PO quantity, committed quantity, order quantity, lead time,
orders per SKU and orders per SKU per year.

UCI and M5 return `available: false` with an explicit reason. Null inventory is
never interpreted as zero stock.

## Reproducibility

```powershell
# Rebuild the processed datasets with the current preprocessing code
python scripts/build_datasets.py --all

# Run all EDA modes
python scripts/run_eda.py --all

# Run one mode, or a bounded subset
python scripts/run_eda.py --modes synthetic
python scripts/run_eda.py --modes uci --sku-limit 200
python scripts/run_eda.py --modes m5 --no-html

# Include UCI non-product audit rows (not recommended for product analysis)
python scripts/run_eda.py --modes uci --include-non-products

# Run the full test suite
python -m pytest -q
```

## Caveats

- UCI is UK general retail in GBP, not Indian retail and not jewellery-only.
- M5 is a Walmart benchmark subset in USD, not jewellery data.
- Synthetic demand, festivals, promotions and inventory are simulated from
  documented assumptions.
- Zero-sales rows created by date completion are observations for open days, not
  invented demand.
- Correlation is not causation.
- The EDA does not evaluate forecast accuracy; no forecasting models are trained
  in M2.
