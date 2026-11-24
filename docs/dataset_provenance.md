# Dataset Provenance Report

This report documents the origin, treatment and limitations of every dataset used
by JewelCast AI. Machine-readable hashes, sizes and row counts are in
`data/dataset_metadata.json`.

## Summary

| Mode | Origin | Real? | Currency | Period | Daily rows | SKUs |
|---|---|---|---:|---|---:|---:|
| synthetic | JewelCast AI generator | No (simulated) | INR | 2023-01-01 → 2024-12-31 | 175,165 | 300 |
| uci | UCI Online Retail II | Yes (public) | GBP | 2009-12-01 → 2011-12-09 | 2,292,587 | 4,916 |
| m5 | M5 Forecasting Accuracy | Yes (public benchmark) | USD | 2014-04-26 → 2016-04-24 | 65,700 | 90 |

## UCI Online Retail II

- **Official source:** <https://archive.ics.uci.edu/dataset/502/online+retail+ii>
- **Local archive:** `online+retail+ii.zip` (preserved, git-ignored)
- **Extracted file:** `data/raw/online_retail_II.xlsx`
- **Sheets:** `Year 2009-2010` (525,461 data rows), `Year 2010-2011` (541,910).
- **Actual columns:** `Invoice`, `StockCode`, `Description`, `Quantity`,
  `InvoiceDate`, `Price`, `Customer ID`, `Country`.
- **Raw rows:** 1,067,371.
- **Valid sale rows:** 1,007,913.
- **Excluded rows:** 59,458, with a recorded reason per row:
  - cancellations: 19,494
  - negative-quantity returns: 3,457
  - non-positive price: 2,750
  - duplicate transactions: 33,757
  - missing SKU / invalid date: 0 / 0
- **Audit trail:** `data/processed/uci/transactions_clean.parquet` retains every
  row with `is_cancellation`, `is_return`, `is_duplicate`, `is_non_product`,
  `exclusion_reason` and `is_valid_sale`.
- **Jewellery classification:** keyword rules on descriptions, checked after
  exclusions (e.g. `key ring`, `jewellery box`). Category order prioritises
  `Jewellery Sets` before generic terms. **357 jewellery SKUs** were found and
  validated (minimum 20). 4,559 SKUs remain `Other`; the dataset is not
  jewellery-only.
- **Currency:** GBP. Dates are historical. **Not Indian retail data.**

## M5 Forecasting Accuracy

- **Official source:** <https://www.kaggle.com/competitions/m5-forecasting-accuracy/data>
- **Local archive:** `m5-forecasting-accuracy.zip` (preserved, git-ignored)
- **Extracted files:** `sales_train_validation.csv` (30,490 × 1,919),
  `calendar.csv` (1,969 rows), `sell_prices.csv` (6,841,121 rows), plus optional
  evaluation and sample-submission files.
- **Subset:** 30 items (stratified by category) × 3 stores (one per state) ×
  730 days = 90 SKU-store series, 65,700 rows.
- **Price join:** `(store_id, item_id, wm_yr_wk)` from `sell_prices.csv`; missing
  prices forward-filled within series; zero missing prices in the current build.
- **Hierarchies preserved:** item, department, category, store and state IDs are
  retained in the SKU master.
- **Currency:** USD (M5 prices). **Walmart benchmark, not jewellery data and not
  Voylla data.**
- **Events:** M5 US calendar events are stored in `event_name`; `festival_flag` is
  0 for M5 to avoid misrepresenting them as Indian festivals.

## Synthetic Indian Jewellery Dataset

- **Origin:** `src/data/synthetic.py`, seed 42 (configurable).
- **Nature:** fully simulated. **Not real company data.**
- **Products:** 300 SKUs across 8 categories, with material, design, price,
  launch date, lead time, popularity and elasticity attributes.
- **Demand model:** negative-binomial draws around a mean driven by category base
  demand, SKU popularity, weekday/month seasonality, festival and wedding-season
  multipliers, price relative to category, lifecycle ramp/decline and discount
  elasticity.
- **Festivals:** configurable Indian calendar (Makar Sankranti, Valentine's Day,
  Holi, Akshaya Tritiya, Raksha Bandhan, Janmashtami, Ganesh Chaturthi, Navratri,
  Durga Puja, Karva Chauth, Dhanteras, Diwali, Bhai Dooj, Christmas) plus wedding
  seasons. These are documented assumptions.
- **Inventory:** simulated (s, S)-style replenishment with lead times, weekly
  snapshots and purchase orders. Sales are censored at zero stock and flagged with
  `stockout_flag`.
- **Outputs:** 175,165 SKU-day rows, 25,103 weekly inventory snapshots, 10,952
  purchase orders.
- **Reproducibility:** re-running with the same seed produces identical sales,
  products and purchase orders (tested).

## Prohibited interpretations

- Do not describe UCI as jewellery-only or Indian retail data.
- Do not describe M5 as jewellery or Voylla data.
- Do not describe synthetic records as real company sales or inventory.
- Do not treat null `available_stock` in UCI/M5 as zero inventory.
- Do not interpret `date_imputed` rows as observed demand; they are zero-sales
  calendar completions.

## Inventory optimization data use (M4)

Inventory optimization is implemented for the **synthetic dataset only**. It uses
the simulated daily `available_stock`, weekly inventory snapshots and simulated
purchase orders described above. The weekly `committed_qty` was derived from the
following seven days of demand and is excluded from historical decisions by
default to avoid look-ahead. UCI and M5 have no inventory or purchase-order
records; the inventory engine rejects those modes rather than assigning inventory.
No supplier cost, MOQ, pack-size, warehouse-capacity or holding-cost data exists.
