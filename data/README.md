# Data Directory

This directory holds raw, processed, external and synthetic data. **No dataset is
committed to Git.** The folder structure is preserved with `.gitkeep` files, and
all data can be regenerated locally.

## Provenance summary

| Mode | Source | Currency | Real or synthetic | Not to be described as |
|---|---|---|---|---|
| `synthetic` | JewelCast AI generator | INR | **Synthetic** | Voylla or any real company data |
| `uci` | UCI Online Retail II (UK) | GBP | Real public transactions | Indian retail or jewellery-only data |
| `m5` | M5 Forecasting Accuracy (Walmart) | USD | Real public benchmark | Jewellery or Voylla data |

## Layout

```
data/
├── raw/          Original extracted files (xlsx/csv). Git-ignored.
├── processed/    Cleaned SKU-day Parquet datasets, one folder per mode.
├── synthetic/    Generator outputs (products, sales, inventory, POs).
├── external/     Optional third-party reference files.
├── dataset_metadata.json   Hashes, sizes, row counts, schemas, provenance.
├── data_dictionary.md      Column-level documentation.
└── README.md
```

## Placing the real datasets

1. Keep the official archives in the repository root (they are git-ignored):
   - `online+retail+ii.zip`
   - `m5-forecasting-accuracy.zip`
2. Extract them without deleting the archives:

```powershell
python scripts/download_data.py
```

3. The extractor writes `online_retail_II.xlsx`, `sales_train_validation.csv`,
   `calendar.csv`, `sell_prices.csv`, and the optional M5 evaluation/sample files
   into `data/raw/`.

If the archives are missing, the script prints official download links and does
not attempt to download anything. M5 requires a Kaggle account.

## Build processed datasets

```powershell
# All modes (UCI takes a few minutes; M5 ~10 seconds)
python scripts/build_datasets.py --all

# One mode
python scripts/build_datasets.py --modes synthetic
python scripts/build_datasets.py --modes uci
python scripts/build_datasets.py --modes m5
```

Outputs per mode under `data/processed/<mode>/`:

- `daily_sales.parquet` — unified SKU-day dataset
- `sku_master.parquet` — product attributes
- `quality_report.json` / `quality_report.md`
- `dataset_summary.json`
- `provenance.json`
- `audit.json` (UCI/M5 source audit)
- `transactions_clean.parquet` (UCI transaction-level audit trail)
- `inventory_snapshots.parquet` and `purchase_orders.parquet` (synthetic only)

## Actual build statistics (current local build)

| Mode | Daily rows | SKUs | Categories | Date range | Units | Revenue |
|---|---:|---:|---:|---|---:|---:|
| synthetic | 175,165 | 300 | 8 | 2023-01-01 → 2024-12-31 | 372,343 | ₹481,972,741.96 |
| uci | 2,292,587 | 4,916 | 9 | 2009-12-01 → 2011-12-09 | 11,205,148 | £20,476,260.43 |
| m5 | 65,700 | 90 | 3 | 2014-04-26 → 2016-04-24 | 1,037,008 | $1,642,029.57 |

Synthetic inventory: 25,103 weekly snapshots and 10,952 purchase orders.
UCI jewellery subset: **357 SKUs** (validated above the 20-SKU minimum).
M5 subset: 30 items × 3 stores, one store per state, 730 days.

## Important limitations

- **Synthetic data is simulated.** It is reproducible from `synthetic_config.json`
  and must never be presented as real sales.
- **UCI is UK/GBP and not jewellery-only.** The jewellery view uses transparent
  keyword rules; 4,559 of 4,916 SKUs remain `Other`.
- **M5 is a Walmart benchmark.** It is not jewellery and not Voylla data. Only a
  subset is processed for memory reasons.
- **UCI and M5 have no inventory records.** `available_stock` is null, not zero,
  and true lost demand cannot be identified.
- Missing calendar days are filled per SKU with `date_imputed=1`; this is not
  invented demand, just a zero-sales observation for an open day.
- Outliers are flagged in quality reports but never deleted.

## Regenerating metadata

```powershell
python scripts/build_metadata.py
```

`dataset_metadata.json` records SHA-256 hashes of the source archives and
extracted files, row counts, sheet names, schemas, processed artefact hashes and
provenance statements.
