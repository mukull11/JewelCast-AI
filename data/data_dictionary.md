# Data Dictionary

## Unified daily dataset (`daily_sales.parquet`)

One row per SKU per calendar day. Required columns are present for every mode;
optional fields remain null when the source does not provide them.

| Column | Type | Nullable | Description |
|---|---|---|---|
| `date` | datetime64[ns] | No | Normalized calendar date. |
| `sku` | string | No | Product identifier. UCI: `StockCode`; M5: `<item_id>_<store_id>`; synthetic: `JWL-...`. |
| `product_name` | string | No | Product name/description. M5 uses the item id (no description available). |
| `category` | string | No | Product category. Synthetic: 8 jewellery categories. UCI: derived jewellery categories or `Other`. M5: `cat_id`. |
| `units_sold` | int32 | No | Observed units sold (returns/cancellations excluded from UCI). |
| `selling_price` | float64 | Yes | Transaction/weighted unit price. Currency depends on mode. |
| `discount_percentage` | float64 | Yes | Discount applied. Null when unknown (UCI/M5). |
| `revenue` | float64 | No | `units_sold × selling_price` (rounded). |
| `festival_flag` | int8 | No | 1 on a configured festival/event day. UCI uses UK bank holidays; M5 uses 0. |
| `festival_name` | string | Yes | Name of the active festival/holiday on that date. Sparse: null on non-festival/non-holiday days. Not forward-filled. |
| `promotion_flag` | int8 | No | 1 when a promotion is known to be active. |
| `available_stock` | float64 | Yes | Stock available at the start of the day. **Null means unknown, not zero.** |
| `stockout_flag` | Int8 | Yes | 1 when observed demand exceeded available stock (synthetic only). |
| `event_name` | string | Yes | M5 US calendar event name; null on non-event days and for other modes. Sparse: not forward-filled. |
| `date_imputed` | int8 | No | 1 when the row was created to complete a missing calendar day. |
| `data_source` | string | No | `synthetic`, `uci` or `m5`. Never mixed within a mode. |

### Null-inventory semantics

- **Synthetic:** inventory is simulated and populated.
- **UCI / M5:** no inventory records exist, so `available_stock` is entirely null.
  It must not be interpreted as zero stock, and lost demand cannot be estimated.

### Calendar-field semantics

`festival_name` and `event_name` describe only the specific date where the event
occurs. Date completion forward-fills product attributes and prices, but it does
**not** forward-fill calendar event names. This keeps M5 event-day analysis
limited to actual event dates.

## SKU master (`sku_master.parquet`)

Common columns: `sku`, `product_name`, `category`, `data_source`.

Synthetic additionally includes: `material`, `design_type`, `list_price`,
`launch_date`, `supplier_lead_time_days`, `initial_inventory`,
`popularity_score`, `festival_sensitivity`, `price_sensitivity`,
`discount_elasticity`, `category_median_price`, `product_age_days`,
`is_cold_start`.

UCI additionally includes: `total_units`, `total_revenue`, `first_sale_date`,
`last_sale_date`, `avg_unit_price`, `is_jewellery`, `is_product`.

M5 additionally includes: `item_id`, `store_id`, `dept_id`, `cat_id`, `state_id`,
`total_units`, `total_revenue`, `avg_price`, `first_date`, `last_date`.

## Synthetic inventory (`inventory_snapshots.parquet`)

| Column | Description |
|---|---|
| `snapshot_date` | Weekly snapshot date (Mondays). |
| `sku` | Product identifier. |
| `available_stock` | Simulated on-hand stock after the day's sales. |
| `incoming_po_qty` | Open purchase-order quantity. |
| `committed_qty` | Simulated outstanding commitments (10% of next 7 days' demand). |
| `data_source` | `synthetic`. |

## Synthetic purchase orders (`purchase_orders.parquet`)

| Column | Description |
|---|---|
| `po_id` | Purchase-order identifier. |
| `sku` | Product identifier. |
| `order_date` | Simulated order date. |
| `expected_arrival_date` | Order date + supplier lead time. |
| `quantity` | Ordered units. |
| `status` | `received` or `open` at the end of the simulation window. |
| `supplier_lead_time_days` | Lead time used for the order. |
| `data_source` | `synthetic`. |

## Raw UCI Online Retail II columns

The actual workbook uses `Invoice`, `StockCode`, `Description`, `Quantity`,
`InvoiceDate`, `Price`, `Customer ID`, `Country`. The loader also accepts the
`InvoiceNo` / `UnitPrice` / `CustomerID` naming variant. Sheets:
`Year 2009-2010`, `Year 2010-2011`. Currency: GBP.

## Raw M5 files

- `sales_train_validation.csv`: `id`, `item_id`, `dept_id`, `cat_id`, `store_id`,
  `state_id`, `d_1` … `d_1913`.
- `calendar.csv`: `date`, `wm_yr_wk`, `weekday`, `wday`, `month`, `year`, `d`,
  `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, SNAP flags.
- `sell_prices.csv`: `store_id`, `item_id`, `wm_yr_wk`, `sell_price`.

## Festival calendar

`festivals.indian` in `config.yaml` defines date, type, strength and window for
each Indian festival. `festivals.wedding_seasons` defines wedding-season periods.
These are **documented assumptions for synthetic data**, not observed events.

## Inventory optimization (M4)

The M4 inventory engine uses the synthetic daily stock, weekly snapshots and
purchase orders above. `available_stock` from `daily_sales.parquet` is start-of-day
stock; pending purchase orders are aligned by `order_date`, `expected_arrival_date`
and `status` and de-duplicated by `po_id`. The snapshot `committed_qty` was derived
from future demand by the generator and is excluded from decisions by default.
UCI and M5 have no inventory or purchase-order records and are rejected by the
inventory engine; they remain forecasting-only.
