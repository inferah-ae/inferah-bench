# Semantic layer: `orders` table

This is the complete, authoritative documentation of the dataset you are
analyzing. Everything an analyst could ask for is here: grain, column
definitions, metric formulas, decomposition identities, and the standard
pitfalls of averaged metrics.

## Grain and scope

* **One row = one delivered order.** No returns, cancellations, test orders or
  duplicates — the table is pre-cleaned.
* The table covers exactly **two comparison windows**:
  * `period = 'p0'` — the baseline week, days `2026-05-04 .. 2026-05-10`;
  * `period = 'p1'` — the current week, days `2026-05-11 .. 2026-05-17`.
* A healthy load contains **7 distinct `ts_day` values per period** and every
  active segment present on every day. Before attributing any change to
  demand, verify completeness: compare `COUNT(DISTINCT ts_day)` and the set of
  segments per period. **A segment with zero rows in one period, or a missing
  day, is a pipeline/data-load failure signature, not a demand signal.**

## Columns

| column | type | meaning |
|---|---|---|
| `order_id` | integer | Unique order identifier. |
| `ts_day` | date | Calendar day the order was placed. |
| `period` | text | Comparison window: `'p0'` (baseline) or `'p1'` (current). |
| `country` | text, nullable | Delivery country (fictional market names). **NULL means the order came from a source that is not geo-enriched** — these rows are real revenue but belong to no modeled country. |
| `city` | text, nullable | Delivery city; each city belongs to exactly one country. NULL has the same meaning as for `country`. |
| `order_type` | text | `'organic'` (full price) or `'promo'` (discounted basket). |
| `category` | text, nullable | Product category: electronics, fashion, grocery, home. NULL = category not mapped for that source. |
| `gmv` | numeric | Gross merchandise value of the order, USD. |
| `user_id` | text | Pseudonymous buyer identifier, stable across periods. |

## Metrics and formulas

* **GMV** `= SUM(gmv)` — the metric every question is about. Extensive
  (additive across any segment dimension).
* **Orders** `= COUNT(*)` — extensive.
* **Buyers** `= COUNT(DISTINCT user_id)` — *not* additive across dimensions
  (a user can buy in two categories); never sum per-segment buyer counts.
* **AOV** `= GMV / Orders` — average order value. **Intensive: never average
  AOVs across segments and never sum them.**
* **Frequency** `= Orders / Buyers`.

Decomposition identities (each exact, in both periods):

```
GMV = Orders x AOV
Orders = Buyers x Frequency
GMV = Buyers x Frequency x AOV
```

A change in GMV must be attributable through these identities: volume moves
live in Orders (and below it in Buyers or Frequency); basket moves live in
AOV.

## Mandatory caution: mix effects in averaged metrics

When analyzing any **average** (AOV is the canonical one), remember it can
move while **no underlying segment's own average moved at all** — purely
because the order mix shifted between segments with different levels
(Simpson's paradox). Before concluding "baskets got cheaper" (a *rate*
change), you MUST check per-segment rates:

```sql
SELECT period, order_type, COUNT(*) AS orders, AVG(gmv) AS aov
FROM orders GROUP BY 1, 2 ORDER BY 2, 1;
```

* If per-segment AOV is unchanged but segment shares moved → the change is
  **mix**, not rate. Repeat the check on `category` as well: a shift from
  expensive to cheap categories moves aggregate AOV with every category's own
  AOV flat.
* The reverse also happens: a flat aggregate can hide compensating segment
  moves. A stable total is not evidence that nothing happened.

## Honest answers

* **Rows with NULL `country`/`city`/`category` are real revenue that no
  modeled dimension explains.** If the period-over-period move is concentrated
  in NULL rows, no split by the mapped dimensions can account for it — the
  correct conclusion is that the driver is **outside the modeled dimensions**
  (an unmapped source), not whichever mapped segment happens to wiggle.
* **A move that is uniformly present in every segment of every dimension**
  (same relative change in each country, city, category, and order type, with
  stable mix and volume) is not localizable to any column. The honest
  conclusion is an **external, market-wide cause** — do not pick a segment at
  random.
* **Data completeness comes before causal attribution.** If `p1` is missing a
  day, or a whole segment has zero rows, or a segment's rows stop mid-period,
  report a data gap instead of a demand explanation.
* A week-over-week move smaller than the typical day-to-day noise of the
  series (~daily volumes vary by several percent) is **not significant** —
  the right answer is that there is no provable single driver.
