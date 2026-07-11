# `sales.csv` — data dictionary

Export of a Q3 e-commerce campaign test. Pulled from the orders table and opened in Excel before
being saved, which is relevant (see below).

| Column | Notes |
|---|---|
| `order_id` | Unique per order. |
| `channel` | `email` or `paid_search`. **Not randomised** — the email campaign was deliberately targeted at existing high-value customers. |
| `customer_segment` | `new`, `returning`, `loyal`. |
| `customer_age` | Years. **`-1` means the customer did not supply an age.** It is not an age. |
| `status` | `completed` or `refunded`. |
| `revenue` | Order value in USD. **Exported as text with thousands separators** (`"1,234.56"`), because the file went through Excel. It will not sum or average until it is parsed. |
| `converted` | 1 if the order completed checkout. |

---

## ⚠️ Things that will silently corrupt an analysis

**1. `revenue` is a string, not a number.** `"1,234.56"`. Averaging it either raises, or — worse —
silently concatenates or sorts lexicographically. Strip the commas and cast to float first.

**2. Six internal QA orders were left in the export**, each with `revenue = 999999.99`. They are
not real orders. Six rows out of 900 are enough to move the mean revenue by two orders of
magnitude.

**3. Refunded orders are still in the file.** `status == 'refunded'` means the money went back.
Counting them as revenue overstates it.

**4. Channel was not randomly assigned.** The email campaign was aimed at loyal customers, who buy
more regardless of what you send them. Any comparison of `converted` between channels that ignores
`customer_segment` is not measuring the campaign — it is measuring who was targeted.
