# FUTURE_ML_01: Sales & Demand Forecasting for Businesses

Future Interns · Machine Learning Track · Task 1

A 90-day sales forecasting model for a retail store, with an honest backtest, error analysis, and
business-ready charts and recommendations.

## The business problem
A store manager needs to know how much will sell next quarter, which days will be busy, and how much
stock to hold. This project turns four years of daily sales into a forecast that answers those questions.

## What is in this repo
| File | Purpose |
|---|---|
| `Sales_Demand_Forecasting.ipynb` | Full workflow: cleaning, EDA, features, models, backtest, forecast, insights |
| `images/` | Charts used in the notebook (history, seasonality, backtest, errors, forecasts) |
| `outputs/forecast_next_90_days.csv` | Daily forecast with a 90% range |
| `outputs/model_comparison.csv` | Backtest scores for all five models |
| `requirements.txt` | Python dependencies |

## Approach
1. **Data cleaning**: duplicates, missing calendar days, blank values and data-entry outliers are found, counted and fixed.
2. **Time-based features**: trend, weekday, yearly seasonality (Fourier terms), holiday flags (Black Friday, Christmas, New Year) and a promotion flag.
3. **Models compared**: seasonal naive, recent weekday average, Ridge regression, gradient boosting, and a Ridge + boosting hybrid. Sales are modelled on a log scale because retail effects are multiplicative.
4. **Rolling-origin backtest**: four consecutive 90-day windows over the last year. The model is trained only on data before each window, so there is no look-ahead leakage.
5. **Error analysis**: errors broken down by weekday and by holiday vs ordinary days.
6. **Forecast**: the best model is retrained on all history and projected 90 days ahead, with a 90% range built from its own backtest errors.

## Results (simulated data)
| Model | WAPE (%) |
|---|---|
| Ridge regression (best) | 6.3 |
| Hybrid: Ridge + boosting | 6.5 |
| Gradient boosting | 9.5 |
| Recent weekday average | 13.8 |
| Seasonal naive | 14.0 |

WAPE is total absolute error as a percentage of total sales. The best model cuts the error of the
"same day last year" baseline by about 55%.

**Forecast for 1 Jan - 31 Mar 2026:** about $228K in sales, roughly +10% on the same period last year.
The notebook also reports the busiest and quietest days, a weekly stock buffer, and a promotion what-if.

## Important note on the data
The notebook runs on a **simulated four-year daily sales history** (with deliberately messy records) so it works
with no downloads. Because the simulation is multiplicative, the log-linear Ridge model fits it especially well.
On real data the ranking of models may differ, so re-run the notebook and check the leaderboard.

## Using a real dataset
Set `DATA_PATH` (and, if auto-detection fails, `DATE_COL` / `SALES_COL`) in the configuration cell, for example with:
- Kaggle *Store Sales - Time Series Forecasting* (`train.csv`)
- Kaggle *Superstore* dataset

The loader aggregates transactions to daily sales and treats days with no orders as zero sales. Everything
downstream runs unchanged. Very lumpy datasets (few orders per day) forecast better at weekly level.

## Run it
```bash
pip install -r requirements.txt
jupyter notebook Sales_Demand_Forecasting.ipynb
```
