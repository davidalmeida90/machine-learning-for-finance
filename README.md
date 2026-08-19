# Machine learning for equity and fixed income

Five models on real market and macro data, in one notebook that runs top to bottom.

| | model | data | what it is asked |
|---|---|---|---|
| 1 | Decision tree | S&P 500 panel | beat the peer median over 5 days |
| 2 | Random forest | S&P 500 panel | same |
| 3 | XGBoost | S&P 500 panel | same |
| 4 | PCA | Treasury curve, FRED | level, slope and curvature |
| 5 | Lasso | 50 factor candidates | forward 21 day return |

Write up, with the charts and the reasoning:
<https://davidariasfinance.com/scripts/machine-learning-for-finance/>

## Running it

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib requests lxml pyarrow
jupyter notebook machine_learning_finance.ipynb
```

No API key and no paid data. Prices come from Yahoo Finance, rates and macro from the public
FRED CSV endpoint, and index membership from the Wikipedia API, all fetched at run time.

Budget ten to fifteen minutes on the first run, almost all of it downloading. The notebook
writes `membership.json` and `precos.parquet` next to itself and reads them back by relative
path, so keep them together and a rerun takes about a minute and a half.

## Survivorship bias

Taking today's S&P 500 list and running it back to 2007 drops every company that went bankrupt
or was acquired, so the sample is made only of survivors and the model looks cleverer than it is.

Wikipedia keeps a full revision history of its S&P 500 page, so the membership as it stood on any
past date can be read back from the revision live at that date. Snapshots every six months give
959 tickers that were members at some point, against roughly 500 at any one moment, and each
company contributes rows only on the dates it was genuinely a member.

Still incomplete: Yahoo serves usable history for 640 of the 959, and the missing names skew
towards companies that failed. Read it as a large improvement rather than a fix.

## Results, out of sample

Every model is measured against two yardsticks: chance, and the single best feature used on
its own. The second is the one that matters, because if one sorted column does the same job
then the algorithm is decoration.

| target | model | AUC | vs 0.50 | vs best feature |
|---|---|---|---|---|
| 5 day return | best feature, `mom_252_rk` | 0.5126 | +0.0126 | |
| 5 day return | Decision tree | 0.5070 | +0.0070 | -0.0056 |
| 5 day return | Random forest | 0.5120 | +0.0120 | -0.0006 |
| 5 day return | XGBoost | 0.5111 | +0.0111 | -0.0015 |
| 21 day volatility | best feature, `vol_126_rk` | 0.8299 | +0.3299 | |
| 21 day volatility | Decision tree | 0.8275 | +0.3275 | -0.0024 |
| 21 day volatility | Random forest | 0.8344 | +0.3344 | **+0.0045** |
| 21 day volatility | XGBoost | 0.8384 | +0.3384 | **+0.0085** |

Four of the six lose to a single column. On return the signal is momentum, and sorting stocks
by their one year return beats all three models. On volatility there is far more signal, it is
persistence, and the ensembles finally add something, under a point of AUC.

PCA recovers level, slope and curvature from daily changes in the curve, 95.6% of the variation
in three components. Lasso cuts 50 candidates to 8, out of sample R2 0.091 against -0.008 for
predicting the mean.

An AUC of 0.512 is a faint edge, and that is the point. It is roughly what genuine cross
sectional equity signals look like before costs. Anything in the sixties on a setup like this
would mean a leak rather than a discovery.

## Limitations

No transaction costs, spread, slippage or borrow anywhere. One train and test split rather than a
walk forward. Hyperparameters chosen by hand while test results were visible. One market and one
era. Accuracy and AUC are not money.

Worked examples, not production code.

## Licence

MIT
