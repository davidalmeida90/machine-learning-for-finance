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

| model | accuracy | AUC | z |
|---|---|---|---|
| Decision tree | 0.5083 | 0.5070 | +8.0 |
| Random forest | 0.5068 | 0.5120 | +13.7 |
| XGBoost | 0.5076 | 0.5111 | +12.6 |
| Base rate | 0.5009 | 0.5000 | 0.0 |

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
