# Machine learning for equity and fixed income

A replication of a published statistical arbitrage paper, plus two unsupervised and sparse
methods on rates data, in one notebook that runs top to bottom.

| | model | data | what it is asked |
|---|---|---|---|
| 1 | Decision tree | S&P 500 panel, point in time | beat the peer median tomorrow |
| 2 | Random forest | same | same |
| 3 | XGBoost | same | same |
| 4 | PCA | Treasury curve, FRED | level, slope and curvature |
| 5 | Lasso | 50 factor candidates | forward 21 day return |

Write up, with the charts and the reasoning:
<https://davidariasfinance.com/scripts/machine-learning-for-finance/>

## The paper

Steps 4 to 7 rebuild **Krauss, Do and Huck (2017)**, *Deep neural networks, gradient-boosted
trees, random forests: Statistical arbitrage on the S&P 500*, European Journal of Operational
Research 259(2). Three choices carry their method, and all three are reproduced here:

1. **Sliding window.** Train on 750 trading days, trade the next 250, step forward 250 and refit.
2. **31 features, nothing else.** Cumulative returns over 1 to 20 days, then 40, 60 up to 240,
   each standardised across the cross section of its own day.
3. **Trade the extremes only.** Long the 10 highest probabilities of the day, short the 10
   lowest, discard the middle.

Their headline, before costs, on 1992 to 2015: 0.43% a day for the random forest at t = 14.93.
Neural networks are left out here, so three of their four models appear.

## What is in here

Two ways through the same work, and they produce the same numbers.

| file | what it is |
|---|---|
| `machine_learning_finance.ipynb` | the whole thing top to bottom, 36 cells, every chart inline |
| `dados.py` | builds the two caches: index membership from Wikipedia, prices from Yahoo |
| `krauss.py` | the replication engine, sliding window and long short portfolio |
| `krauss_auc.py` | ROC curves in and out of sample, plus the look ahead and survivorship tests |
| `pca_lasso.py` | PCA on the Treasury curve and Lasso on the macro panel, fully standalone |
| `figuras.py` | the charts the write up uses |
| `*.json` | results, so you can compare against a run of your own without repeating it |

## Running it

```bash
pip install yfinance pandas numpy scikit-learn xgboost matplotlib requests lxml pyarrow
```

Notebook path, nothing else needed:

```bash
jupyter notebook machine_learning_finance.ipynb
```

Script path, and order matters because the engines read caches rather than build them:

```bash
python dados.py                  # 39 Wikipedia revisions, 640 tickers, writes both caches
python krauss.py --desde 2007-01-01 --ate 2015-12-31    # the five window replication
python krauss_auc.py             # ROC curves and the integrity tests
python figuras.py                # the charts

python dados.py --ate 2026-08-19 # wider panel, then:
python krauss.py                 # all fifteen windows through to 2025

python pca_lasso.py              # independent of the two above, downloads its own data
```

No API key and no paid data. Prices come from Yahoo Finance, rates and macro from the public
FRED CSV endpoint, and index membership from the Wikipedia API, all fetched at run time.

Budget fifteen to twenty minutes on the first run, most of it downloading. Fitting the fifteen
models on panels of about 217,000 rows takes roughly four minutes. `membership.json` and
`precos.parquet` are written next to the scripts and read back by relative path, so keep them
together and a rerun skips the download.

Comments inside the `.py` engines are in Portuguese. Notebook and write up are in English, and
they walk the same path, so nothing is only explained in one language.

## Survivorship bias

Taking today's S&P 500 list and running it back to 2007 drops every company that went bankrupt
or was acquired, so the sample is made only of survivors and the model looks cleverer than it is.

Wikipedia keeps a full revision history of its S&P 500 page, so the membership as it stood on any
past date can be read back from the revision live at that date. Snapshots every six months give
959 tickers that were members at some point, against roughly 500 at any one moment, and each
company contributes rows only on the dates it was genuinely a member.

Still incomplete: Yahoo serves usable history for 640 of the 959, and the missing names skew
towards companies that failed. Read it as a large improvement rather than a fix.

Step 7 checks all of this as executable tests rather than claims: that the target holds
tomorrow's return and not today's, that only same day members enter the panel, and that
companies which left the index are still present.

## Results, out of sample

Five windows, each trained on 750 days and traded on the 250 that follow. Return per day of the
long short book, k = 10, before costs.

| window | trains on | trades | tree | forest | boosting |
|---|---|---|---|---|---|
| 1 | 2007-12-14 to 2010-12-06 | 2010-12-07 to 2011-12-01 | +0.081% | +0.265% | +0.295% |
| 2 | 2008-12-11 to 2011-12-01 | 2011-12-02 to 2012-11-30 | +0.019% | +0.115% | +0.048% |
| 3 | 2009-12-09 to 2012-11-30 | 2012-12-03 to 2013-11-27 | +0.046% | +0.124% | +0.002% |
| 4 | 2010-12-07 to 2013-11-27 | 2013-11-29 to 2014-11-25 | +0.120% | +0.116% | +0.081% |
| 5 | 2011-12-02 to 2014-11-25 | 2014-11-26 to 2015-11-23 | +0.010% | +0.182% | +0.188% |

Pooled over those 1,250 trading days:

| model | return/day | annualised | t-stat | windows positive |
|---|---|---|---|---|
| Decision tree | 0.055% | 14.9% | 2.02 | 5 of 5 |
| Random forest | **0.160%** | **49.7%** | **4.80** | 5 of 5 |
| XGBoost | 0.123% | 36.3% | 3.36 | 5 of 5 |

Roughly 37% of the paper's headline figure, same direction, same ordering of models. Our overlap
with their sample is 2010 to 2015, the tail end where they already report profits declining, and
our universe is smaller than theirs.

### AUC 0.51 and 0.16% a day, at the same time

Averaged over the five windows:

| model | train AUC | test AUC | gap |
|---|---|---|---|
| Decision tree | 0.5366 | 0.5048 | 0.0318 |
| Random forest | 0.6315 | 0.5071 | 0.1244 |
| XGBoost | 0.5485 | 0.5076 | 0.0409 |

Both numbers are true. AUC scores the whole ranking, most of which is noise, while the book only
ever holds the twenty names at the two ends of it. A model can be nearly worthless as a
classifier and still useful as a sorter of extremes, and that gap is the single most useful
thing in this repo.

### Running the same code forward to 2025

Nothing changes except the dates. Fifteen consecutive windows, grouped in blocks of five:

| traded | tree | forest | boosting |
|---|---|---|---|
| 2010 to 2015 | +0.055% | **+0.160%** | +0.123% |
| 2015 to 2020 | -0.019% | +0.098% | +0.081% |
| 2020 to 2025 | +0.028% | +0.031% | -0.012% |

Forest returns fall by roughly half, then by half again. Pooled across all fifteen windows the
forest still reads 0.097% a day at t = 3.97, carried almost entirely by the first third. Either
the anomaly was arbitraged away as these methods became standard, which is the direction the
paper itself documents after 2001, or it survives and needs more than 31 momentum features to
reach.

### PCA and Lasso

PCA recovers level, slope and curvature from daily changes in the Treasury curve, 95.6% of the
variation in three components out of ten maturities. Lasso cuts 50 candidates to 8, out of
sample R2 0.091 against -0.008 for predicting the mean.

## Limitations

No transaction costs, spread, slippage or borrow anywhere; the paper charged 0.05% per half turn
and still kept t = 7.91, so read every figure here as gross. Testing ends November 2015 by
design, to overlap the paper. 640 tickers rather than the full index, so roughly 345 names on a
given day. 60 trees in the forest against the thousand a paper would use. Consecutive training
sets overlap by two years, so five windows are not five independent observations. One market and
one era.

Worked examples, not production code.

## Licence

MIT
