# Deep learning for finance

Five architectures, each pointed at the job it was invented for, on real market data pulled
from one free source. Two scripts run top to bottom and write every chart and every number
in this file.

| | architecture | data | what it is asked |
|---|---|---|---|
| 1 | Feed forward | live SPY option chain | fit the implied volatility surface |
| 2 | Recurrent, an LSTM | 20 years of SPY closes | realised vol over the next 10 days |
| 3 | Convolutional | 45 US large caps, 2013 to 2026 | read 20 days as a picture |
| 4 | Transformer | same target as the LSTM | same, with attention instead of memory |
| 5 | Autoencoder | daily cross section of 45 names | latent factors, and days that break |

Write up, with the charts and the reasoning:
<https://davidariasfinance.com/scripts/deep-learning-for-finance/>

## Results

Out of sample throughout, split by date except where noted.

| architecture | result | against |
|---|---|---|
| Feed forward | 0.251 vol points held out error | 998 live quotes, spot 767.32 |
| LSTM | correlation 0.457, RMSE 7.74 vol points | 4,978 windows from 7 September 2021 |
| CNN, magnitude | correlation 0.225, ladder 22.9% to 30.6% | 153,000 images, 45 names |
| CNN, direction | unstable, see below | four runs of identical code |
| Transformer | correlation 0.446, RMSE 7.81 vol points | same windows as the LSTM |
| Autoencoder | 35.0% of variance through 3 units | 3,440 days, worst rebuild 9 April 2025 |

Every figure above is written to `results.json` by the run that produced it, and charts land
in `figures/`, overwriting the ones shipped here. A fresh run can be compared against this one
without reading the charts.

### One result that does not hold

Asking the CNN which way a name goes next gives quintile spreads of 0.18, 0.19, 0.01 and
0.16 percentage points across four runs of the same code. Forty five names per date is a tiny
cross section to sort into buckets, so which day fell on which side of the split moves the
answer more than the model does. Direction is reported here as a failure. Asking the same
images how much a name moves is stable, and that is the result in the table.

Classification accuracy on that direction target reads 0.553, which also means nothing:
twenty day windows on US large caps are up more often than down, so a model that always
answers "up" scores about the same.

## Baselines, on the same rows

`baselines.py` scores every network against something that costs nothing, on the identical
windows, rows and date split the networks were scored on.

| target | baseline | corr | network | corr |
|---|---|---|---|---|
| 10 day forward vol | persistence, last 10 days | 0.491 | LSTM, mean of 5 seeds | 0.461 |
| 10 day forward vol | EWMA, decay 0.94 | 0.491 | transformer, mean of 5 seeds | 0.434 |
| 10 day forward vol | HAR-RV, Corsi 2009 | **0.528** | | |
| 20 day forward vol per name | trailing 20 day realised vol | **0.513** | CNN on the normalised image | 0.225 |

HAR-RV, four coefficients by least squares, beats the best seed of either network. Trailing
realised vol, one number per row, carries more than twice the signal the CNN recovers from a
scale free picture of the same twenty days. Both networks were retrained five times from
different seeds: LSTM 0.445, 0.464, 0.464, 0.488, 0.446; transformer 0.445, 0.451, 0.449, 0.384, 0.440. Seed by seed the gap between them is inside 0.015
on three of five, so the two do not separate on a sequence this short.

Reading is the one that matters for a desk. Volatility is forecastable, and on forty daily
numbers a linear model with the right three features forecasts it better than either network.
Deep learning earns its cost when the input holds structure a linear model cannot, a cross
section, an order book, text. Here it did not, and the sections above stand as demonstrations
of how each architecture reads its input rather than as forecasts to trade.

## Papers behind sections 3 and 5

Section 3 follows **Jiang, Kelly and Xiu (2023)**, *(Re-)Imag(in)ing Price Trends*, Journal of
Finance 78(6). Their method is reproduced in shape: render the window as a bar chart image,
three pixels per day, rescale prices inside each window so level disappears and only shape
survives, then let convolution find whatever pattern exists. Their universe is CRSP and the
label is direction; this runs 45 names and reports magnitude, for the reason above.

Section 5 follows the shape of **Gu, Kelly and Xiu (2021)**, *Autoencoder Asset Pricing
Models*, Journal of Econometrics 222(1). Bottleneck carries latent factors out of a panel with
nothing supervising it. Their loadings are a function of firm characteristics; this version is
the plain autoencoder on returns alone.

Sections 1, 2 and 4 are standard constructions rather than replications.

## What is in here

| file | what it is |
|---|---|
| `feedforward_lstm_transformer.py` | sections 1, 2 and 4, plus six charts |
| `cnn_autoencoder.py` | sections 3 and 5, plus four charts |
| `baselines.py` | persistence, EWMA, HAR-RV, trailing vol, and five seeds per network |
| `baselines.json` | every number in the baselines table |
| `results.json` | every number the write up quotes, written by the runs |
| `figures/` | where both scripts write their charts, and the architecture drawings |

## Running it

```bash
pip install torch yfinance pandas numpy matplotlib
```

```bash
python feedforward_lstm_transformer.py   # live option chain, then 20 years of SPY
python cnn_autoencoder.py                # 45 names, 153,000 images, then the cross section
python baselines.py                      # after both, reads figures/cnn_eval.npz
```

Both scripts pull from `yfinance` and cache nothing, so numbers move a little with the data.
Training uses CUDA when it is available and falls back to CPU on its own. Section 1 quotes a
live option chain, so it changes every day the market opens, and outside US market hours
Yahoo returns the chain with zero bids: the script then skips section 1 with a message and
runs sections 2 and 3, which only need daily closes. Run it during US hours for the surface.

Runtime on a laptop GPU is about a minute for the first script and about seven minutes for
the second, most of it spent rendering 153,000 images rather than training.

## What this does not do

No transaction costs anywhere. No commission, spread, slippage or borrow, and the quintile
ladder in section 3 is a sort rather than a strategy.

Forty five names is a sample, not a market. All of them are alive today, which is survivorship
bias by construction, and no small cap, no non US name and nothing delisted appears.

Volatility windows overlap. Ten day forward vol computed every day means consecutive labels
share nine of their ten days, so 4,978 windows are nowhere near 4,978 independent
observations and the uncertainty around 0.457 and 0.446 is wider than a textbook standard
error suggests.

Surface fit in section 1 is one snapshot, split at random rather than by date, which is
defensible for a static map and transfers to nothing else here.

Nothing forces the fitted surface to be arbitrage free, so the interpolant can imply a
negative density in the wings. Desk use needs that constraint added.

Headline numbers are single runs from a fixed seed. `baselines.py` retrains the LSTM and the
transformer five times each; feed forward, CNN and autoencoder remain single runs.

## Licence

MIT.
