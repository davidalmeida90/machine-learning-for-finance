"""Naive baselines and seed spread for the deep learning write up.

Answers the question the first version left open: did the networks add anything, or
is the target simply predictable? Three checks, each on the identical rows the
networks were scored on.

  1. Volatility target of the LSTM and transformer: persistence, EWMA (RiskMetrics)
     and HAR-RV, same 40 day windows, same date split.
  2. Same two networks trained five times each from different seeds, so the gap
     between them can be read against its own noise.
  3. CNN magnitude target: trailing 20 day realised vol as the baseline, on the same
     out of sample (date, ticker) rows the CNN was scored on.

Writes baselines.json at the repo root and two charts into figures/. Runs with py -3.11 for torch.

  py -3.11 baselines.py
"""
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
INK, GRID_L, GRID_D = "#12151c", "#e6e8ee", "#2a3040"
RES = {}


def fig(dark: bool, w=7.2, h=5.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=160)
    bg = "#0d1117" if dark else "#ffffff"
    f.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.grid(color=GRID_D if dark else GRID_L, lw=0.9); ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID_D if dark else GRID_L)
    ax.tick_params(colors="#8b93a7" if dark else "#6b7280", labelsize=9, length=0)
    return f, ax, ("#e6e9f0" if dark else INK)


def save(f, name):
    f.savefig(OUT / name, facecolor=f.get_facecolor(), bbox_inches="tight", pad_inches=0.16)
    plt.close(f); print("  saved", name)


def score(pred, act):
    return (float(np.corrcoef(pred, act)[0, 1]),
            float(np.sqrt(((pred - act) ** 2).mean())) * 100)


# ------------------------------------------------ 1. the volatility target, rebuilt
# Identical construction to the training script: 20 years of SPY, 40 day windows of
# annualised absolute returns, forward 10 day realised vol, split at 75% in time.
print("[1/3] volatility target, same windows as the LSTM and transformer")
px = yf.Ticker("SPY").history(period="20y")["Close"].dropna()
r = np.log(px / px.shift(1)).dropna()
LOOK, FWD = 40, 10
fwd_vol = r[::-1].rolling(FWD).std()[::-1].shift(-1) * np.sqrt(252)
feat = np.abs(r.values) * np.sqrt(252)

# Baseline inputs, all known at the close of day i, nothing from the future.
r2 = r ** 2
rv1 = np.sqrt(r2) * np.sqrt(252)                                # today's absolute move
rv5 = np.sqrt(r2.rolling(5).mean()) * np.sqrt(252)              # weekly realised vol
rv22 = np.sqrt(r2.rolling(22).mean()) * np.sqrt(252)            # monthly realised vol
rv10 = np.sqrt(r2.rolling(FWD).mean()) * np.sqrt(252)           # persistence, last 10 days
ewma = np.sqrt(r2.ewm(alpha=1 - 0.94, adjust=False).mean()) * np.sqrt(252)

Xs, ys, ds, B = [], [], [], []
for i in range(LOOK, len(r) - FWD - 1):
    v = fwd_vol.iloc[i]
    if np.isfinite(v) and np.isfinite(rv22.iloc[i]):
        Xs.append(feat[i - LOOK:i]); ys.append(v); ds.append(r.index[i])
        B.append((rv10.iloc[i], ewma.iloc[i], rv1.iloc[i], rv5.iloc[i], rv22.iloc[i]))
Xs = np.array(Xs, dtype=np.float32)[:, :, None]
ys = np.array(ys, dtype=np.float32)[:, None]
B = np.array(B, dtype=np.float64)
cut = int(len(Xs) * 0.75)
act = ys[cut:].ravel()
print(f"      {len(Xs):,} windows, out of sample from {ds[cut].date()}")

# persistence and EWMA need no fitting at all
c, e = score(B[cut:, 0], act); RES["persist_corr"], RES["persist_rmse"] = round(c, 3), round(e, 2)
c, e = score(B[cut:, 1], act); RES["ewma_corr"], RES["ewma_rmse"] = round(c, 3), round(e, 2)

# HAR-RV, Corsi 2009: forward vol on daily, weekly and monthly realised vol. Four
# numbers fitted by least squares on the training block only.
A = np.c_[np.ones(len(B)), B[:, 2:5]]
beta, *_ = np.linalg.lstsq(A[:cut], ys[:cut].ravel(), rcond=None)
har = A[cut:] @ beta
c, e = score(har, act); RES["har_corr"], RES["har_rmse"] = round(c, 3), round(e, 2)
RES["har_beta"] = [round(float(b), 3) for b in beta]
print(f"      persistence corr {RES['persist_corr']}, EWMA {RES['ewma_corr']}, HAR-RV {RES['har_corr']}")

# ------------------------------------------------------- 2. five seeds per network
# Same architectures, optimiser, epochs and split as the published run. Only the seed
# moves, so whatever spread appears is the noise the single published number sits in.
print("[2/3] five seeds each, LSTM and transformer")
Xtr, Xte = torch.tensor(Xs[:cut], device=DEV), torch.tensor(Xs[cut:], device=DEV)
ytr = torch.tensor(ys[:cut], device=DEV)


class LSTMVol(nn.Module):
    def __init__(s, h=48):
        super().__init__(); s.l = nn.LSTM(1, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1])


class Attn(nn.Module):
    def __init__(s, d=32):
        super().__init__()
        s.p = nn.Linear(1, d); s.pos = nn.Parameter(torch.randn(1, LOOK, d) * 0.02)
        s.a = nn.MultiheadAttention(d, 4, batch_first=True)
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, 1)
    def forward(s, x):
        h = s.p(x) + s.pos
        z, _ = s.a(h, h, h, need_weights=False)
        return s.o(s.n(h + z)[:, -1])


SEEDS = [0, 1, 2, 3, 4]
runs = {"lstm": [], "transformer": []}
for nome, Cls in (("lstm", LSTMVol), ("transformer", Attn)):
    for sd in SEEDS:
        torch.manual_seed(sd); np.random.seed(sd)
        m = Cls().to(DEV); opt = torch.optim.Adam(m.parameters(), lr=2e-3)
        for ep in range(400):
            opt.zero_grad(); ((m(Xtr) - ytr) ** 2).mean().backward(); opt.step()
        with torch.no_grad(): p = m(Xte).cpu().numpy().ravel()
        cc, ee = score(p, act)
        runs[nome].append({"seed": sd, "corr": round(cc, 3), "rmse": round(ee, 2)})
        print(f"      {nome:12} seed {sd}  corr {cc:.3f}  RMSE {ee:.2f}")
for nome in runs:
    cs = [x["corr"] for x in runs[nome]]
    RES[f"{nome}_seeds"] = runs[nome]
    RES[f"{nome}_corr_mean"] = round(float(np.mean(cs)), 3)
    RES[f"{nome}_corr_min"] = round(float(np.min(cs)), 3)
    RES[f"{nome}_corr_max"] = round(float(np.max(cs)), 3)
gap = [a["corr"] - b["corr"] for a, b in zip(runs["lstm"], runs["transformer"])]
RES["lstm_minus_transformer_mean"] = round(float(np.mean(gap)), 3)
RES["lstm_minus_transformer_range"] = [round(float(min(gap)), 3), round(float(max(gap)), 3)]
print(f"      LSTM minus transformer, per seed: {[round(g, 3) for g in gap]}")

# light panel: every method on one axis, seeds as a range
f, ax, tc = fig(False, 7.2, 4.6)
nomes = ["persistence\n(last 10d)", "EWMA\n(0.94)", "HAR-RV", "LSTM\n(5 seeds)", "transformer\n(5 seeds)"]
vals = [RES["persist_corr"], RES["ewma_corr"], RES["har_corr"],
        RES["lstm_corr_mean"], RES["transformer_corr_mean"]]
cores = ["#9aa0ae", "#9aa0ae", "#6b7280", "#1f7a4d", "#C2571A"]
ax.bar(range(5), vals, color=cores, width=0.66)
for k, nome in ((3, "lstm"), (4, "transformer")):
    lo, hi = RES[f"{nome}_corr_min"], RES[f"{nome}_corr_max"]
    ax.plot([k, k], [lo, hi], color=INK, lw=1.6)
    ax.plot([k - 0.12, k + 0.12], [lo, lo], color=INK, lw=1.6)
    ax.plot([k - 0.12, k + 0.12], [hi, hi], color=INK, lw=1.6)
for k, v in enumerate(vals):
    # Nas duas barras com intervalo de sementes o rotulo sobe para cima do bigode,
    # senao o numero cai em cima da linha e vira borrao.
    topo = RES[f"{ {3: 'lstm', 4: 'transformer'}[k] }_corr_max"] if k in (3, 4) else v
    ax.text(k, topo + 0.014, f"{v:.2f}", ha="center", color=tc, fontsize=10)
ax.set_xticks(range(5)); ax.set_xticklabels(nomes, fontsize=9)
ax.set_ylabel("out of sample correlation with realised vol", color=tc)
ax.set_ylim(0, max(vals) + 0.09)
ax.set_title("Same target, same windows, same split", color=tc, fontsize=12, loc="left")
save(f, "base_vol.png")

# ------------------------------------------------ 3. CNN magnitude, trailing vol
# Rows come from the CNN's own out of sample evaluation, saved by cnn_autoencoder.py,
# so the baseline is scored on exactly the (date, ticker) pairs the network saw.
print("[3/3] CNN magnitude target against trailing realised vol")
# allow_pickle is needed for the ticker array, which is an object dtype. The file is
# written by cnn_autoencoder.py in this same repository into figures/, never fetched from anywhere.
z = np.load(OUT / "cnn_eval.npz", allow_pickle=True)
pv, yv = z["pv"], z["yv"]
dias = pd.DatetimeIndex(z["d"]); quem = z["quem"]
tick = sorted(set(quem.tolist()))
closes = yf.download(tick, start="2012-01-01", auto_adjust=True, progress=False)["Close"]
tv = closes.pct_change().rolling(20).std() * np.sqrt(252) * 100     # trailing 20 day vol
tv.index = pd.DatetimeIndex(tv.index).tz_localize(None)
dias = dias.tz_localize(None) if dias.tz is not None else dias
base = np.array([tv.at[d, t] if (d in tv.index and t in tv.columns) else np.nan
                 for d, t in zip(dias, quem)])
ok = np.isfinite(base)
print(f"      {ok.sum():,} of {len(base):,} rows matched to a trailing vol")
c_cnn = float(np.corrcoef(pv[ok], yv[ok])[0, 1])
c_base = float(np.corrcoef(base[ok], yv[ok])[0, 1])
RES["cnn_corr_same_rows"] = round(c_cnn, 3)
RES["trailing_corr"] = round(c_base, 3)


def ladder(sort_by, target):
    q = pd.qcut(pd.Series(sort_by).rank(method="first"), 5, labels=False)
    return pd.Series(target).groupby(q).mean().values


lad_cnn, lad_base = ladder(pv[ok], yv[ok]), ladder(base[ok], yv[ok])
RES["cnn_ladder"] = [round(float(v), 1) for v in lad_cnn]
RES["trailing_ladder"] = [round(float(v), 1) for v in lad_base]
print(f"      CNN corr {c_cnn:.3f}, trailing vol corr {c_base:.3f}")
print(f"      ladders  CNN {RES['cnn_ladder']}  trailing {RES['trailing_ladder']}")

# dark panel: the two ladders side by side
f, ax, tc = fig(True, 7.2, 5.2)
x = np.arange(5); w = 0.38
ax.bar(x - w / 2, lad_base, width=w, color="#8b93a7", label=f"sorted by trailing 20d vol, corr {c_base:.2f}")
ax.bar(x + w / 2, lad_cnn, width=w, color="#1E56B8", label=f"sorted by the CNN, corr {c_cnn:.2f}")
for k in range(5):
    ax.text(x[k] - w / 2, lad_base[k] + 0.5, f"{lad_base[k]:.0f}", ha="center", color=tc, fontsize=9)
    ax.text(x[k] + w / 2, lad_cnn[k] + 0.5, f"{lad_cnn[k]:.0f}", ha="center", color=tc, fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(["1\nquietest", "2", "3", "4", "5\nwildest"])
ax.set_ylabel("realised vol over the next 20 days, %", color=tc)
ax.set_title("Same rows, two ways of sorting them", color=tc, fontsize=12, loc="left")
lg = ax.legend(frameon=False, fontsize=9, loc="upper left")
for t in lg.get_texts(): t.set_color(tc)
ax.margins(y=0.16)
save(f, "base_cnn.png")

(Path(__file__).parent / "baselines.json").write_text(json.dumps(RES, indent=1))
print("\nBASELINES:", json.dumps({k: v for k, v in RES.items() if "seeds" not in k}, indent=1))
