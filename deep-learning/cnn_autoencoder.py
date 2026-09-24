"""Trains the two architectures added in the second round, CNN and autoencoder.

Same rules as `modelos_dl.py`: real data, free sources, temporal splits, and every
number that reaches a page comes back through `facts.json`. Runs with py -3.11,
since the default interpreter has a broken torch install.

  py -3.11 modelos_dl2.py

CNN follows Jiang, Kelly and Xiu, (Re-)Imag(in)ing Price Trends, JF 2023: render the
last twenty days as a picture, three pixels per day, and let a convolutional net read
the picture instead of the numbers. Autoencoder follows the shape of Gu, Kelly and Xiu
2021: squeeze a cross section down to a few latent units and see what survives.
"""
import json, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Charts land in figures/, which is the folder shipped with this repository, so a
# fresh run reproduces exactly what you see here.
OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0)
np.random.seed(0)
INK, GRID_L, GRID_D = "#12151c", "#e6e8ee", "#2a3040"


def fig(dark: bool, w=7.2, h=5.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=160)
    bg = "#0d1117" if dark else "#ffffff"
    f.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    ax.grid(color=GRID_D if dark else GRID_L, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID_D if dark else GRID_L)
    ax.tick_params(colors="#8b93a7" if dark else "#6b7280", labelsize=9, length=0)
    return f, ax, ("#e6e9f0" if dark else INK)


def save(f, name):
    f.savefig(OUT / name, facecolor=f.get_facecolor(), bbox_inches="tight",
              pad_inches=0.16)
    plt.close(f)
    print("  saved", name)


CAMINHO = Path(__file__).parent / "results.json"
FACTS = json.loads(CAMINHO.read_text()) if CAMINHO.exists() else {}

# Universe. Liquid, long history, and spread across sectors so the cross section the
# autoencoder squeezes is not five flavours of the same tech trade.
TICKERS = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO",
           "JPM", "BAC", "GS", "WFC", "C", "AXP", "BLK",
           "XOM", "CVX", "COP", "SLB", "PSX",
           "JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY",
           "PG", "KO", "PEP", "WMT", "COST", "MCD",
           "CAT", "HON", "GE", "BA", "UPS", "LMT",
           "NEE", "DUK", "SO", "AMT", "SPG", "LIN", "NEM"]

SETOR = {}
for nomes, setor in [
    (["AAPL", "MSFT", "NVDA", "AVGO", "GOOGL", "META"], "tech"),
    (["AMZN", "TSLA", "MCD"], "cons disc"),
    (["JPM", "BAC", "GS", "WFC", "C", "AXP", "BLK"], "banks"),
    (["XOM", "CVX", "COP", "SLB", "PSX"], "energy"),
    (["JNJ", "UNH", "PFE", "MRK", "ABBV", "LLY"], "health"),
    (["PG", "KO", "PEP", "WMT", "COST"], "staples"),
    (["CAT", "HON", "GE", "BA", "UPS", "LMT"], "industrial"),
    (["NEE", "DUK", "SO"], "utilities"),
    (["AMT", "SPG"], "real estate"),
    (["LIN", "NEM"], "materials"),
]:
    for t in nomes:
        SETOR[t] = setor

print(f"downloading {len(TICKERS)} names")
px = yf.download(TICKERS, start="2012-01-01", auto_adjust=True, progress=False)
close = px["Close"].dropna(axis=1, thresh=int(len(px) * 0.9)).dropna()
high, low, vol = px["High"][close.columns], px["Low"][close.columns], px["Volume"][close.columns]
high, low, vol = high.loc[close.index], low.loc[close.index], vol.loc[close.index]
print(f"  {close.shape[1]} names, {len(close)} days, {close.index[0].date()} to {close.index[-1].date()}")

# ------------------------------------------------------------------ 4. convolutional
# Twenty trading days become a 32 by 60 picture: three columns per day, the middle one
# the high to low bar, a tick left for the open and a tick right for the close, and a
# volume strip along the bottom. Prices are scaled INSIDE each window, so the net sees
# shape and never the level, which is the whole point of the paper.
print("[4/5] CNN on price chart images")
JAN, PIX, ALTP, ALTV = 20, 3, 26, 6
ALT = ALTP + ALTV


def desenha(o, h, l, c, v):
    img = np.zeros((ALT, JAN * PIX), dtype=np.float32)
    lo, hi = l.min(), h.max()
    if not np.isfinite(lo) or hi <= lo:
        return None
    esc = lambda p: int(np.clip((p - lo) / (hi - lo) * (ALTP - 1), 0, ALTP - 1))
    vmax = v.max()
    for d in range(JAN):
        x = d * PIX
        img[esc(o[d]), x] = 1.0
        img[esc(l[d]):esc(h[d]) + 1, x + 1] = 1.0
        img[esc(c[d]), x + 2] = 1.0
        if vmax > 0:
            a = int(np.clip(v[d] / vmax * ALTV, 0, ALTV))
            if a:
                img[ALTP:ALTP + a, x + 1] = 0.6
    return img[::-1]                      # row 0 at the top, like a chart on screen


HOR = 20                                  # label horizon, trading days
X, Y, YV, QUANDO, QUEM = [], [], [], [], []
op = close.shift(1)                       # yfinance open is noisy on splits, use prev close
for t in close.columns:
    c, h, l, v = close[t].values, high[t].values, low[t].values, vol[t].values
    o = op[t].values
    fwd = close[t].shift(-HOR).values / c - 1
    r = pd.Series(c).pct_change()
    # Realised vol over the SAME 20 days ahead, annualised. Second target.
    fv = r.shift(-HOR).rolling(HOR).std().shift(-1).values * np.sqrt(252) * 100
    for i in range(JAN, len(c) - HOR):
        if not np.isfinite(fwd[i]) or not np.isfinite(fv[i]) or fv[i] <= 0:
            continue
        im = desenha(o[i - JAN + 1:i + 1], h[i - JAN + 1:i + 1],
                     l[i - JAN + 1:i + 1], c[i - JAN + 1:i + 1], v[i - JAN + 1:i + 1])
        if im is None:
            continue
        X.append(im)
        Y.append(fwd[i])
        YV.append(fv[i])
        QUANDO.append(close.index[i])
        QUEM.append(t)
X = np.stack(X)[:, None]
Y = np.array(Y, dtype=np.float32)
YV = np.array(YV, dtype=np.float32)
QUANDO = pd.DatetimeIndex(QUANDO)
QUEM = np.array(QUEM)
ordem = np.argsort(QUANDO.values)
X, Y, YV, QUANDO, QUEM = X[ordem], Y[ordem], YV[ordem], QUANDO[ordem], QUEM[ordem]
corte = int(len(X) * 0.75)
print(f"      {len(X):,} images, out of sample from {QUANDO[corte].date()}")
MOSTRA = np.random.default_rng(1).choice(corte, 8, replace=False)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.c1 = nn.Conv2d(1, 16, (5, 3), padding=(2, 1))
        self.c2 = nn.Conv2d(16, 32, (5, 3), padding=(2, 1))
        self.bn1, self.bn2 = nn.BatchNorm2d(16), nn.BatchNorm2d(32)
        self.fc = nn.Linear(32 * (ALT // 4) * (JAN * PIX // 4), 1)
        self.drop = nn.Dropout(0.4)

    def forward(self, x):
        x = F.max_pool2d(F.leaky_relu(self.bn1(self.c1(x))), 2)
        x = F.max_pool2d(F.leaky_relu(self.bn2(self.c2(x))), 2)
        return self.fc(self.drop(x.flatten(1)))


cnn = CNN().to(DEV)
opt = torch.optim.Adam(cnn.parameters(), lr=8e-4, weight_decay=1e-4)
alvo = torch.tensor((Y > 0).astype(np.float32))[:, None]
xt = torch.tensor(X)
lote = 512
for ep in range(12):
    cnn.train()
    idx = np.random.permutation(corte)
    perda = 0.0
    for k in range(0, corte, lote):
        b = idx[k:k + lote]
        xb, yb = xt[b].to(DEV), alvo[b].to(DEV)
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(cnn(xb), yb)
        loss.backward()
        opt.step()
        perda += float(loss) * len(b)
    print(f"      epoch {ep + 1}  loss {perda / corte:.4f}")

cnn.eval()
with torch.no_grad():
    p = []
    for k in range(corte, len(X), 2048):
        p.append(torch.sigmoid(cnn(xt[k:k + 2048].to(DEV))).cpu().numpy().ravel())
p = np.concatenate(p)
yo = Y[corte:]
acc = float(((p > 0.5) == (yo > 0)).mean())
# Deciles are formed WITHIN each date, across names, and the return is measured
# against that day's own cross section. Pooling the probabilities across dates instead
# ranks a calm Tuesday against a crash, so the market's own drift decides the sort and
# the first version of this came back at zero for that reason alone.
NB = 5
ev = pd.DataFrame({"d": QUANDO[corte:], "p": p, "y": yo * 100})
ev["rel"] = ev.groupby("d")["y"].transform(lambda v: v - v.mean())
ev["dec"] = ev.groupby("d")["p"].transform(
    lambda v: pd.qcut(v.rank(method="first"), NB, labels=False, duplicates="drop"))
spread = float(ev.groupby("dec")["rel"].mean().iloc[-1]
               - ev.groupby("dec")["rel"].mean().iloc[0])
print(f"      direction: accuracy {acc:.3f}, quintile spread {spread:.2f}% over {HOR}d")
FACTS |= {"cnn_n": int(len(X)), "cnn_oos_from": str(QUANDO[corte].date()),
          "cnn_acc": round(acc, 3), "cnn_names": int(close.shape[1])}
FACTS.pop("cnn_spread", None)

# Direction is noise here, not a result. Re-running the same code with the same seed
# gave spreads of 0.19, then 0.18, then 0.01, because forty five names per date is a
# minuscule cross section and the answer depends on which day landed on which side of
# the split. Reported target below is the one the picture actually carries: HOW MUCH a
# name moves, rather than which way.
print("      same images, now asking how much it moves")
volnet = CNN().to(DEV)
opt = torch.optim.Adam(volnet.parameters(), lr=8e-4, weight_decay=1e-4)
lv = torch.tensor(np.log(YV))[:, None]
mv, sv = float(lv[:corte].mean()), float(lv[:corte].std())
lvn = (lv - mv) / sv
for ep in range(10):
    volnet.train()
    idx = np.random.permutation(corte)
    perda = 0.0
    for k in range(0, corte, lote):
        b = idx[k:k + lote]
        xb, yb = xt[b].to(DEV), lvn[b].to(DEV)
        opt.zero_grad()
        loss = F.mse_loss(volnet(xb), yb)
        loss.backward()
        opt.step()
        perda += float(loss) * len(b)
    print(f"      epoch {ep + 1}  loss {perda / corte:.4f}")

volnet.eval()
with torch.no_grad():
    pv = np.concatenate([volnet(xt[k:k + 2048].to(DEV)).cpu().numpy().ravel()
                         for k in range(corte, len(X), 2048)])
pv = np.exp(pv * sv + mv)
vo = YV[corte:]
volcorr = float(np.corrcoef(pv, vo)[0, 1])
q = pd.qcut(pd.Series(pv).rank(method="first"), NB, labels=False)
por_q = pd.Series(vo).groupby(q).mean()
print(f"      vol: out of sample corr {volcorr:.3f}, "
      f"quintile 1 {por_q.iloc[0]:.1f}% to quintile 5 {por_q.iloc[-1]:.1f}%")

FACTS |= {"cnn_volcorr": round(volcorr, 3), "cnn_vq1": round(float(por_q.iloc[0]), 1),
          "cnn_vq5": round(float(por_q.iloc[-1]), 1)}
np.savez_compressed(OUT / "cnn_eval.npz", p=p, pv=pv, y=yo, yv=vo,
                    d=QUANDO[corte:].values, quem=QUEM[corte:], amostra=X[MOSTRA],
                    amostra_vol=YV[MOSTRA])

# dark panel: what the net actually reads. Sample titles carry the REALISED VOL of the
# window, since that is the reported target. Labelled by direction they contradicted the
# text beside them, which calls direction a coin flip.
f, axs = plt.subplots(2, 4, figsize=(7.2, 3.4), dpi=160)
f.patch.set_facecolor("#0d1117")
for ax, i in zip(axs.ravel(), MOSTRA):
    ax.imshow(X[i, 0], cmap="magma", aspect="auto", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color("#2a3040")
    ax.set_title(f"{YV[i]:.0f}% vol next {HOR}d", color="#8b93a7", fontsize=9, pad=4)
f.suptitle("Twenty days as a picture, three pixels per day",
           color="#e6e9f0", fontsize=12, x=0.11, ha="left", y=1.02)
save(f, "cnn_imagens.png")

# light panel: the ladder the images DO produce
f, ax, tc = fig(False, 7.2, 5.2)
ax.bar(range(NB), por_q.values, color="#1E56B8", width=0.7)
for i, v in enumerate(por_q.values):
    ax.text(i, v + 0.6, f"{v:.0f}%", ha="center", color=tc, fontsize=11)
ax.set_xlabel("predicted volatility, quintile", color=tc)
ax.set_ylabel(f"realised vol over the next {HOR} days, %", color=tc)
ax.set_title("Out of sample, sorted by what the CNN saw", color=tc, fontsize=12, loc="left")
ax.set_xticks(range(NB))
ax.set_xticklabels(["1\nquietest", "2", "3", "4", "5\nwildest"])
ax.margins(y=0.14)
save(f, "cnn_decis.png")

# ---------------------------------------------------------------- 5. autoencoder
# Squeeze the daily cross section of returns through three units and rebuild it. What
# survives the squeeze is a latent factor, and what the rebuild misses on a given day
# is that day refusing to look like the others.
print("[5/5] autoencoder on the cross section of returns")
R = close.pct_change().dropna()
mu, sd = R.iloc[:int(len(R) * 0.75)].mean(), R.iloc[:int(len(R) * 0.75)].std()
Z = ((R - mu) / sd).clip(-8, 8).values.astype(np.float32)
n = Z.shape[1]
cut = int(len(Z) * 0.75)
print(f"      {n} names, {len(Z):,} days, out of sample from {R.index[cut].date()}")


class AE(nn.Module):
    def __init__(self, n, k=3):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(n, 24), nn.Tanh(), nn.Linear(24, k))
        self.dec = nn.Sequential(nn.Linear(k, 24), nn.Tanh(), nn.Linear(24, n))

    def forward(self, x):
        z = self.enc(x)
        return self.dec(z), z


ae = AE(n).to(DEV)
opt = torch.optim.Adam(ae.parameters(), lr=2e-3, weight_decay=1e-5)
zt = torch.tensor(Z)
for ep in range(400):
    ae.train()
    idx = np.random.permutation(cut)[:1024]
    xb = zt[idx].to(DEV)
    opt.zero_grad()
    rec, _ = ae(xb)
    loss = F.mse_loss(rec, xb)
    loss.backward()
    opt.step()
ae.eval()
with torch.no_grad():
    rec, lat = ae(zt.to(DEV))
    rec, lat = rec.cpu().numpy(), lat.cpu().numpy()
erro = ((Z - rec) ** 2).mean(1)
r2 = float(1 - erro[cut:].mean() / (Z[cut:] ** 2).mean())
pior = R.index[cut + int(np.argmax(erro[cut:]))]
print(f"      out of sample variance captured {r2:.3f}, worst day {pior.date()}")
FACTS |= {"ae_n": int(n), "ae_dias": int(len(Z)), "ae_oos_from": str(R.index[cut].date()),
          "ae_r2": round(r2, 3), "ae_pior": str(pior.date())}

# dark panel: what each of the three units ended up standing for
f, ax, tc = fig(True, 7.2, 5.2)
carga = np.array([[np.corrcoef(lat[:cut, j], Z[:cut, i])[0, 1] for i in range(n)]
                  for j in range(3)])
# Grouped by SECTOR rather than by name. Forty five one pixel columns say nothing, and
# the three units seen raw are close to the market with a sign flip. Each row is also
# centred on itself, otherwise unit 1, which IS the market, paints the whole panel and
# hides the structure left in the other two.
setores = pd.Series([SETOR.get(t, "other") for t in close.columns])
grupos = [g for g in setores.drop_duplicates().tolist()]
M = np.array([[carga[j][(setores == g).values].mean() for g in grupos] for j in range(3)])
M = M - M.mean(axis=1, keepdims=True)
ordem = np.argsort(M[1])
im = ax.imshow(M[:, ordem], aspect="auto", cmap="RdBu_r",
               vmin=-np.abs(M).max(), vmax=np.abs(M).max())
for j in range(3):
    for i, k in enumerate(ordem):
        ax.text(i, j, f"{M[j, k]:+.2f}", ha="center", va="center", fontsize=9,
                color="#12151c" if abs(M[j, k]) < np.abs(M).max() * 0.55 else "#ffffff")
ax.set_yticks(range(3))
ax.set_yticklabels([f"unit {i + 1}" for i in range(3)], color="#e6e9f0", fontsize=11)
ax.set_xticks(range(len(grupos)))
ax.set_xticklabels([grupos[k] for k in ordem], fontsize=9, rotation=30, ha="right")
ax.set_title(f"What the three units ended up standing for, {n} names",
             color=tc, fontsize=12, loc="left")
ax.grid(False)
save(f, "ae_cargas.png")

# light panel: the days the rebuild fails are the days worth looking at
f, ax, tc = fig(False, 7.2, 5.2)
# RAW series, no rolling mean. Smoothing put the chart peak on one day while the
# `ae_pior` value written to results.json landed on the day before, so the write up
# quoted one date and showed another.
s = pd.Series(erro[cut:], index=R.index[cut:])
ax.fill_between(s.index, 0, s.values, color="#7C4DFF", alpha=0.28)
ax.plot(s.index, s.values, color="#5B2FD6", lw=1.2)
# Annotate the series THAT WAS DRAWN. First version took the label positions from the
# raw series, which sits higher than the plotted line, so they fell off the axis.
# One label only. Two largest both landed on the same April 2025 spike and the two
# dates came out stacked on top of each other.
topo = s.dropna().nlargest(1)
for dia, v in topo.items():
    ax.annotate(dia.strftime("%d %b %Y"), xy=(dia, v), xytext=(0, 16),
                textcoords="offset points", ha="center", fontsize=10, color=INK,
                arrowprops=dict(arrowstyle="-", color="#9aa0ae", lw=1.0))
ax.set_ylabel("reconstruction error", color=tc)
ax.set_title("Days the rebuild fails, out of sample", color=tc, fontsize=12, loc="left")
ax.margins(y=0.16)
save(f, "ae_erro.png")

np.savez_compressed(OUT / "ae_eval.npz", lat=lat, erro=erro,
                    dias=R.index.values, cut=cut, cargas=carga)
CAMINHO.write_text(json.dumps(FACTS, indent=1))
print("\nFACTS:", json.dumps(FACTS, indent=1))
