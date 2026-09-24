"""Trains the three architectures on real market data and saves the panels.

Run with py -3.11: the default interpreter has a broken torch install
(WinError 193 on shm.dll). 3.11 carries torch 2.6.0+cu124 with CUDA.

Each page gets one dark panel and one light panel, which is the half-and-half
rule from the carousel spec, and every chart has its own palette so six panels
do not read as the same chart repeated.
"""
import json, warnings
from pathlib import Path
import numpy as np, pandas as pd, yfinance as yf
import torch, torch.nn as nn
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

# Charts land in figures/, which is the folder shipped with this repository, so a
# fresh run reproduces exactly what you see here.
OUT = Path(__file__).parent / "figures"; OUT.mkdir(exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(0); np.random.seed(0)
INK, GRID_L, GRID_D = "#12151c", "#e6e8ee", "#2a3040"
# Merge into whatever results.json already holds, so running this after
# cnn_autoencoder.py keeps the CNN and autoencoder numbers instead of wiping them.
CAMINHO = Path(__file__).parent / "results.json"
FACTS = json.loads(CAMINHO.read_text()) if CAMINHO.exists() else {}

def fig(dark: bool, w=7.2, h=5.2):
    f, ax = plt.subplots(figsize=(w, h), dpi=160)
    bg = "#0d1117" if dark else "#ffffff"
    f.patch.set_facecolor(bg); ax.set_facecolor(bg)
    ax.grid(color=GRID_D if dark else GRID_L, lw=0.9)
    ax.set_axisbelow(True)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID_D if dark else GRID_L)
    ax.tick_params(colors="#8b93a7" if dark else "#6b7280", labelsize=9, length=0)
    return f, ax, ("#e6e9f0" if dark else INK)

def save(f, name):
    f.savefig(OUT / name, facecolor=f.get_facecolor(), bbox_inches="tight", pad_inches=0.16)
    plt.close(f); print("  saved", name)

# ---------------------------------------------------------------- 1. feed forward
# Task: learn the implied volatility surface, a static map from (moneyness, maturity)
# to IV. Static input, static output, so a plain feed forward network is the honest
# choice here rather than anything with memory.
print("[1/3] feed forward on the SPY implied volatility surface")
spy = yf.Ticker("SPY"); spot = float(spy.history(period="5d")["Close"].iloc[-1])
rows = []
for exp in spy.options[:16]:
    try: ch = spy.option_chain(exp)
    except Exception: continue
    t = (pd.Timestamp(exp) - pd.Timestamp.today()).days / 365.0
    if t <= 0.02: continue
    # OTM only: calls above spot, puts below. A call and a put on the same strike
    # carry different quoted IVs, and feeding both taught the net to average them.
    for df, side in ((ch.calls, "c"), (ch.puts, "p")):
        d = df[(df.impliedVolatility > 0.03) & (df.impliedVolatility < 1.5)
               & (df.volume > 5) & (df.bid > 0.05)]
        d = d[d.strike >= spot] if side == "c" else d[d.strike < spot]
        for k, iv in zip(d.strike, d.impliedVolatility):
            rows.append((np.log(k / spot), t, iv))
S = pd.DataFrame(rows, columns=["m", "t", "iv"])
S = S[(S.m.abs() < 0.35)].dropna()
FACTS["ffn_quotes"] = len(S); FACTS["ffn_spot"] = round(spot, 2)
print(f"      {len(S):,} live quotes, spot {spot:.2f}")

# Outside US market hours Yahoo returns the chain with every bid at zero and IV at
# 0.00001, and the filters above keep nothing. Section 1 is skipped with a message
# rather than crashing, because sections 2 and 3 only need daily closes.
if len(S) < 200:
    print("      chain is empty (market closed?), section 1 skipped; run during US hours for the surface")
else:
    X = torch.tensor(S[["m", "t"]].values, dtype=torch.float32, device=DEV)
    y = torch.tensor(S[["iv"]].values, dtype=torch.float32, device=DEV)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    Xn = (X - mu) / sd
    n = len(S); idx = torch.randperm(n, device=DEV); cut = int(n * 0.8)
    tr, te = idx[:cut], idx[cut:]

    ffn = nn.Sequential(nn.Linear(2, 96), nn.Tanh(), nn.Linear(96, 96), nn.Tanh(),
                        nn.Linear(96, 96), nn.Tanh(), nn.Linear(96, 1)).to(DEV)
    opt = torch.optim.Adam(ffn.parameters(), lr=3e-3)
    hist = []
    for ep in range(6000):
        opt.zero_grad(); l = ((ffn(Xn[tr]) - y[tr]) ** 2).mean(); l.backward(); opt.step()
        if ep % 50 == 0:
            with torch.no_grad(): v = ((ffn(Xn[te]) - y[te]) ** 2).mean().item()
            hist.append((ep, l.item(), v))
    H = np.array(hist)
    with torch.no_grad():
        rmse = float(((ffn(Xn[te]) - y[te]) ** 2).mean().sqrt())
    FACTS["ffn_rmse"] = round(rmse * 100, 3)
    print(f"      held out RMSE {rmse*100:.3f} vol points")

    # dark panel: the fitted surface as a smile per maturity, over the real quotes
    f, ax, tc = fig(True)
    gm = np.linspace(-0.33, 0.33, 120)
    mats = sorted(S.t.unique())
    pick = [mats[0], mats[len(mats)//3], mats[2*len(mats)//3], mats[-1]]
    cols = ["#5AB4F0", "#7EE3B0", "#F2C14E", "#F2765C"]
    for t_, c in zip(pick, cols):
        sub = S[np.isclose(S.t, t_)]
        ax.scatter(sub.m, sub.iv * 100, s=7, color=c, alpha=0.28, lw=0)
        gx = torch.tensor(np.c_[gm, np.full_like(gm, t_)], dtype=torch.float32, device=DEV)
        with torch.no_grad(): gy = ffn((gx - mu) / sd).cpu().numpy().ravel()
        ax.plot(gm, gy * 100, color=c, lw=2, label=f"{t_*365:.0f}d")
    ax.set_xlabel("log moneyness", color=tc); ax.set_ylabel("implied vol %", color=tc)
    ax.set_title("Fitted surface against live quotes", color=tc, fontsize=12, loc="left")
    lg = ax.legend(frameon=False, fontsize=9, ncol=4)
    for t_ in lg.get_texts(): t_.set_color(tc)
    save(f, "ffn_surface.png")

    # light panel: training curve
    # Log scale on the error, otherwise the first fifty epochs own the whole axis and the
    # other 5,950 read as one flat line, which is what the first version looked like.
    f, ax, tc = fig(False)
    ax.plot(H[:, 0], H[:, 1] * 1e4, color="#1E56B8", lw=2, label="train")
    ax.plot(H[:, 0], H[:, 2] * 1e4, color="#C0395C", lw=2, label="held out")
    ax.set_yscale("log")
    ax.set_xlabel("epoch", color=tc); ax.set_ylabel("MSE x 10,000, log scale", color=tc)
    ax.set_title("Training and held out error stay together", color=tc, fontsize=12, loc="left")
    ax.legend(frameon=False, fontsize=9)
    np.save(OUT / "ffn_hist.npy", H)
    save(f, "ffn_loss.png")
    json.dump(FACTS, open(Path(__file__).parent / "results.json", "w"), indent=1)
    print("stage 1 done")

    # ------------------------------------------------------- 2. recurrent (LSTM)
    # Task: forecast forward realised volatility from a sequence of past returns.
    # Order matters here, which is the whole reason to reach for memory.
print("[2/3] LSTM on SPY realised volatility")
px = yf.Ticker("SPY").history(period="20y")["Close"].dropna()
r = np.log(px / px.shift(1)).dropna()
LOOK, FWD = 40, 10
fwd_vol = r[::-1].rolling(FWD).std()[::-1].shift(-1) * np.sqrt(252)
feat = np.abs(r.values) * np.sqrt(252)
Xs, ys, ds = [], [], []
for i in range(LOOK, len(r) - FWD - 1):
    v = fwd_vol.iloc[i]
    if np.isfinite(v):
        Xs.append(feat[i - LOOK:i]); ys.append(v); ds.append(r.index[i])
Xs = np.array(Xs, dtype=np.float32)[:, :, None]; ys = np.array(ys, dtype=np.float32)[:, None]
cut = int(len(Xs) * 0.75)                      # split in TIME, never shuffled
Xtr, Xte = torch.tensor(Xs[:cut], device=DEV), torch.tensor(Xs[cut:], device=DEV)
ytr, yte = torch.tensor(ys[:cut], device=DEV), torch.tensor(ys[cut:], device=DEV)
FACTS["rnn_train_to"] = str(ds[cut].date()); FACTS["rnn_n"] = len(Xs)
print(f"      {len(Xs):,} windows, out of sample from {ds[cut].date()}")

class LSTMVol(nn.Module):
    def __init__(s, h=48):
        super().__init__(); s.l = nn.LSTM(1, h, batch_first=True); s.o = nn.Linear(h, 1)
    def forward(s, x): return s.o(s.l(x)[0][:, -1])
rnn = LSTMVol().to(DEV); opt = torch.optim.Adam(rnn.parameters(), lr=2e-3)
for ep in range(400):
    opt.zero_grad(); ((rnn(Xtr) - ytr) ** 2).mean().backward(); opt.step()
with torch.no_grad(): pr = rnn(Xte).cpu().numpy().ravel()
act = yte.cpu().numpy().ravel()
corr = float(np.corrcoef(pr, act)[0, 1]); rm = float(np.sqrt(((pr - act) ** 2).mean()))
FACTS["rnn_corr"] = round(corr, 3); FACTS["rnn_rmse"] = round(rm * 100, 2)
print(f"      out of sample corr {corr:.3f}, RMSE {rm*100:.2f} vol points")

f, ax, tc = fig(False)
dts = ds[cut:]
ax.plot(dts, act * 100, color="#8a93a3", lw=1.5, label="realised")
ax.plot(dts, pr * 100, color="#1f7a4d", lw=1.6, label="LSTM forecast")
ax.set_ylabel("annualised vol %", color=tc)
ax.set_title(f"Forward {FWD} day vol, out of sample", color=tc, fontsize=12, loc="left")
ax.legend(frameon=False, fontsize=9)
save(f, "rnn_forecast.png")

f, ax, tc = fig(True)
ac = [float(pd.Series(np.abs(r.values)).autocorr(l)) for l in range(1, 61)]
ax.bar(range(1, 61), ac, color="#B98CE0", width=0.9)
ax.set_xlabel("lag, trading days", color=tc); ax.set_ylabel("autocorrelation", color=tc)
ax.set_title("Volatility remembers, returns do not", color=tc, fontsize=12, loc="left")
save(f, "rnn_memory.png")

# ------------------------------------------------------------ 3. transformer
# Same target, but attention picks which lags matter instead of walking the
# sequence one step at a time.
print("[3/3] transformer on the same target")
class Attn(nn.Module):
    def __init__(s, d=32):
        super().__init__()
        s.p = nn.Linear(1, d); s.pos = nn.Parameter(torch.randn(1, LOOK, d) * 0.02)
        s.a = nn.MultiheadAttention(d, 4, batch_first=True)
        s.n = nn.LayerNorm(d); s.o = nn.Linear(d, 1)
    def forward(s, x, want=False):
        h = s.p(x) + s.pos
        z, w = s.a(h, h, h, need_weights=want, average_attn_weights=True)
        return s.o(s.n(h + z)[:, -1]), w
tr_ = Attn().to(DEV); opt = torch.optim.Adam(tr_.parameters(), lr=2e-3)
for ep in range(400):
    opt.zero_grad(); ((tr_(Xtr)[0] - ytr) ** 2).mean().backward(); opt.step()
with torch.no_grad():
    pt, _ = tr_(Xte); pt = pt.cpu().numpy().ravel()
    _, W = tr_(Xte[:256], want=True); W = W.mean(0).cpu().numpy()
ct = float(np.corrcoef(pt, act)[0, 1]); rt = float(np.sqrt(((pt - act) ** 2).mean()))
FACTS["tr_corr"] = round(ct, 3); FACTS["tr_rmse"] = round(rt * 100, 2)
print(f"      out of sample corr {ct:.3f}, RMSE {rt*100:.2f} vol points")

f, ax, tc = fig(True)
im_ = ax.imshow(W, cmap="magma", aspect="auto", origin="lower")
ax.set_xlabel("attends to lag", color=tc); ax.set_ylabel("from position", color=tc)
ax.set_title("Learned attention over 40 days", color=tc, fontsize=12, loc="left")
cb = f.colorbar(im_, ax=ax, fraction=0.045); cb.ax.tick_params(colors=tc, labelsize=8)
cb.outline.set_visible(False)
save(f, "tr_attention.png")

f, ax, tc = fig(False)
ax.scatter(act * 100, pt * 100, s=9, color="#C2571A", alpha=0.45, lw=0)
lo, hi = act.min() * 100, act.max() * 100
ax.plot([lo, hi], [lo, hi], color="#8a93a3", lw=1.4, ls=(0, (5, 4)))
ax.set_xlabel("realised vol %", color=tc); ax.set_ylabel("predicted vol %", color=tc)
ax.set_title(f"Out of sample fit, correlation {ct:.2f}", color=tc, fontsize=12, loc="left")
save(f, "tr_scatter.png")

json.dump(FACTS, open(Path(__file__).parent / "results.json", "w"), indent=1)
print("\nFACTS:", json.dumps(FACTS, indent=1))
