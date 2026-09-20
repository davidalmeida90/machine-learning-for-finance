# -*- coding: utf-8 -*-
"""PCA na curva de juros e Lasso num painel macro, com dado puxado ao vivo.

Os tres modelos de arvore que moravam aqui sairam. Eles agora estao em `krauss.py`,
que replica Krauss, Do e Huck (2017) com janela deslizante e avaliacao por carteira,
que e a forma certa de julgar aquele tipo de modelo. O que sobra neste arquivo sao os
dois metodos que continuam sendo o que a pagina mostra, e que nao mudaram:

  1. PCA da curva do Tesouro. Nao supervisionado, entao nao ha alvo, nao ha divisao
     temporal e nao ha o que vazar. Mede quanta da variacao da curva inteira cabe em
     tres numeros.
  2. Lasso num painel de momento, volatilidade e macro. Supervisionado, com divisao
     TEMPORAL, e o ponto dele e a selecao: o L1 zera coeficiente, entao o modelo diz
     quais colunas usou e quais descartou.

Precos vem do Yahoo Finance, a curva e as series macro vem do FRED pelo endpoint CSV
publico, sem chave de API.

  py -3 pca_lasso.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import yfinance as yf

AQUI = Path(__file__).parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# Paleta clara, para a pagina do site.
TINTA, GRADE = "#1b2a4a", "#dfe3ec"
AZUL, TEAL, AMBAR, UVA, ROSA = "#2f5db0", "#0ca678", "#d97706", "#7048e8", "#d6336c"
FUNDO = "#ffffff"
INICIO, FIM = "2010-01-01", "2026-08-18"

resultados: dict = {}


def eixo(larg=12.0, alt=5.4):
    fig, ax = plt.subplots(figsize=(larg, alt), facecolor=FUNDO)
    ax.set_facecolor(FUNDO)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color(GRADE)
    ax.tick_params(colors="#6b7280", labelsize=10)
    ax.grid(color=GRADE, lw=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def salva(fig, nome):
    fig.savefig(AQUI / nome, facecolor=FUNDO, dpi=100, bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print(f"    {nome}")


# --------------------------------------------------------------------------- dados
def precos() -> pd.DataFrame:
    """SPY diario. auto_adjust=True ja devolve o preco ajustado por dividendo e split."""
    px = yf.download("SPY", start=INICIO, end=FIM, auto_adjust=True, progress=False)
    px = px["Close"]
    if isinstance(px, pd.DataFrame):
        px = px.iloc[:, 0]
    return px.dropna()


def cesta() -> pd.DataFrame:
    """Onze ETFs setoriais mais alguns fatores. Materia prima do Lasso."""
    tickers = ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLB", "XLRE",
               "XLC", "IWM", "EFA", "EEM", "TLT", "IEF", "HYG", "LQD", "GLD", "USO",
               "UUP", "VNQ"]
    px = yf.download(tickers, start=INICIO, end=FIM, auto_adjust=True, progress=False)
    px = px["Close"] if "Close" in px else px
    return px.dropna(axis=1, how="all").ffill().dropna()


def fred(series: list[str]) -> pd.DataFrame:
    """FRED pelo endpoint CSV publico, que dispensa chave de API."""
    saida = {}
    for s in series:
        r = requests.get("https://fred.stlouisfed.org/graph/fredgraph.csv",
                         params={"id": s}, timeout=90)
        r.raise_for_status()
        d = pd.read_csv(io.StringIO(r.text))
        d.columns = ["data", s]
        d["data"] = pd.to_datetime(d["data"])
        saida[s] = pd.to_numeric(d.set_index("data")[s], errors="coerce")
        print(f"    {s}: {saida[s].dropna().shape[0]} observacoes")
    return pd.DataFrame(saida)


def componentes(curva: pd.DataFrame):
    from sklearn.decomposition import PCA
    d = curva.dropna().diff().dropna()
    p = PCA(n_components=5).fit(d)
    var = p.explained_variance_ratio_ * 100
    anos = [0.25, 0.5, 1, 2, 3, 5, 7, 10, 20, 30]

    fig, ax = eixo(14.4, 7.6)
    nomes = [f"PC1  level  {var[0]:.1f}%", f"PC2  slope  {var[1]:.1f}%",
             f"PC3  curvature  {var[2]:.1f}%"]
    for i, (c, nome) in enumerate(zip((AZUL, AMBAR, TEAL), nomes)):
        ax.plot(anos, p.components_[i], color=c, lw=2.2, marker="o", ms=5, label=nome)
    ax.axhline(0, color=GRADE, lw=1.2)
    ax.set_xscale("log")
    ax.set_xticks(anos)
    ax.set_xticklabels([f"{a:g}" for a in anos])
    ax.set_title("Treasury curve, loadings of the first three components",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_xlabel("maturity in years", color="#6b7280")
    ax.legend(frameon=False, fontsize=10, labelcolor=TINTA)
    salva(fig, "pca_loadings.png")

    fig, ax = eixo(12, 4.6)
    ax.bar(range(1, 6), var, color=AZUL, width=0.58)
    ax.plot(range(1, 6), np.cumsum(var), color=UVA, lw=2.0, marker="o", ms=5)
    for i, v in enumerate(var, 1):
        ax.text(i, v + 1.6, f"{v:.1f}%", ha="center", color=TINTA, fontsize=10)
    ax.set_title("explained_variance_ratio_, daily changes in yield",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_xlabel("component", color="#6b7280")
    salva(fig, "pca_variance.png")

    return {"var": [round(float(v), 2) for v in var],
             "tres_primeiras": round(float(var[:3].sum()), 2),
             "observacoes": int(len(d)),
             "de": str(d.index[0].date()), "ate": str(d.index[-1].date())}


# ------------------------------------------------------------------------- 5. Lasso
def laco(painel: pd.DataFrame, alvo: pd.Series):
    from sklearn.linear_model import LassoCV, lasso_path
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit

    comum = painel.index.intersection(alvo.index)
    X, y = painel.loc[comum], alvo.loc[comum]
    corte = int(len(X) * 0.7)
    esc = StandardScaler().fit(X[:corte])
    Xtr, Xte = esc.transform(X[:corte]), esc.transform(X[corte:])

    cv = LassoCV(cv=TimeSeriesSplit(5), n_alphas=120, max_iter=20000,
                 random_state=0).fit(Xtr, y[:corte])
    vivos = [c for c, b in zip(X.columns, cv.coef_) if abs(b) > 1e-10]

    alphas, coefs, _ = lasso_path(Xtr, y[:corte], n_alphas=120)
    fig, ax = eixo()
    cores = [AZUL, AMBAR, TEAL, UVA, ROSA, "#0b7285", "#5c940d", "#862e9c"]
    for i in range(coefs.shape[0]):
        ax.plot(alphas, coefs[i], lw=1.4, color=cores[i % len(cores)], alpha=0.9)
    ax.axvline(cv.alpha_, color="#9aa1ae", ls=":", lw=1.6)
    ax.text(cv.alpha_ * 1.15, coefs.max() * 0.92, "alpha chosen by\ntime series CV",
            color="#6b7280", fontsize=10)
    ax.axhline(0, color=GRADE, lw=1.2)
    ax.set_xscale("log")
    ax.set_title(f"Coefficient path, {X.shape[1]} candidate predictors",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_xlabel("alpha", color="#6b7280")
    salva(fig, "lasso_path.png")

    nao_zero = (np.abs(coefs) > 1e-10).sum(axis=0)
    fig, ax = eixo(12, 4.6)
    ax.step(alphas, nao_zero, color=TEAL, lw=2.2, where="mid")
    ax.fill_between(alphas, nao_zero, step="mid", color=TEAL, alpha=0.13)
    ax.axvline(cv.alpha_, color="#9aa1ae", ls=":", lw=1.6)
    ax.axhline(len(vivos), color=AMBAR, ls="--", lw=1.4)
    ax.text(alphas.min(), len(vivos) + 1.2, f"{len(vivos)} survive at the chosen alpha",
            color=AMBAR, fontsize=10)
    ax.set_xscale("log")
    ax.set_title("Number of non zero coefficients", color=TINTA, fontsize=13, loc="left")
    ax.set_xlabel("alpha", color="#6b7280")
    salva(fig, "lasso_selection.png")

    from sklearn.metrics import r2_score
    prev = cv.predict(Xte)
    return {"candidatos": int(X.shape[1]), "vivos": len(vivos),
            "nomes": vivos[:10], "alpha": float(cv.alpha_),
            "r2_teste": float(r2_score(y[corte:], prev)),
            "r2_media": float(r2_score(y[corte:], np.full(len(prev), y[:corte].mean())))}


def main():
    print("1. PCA da curva de juros:")
    tenores = ["DGS3MO", "DGS6MO", "DGS1", "DGS2", "DGS3", "DGS5", "DGS7", "DGS10",
               "DGS20", "DGS30"]
    curva = fred(tenores)
    print(f"    curva: {len(curva)} dias, {curva.shape[1]} vertices, "
          f"{curva.index[0].date()} a {curva.index[-1].date()}")
    resultados["pca"] = componentes(curva)

    print("\n2. Lasso:")
    px = precos()
    etfs = cesta()
    mom = pd.concat({f"{c}_mom21": etfs[c].pct_change(21) for c in etfs.columns}, axis=1)
    vol = pd.concat({f"{c}_vol21": etfs[c].pct_change().rolling(21).std()
                     for c in etfs.columns}, axis=1)
    macro = fred(["DGS10", "DGS2", "T10Y2Y", "VIXCLS", "BAMLH0A0HYM2", "DTWEXBGS"])
    macro = macro.reindex(etfs.index).ffill()
    painel = pd.concat([mom, vol, macro], axis=1).dropna()
    frente = (px.shift(-21) / px - 1).reindex(painel.index).dropna()
    painel = painel.loc[frente.index]
    print(f"    painel: {painel.shape[0]} linhas x {painel.shape[1]} colunas")
    resultados["lasso"] = laco(painel, frente)

    resultados["periodo"] = {"de": str(painel.index[0].date()),
                             "ate": str(painel.index[-1].date()),
                             "linhas": int(len(painel))}
    (AQUI / "resultados.json").write_text(
        json.dumps(resultados, indent=1, default=str), encoding="utf-8")
    print("\n" + json.dumps(resultados, indent=1, default=str)[:2000])
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
