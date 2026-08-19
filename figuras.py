# -*- coding: utf-8 -*-
"""As saidas que faltavam na pagina: cada bloco de codigo passa a terminar em algo visivel.

O David pediu isto olhando a pagina: um script que roda e nao mostra o que produziu le como
lista de codigo, nao como notebook. Entao cada etapa ganha a sua saida.

  universo.png    Step 2  quantos nomes cada fotografia semestral tem, e quantos ja passaram
  membros_dia.png Step 3  membros do indice por dia contra os que tem preco utilizavel
  amostra.png     Step 4  linhas de treino e de operacao em cada janela
  janelas.png     Step 5  a linha do tempo das cinco janelas, treino e operacao lado a lado
  arvore.png      Step 6  os primeiros niveis da arvore ajustada, preto no branco

  py -3 figuras.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AQUI = Path(__file__).parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

TINTA, GRADE, FUNDO = "#1b2a4a", "#dfe3ec", "#ffffff"
AZUL, TEAL, AMBAR, UVA, ROSA = "#2f5db0", "#0ca678", "#d97706", "#7048e8", "#d6336c"
CINZA = "#9aa1ae"
DESDE, ATE = "2007-01-01", "2015-12-31"


def eixo(larg=12.0, alt=4.6):
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


def main():
    import krauss as K

    comp = json.loads((AQUI / "membership.json").read_text(encoding="utf-8"))
    px, masc = K.dados(DESDE, ATE)
    dias = px.index

    # ---------------------------------------------------------------- Step 2
    marcos = sorted(comp)
    tamanho = [len(comp[m]) for m in marcos]
    vistos, acumulado = set(), []
    for m in marcos:
        vistos |= set(comp[m])
        acumulado.append(len(vistos))
    x = pd.to_datetime(marcos)

    fig, ax = eixo(12, 4.4)
    ax.plot(x, acumulado, color=UVA, lw=2.2, label="tickers seen at least once")
    ax.plot(x, tamanho, color=AZUL, lw=2.0, label="members in that snapshot")
    ax.fill_between(x, tamanho, acumulado, color=UVA, alpha=0.08)
    ax.annotate(f"{acumulado[-1]} ever\n{tamanho[-1]} today", xy=(x[-1], acumulado[-1]),
                xytext=(-96, -6), textcoords="offset points", color=UVA, fontsize=10)
    ax.set_ylim(0, max(acumulado) * 1.12)
    ax.set_title("Every snapshot holds about 500 names, and the union keeps growing",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_ylabel("tickers", color="#6b7280")
    ax.legend(frameon=False, fontsize=10, labelcolor=TINTA, loc="lower left")
    salva(fig, "universo.png")

    # ---------------------------------------------------------------- Step 3
    membros = masc.sum(axis=1)
    # quantos eram membros no papel naquele dia, tenha preco ou nao
    no_papel = pd.Series(index=dias, dtype=float)
    for k, d0 in enumerate(marcos):
        d1 = marcos[k + 1] if k + 1 < len(marcos) else "2100-01-01"
        faixa = (dias >= d0) & (dias < d1)
        no_papel[faixa] = len(comp[d0])

    fig, ax = eixo(12, 4.4)
    ax.plot(dias, no_papel, color=CINZA, lw=1.6, label="index members that day")
    ax.plot(dias, membros, color=TEAL, lw=1.8, label="of those, with usable Yahoo history")
    ax.fill_between(dias, membros, no_papel, color=ROSA, alpha=0.10)
    ax.set_ylim(0, 560)
    ax.set_xlim(pd.Timestamp(marcos[0]), dias[-1])
    ax.text(dias[int(len(dias) * 0.30)], 430,
            "the shaded gap is what Yahoo no longer serves,\nand it leans towards companies that failed",
            color=ROSA, fontsize=10)
    ax.set_title("Point in time membership, day by day", color=TINTA, fontsize=13, loc="left")
    ax.set_ylabel("stocks in the panel", color="#6b7280")
    ax.legend(frameon=False, fontsize=10, labelcolor=TINTA, loc="lower right")
    salva(fig, "membros_dia.png")

    # ------------------------------------------------- Step 4 e 5, tamanho da amostra
    quadros, alvo, r1 = K.atributos(px, masc)
    janelas = list(range(max(K.PERIODOS), len(dias) - K.TREINO - K.OPERA, K.OPERA))

    def conta(faixa):
        X = pd.concat({n: q.loc[faixa].stack() for n, q in quadros.items()}, axis=1)
        y = alvo.loc[faixa].stack().reindex(X.index)
        ok = y.notna() & X.notna().all(axis=1)
        return int(ok.sum()), float(y[ok].mean())

    linhas = []
    for j, ini in enumerate(janelas, 1):
        f_tr = dias[ini:ini + K.TREINO]
        f_op = dias[ini + K.TREINO:ini + K.TREINO + K.OPERA]
        n_tr, m_tr = conta(f_tr)
        n_op, m_op = conta(f_op)
        linhas.append({"janela": j, "tr0": f_tr[0], "tr1": f_tr[-1], "op0": f_op[0],
                       "op1": f_op[-1], "n_tr": n_tr, "n_op": n_op,
                       "alvo_tr": m_tr, "alvo_op": m_op})
        print(f"  janela {j}: treino {n_tr:,} linhas (alvo {m_tr:.4f}), "
              f"operacao {n_op:,} (alvo {m_op:.4f})")
    tab = pd.DataFrame(linhas)

    fig, ax = eixo(12, 4.4)
    larg = 0.38
    p = np.arange(len(tab))
    ax.bar(p - larg / 2, tab["n_tr"] / 1000, larg, color=AZUL, label="training rows")
    ax.bar(p + larg / 2, tab["n_op"] / 1000, larg, color=AMBAR, label="trading rows")
    for i, r in tab.iterrows():
        ax.text(i - larg / 2, r["n_tr"] / 1000 + 4, f"{r['n_tr']/1000:.0f}k",
                ha="center", color=AZUL, fontsize=9.5)
        ax.text(i + larg / 2, r["n_op"] / 1000 + 4, f"{r['n_op']/1000:.0f}k",
                ha="center", color=AMBAR, fontsize=9.5)
    ax.set_xticks(p, [f"window {i}" for i in tab["janela"]])
    ax.set_ylim(0, tab["n_tr"].max() / 1000 * 1.18)
    ax.set_title("One row is one stock on one day, and every window refits on more of them",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_ylabel("rows, thousands", color="#6b7280")
    ax.legend(frameon=False, fontsize=10, labelcolor=TINTA, loc="upper left")
    salva(fig, "amostra.png")

    # ---------------------------------------------------------------- Step 5, linha do tempo
    fig, ax = eixo(12, 4.0)
    for i, r in tab.iterrows():
        y = len(tab) - i
        ax.barh(y, (r["tr1"] - r["tr0"]).days, left=r["tr0"], height=0.52,
                color=AZUL, alpha=0.85)
        ax.barh(y, (r["op1"] - r["op0"]).days, left=r["op0"], height=0.52,
                color=AMBAR)
        ax.text(r["tr0"], y + 0.42, f"window {r['janela']}", color=TINTA, fontsize=9.5)
    ax.set_yticks([])
    ax.set_ylim(0.3, len(tab) + 1.1)
    ax.grid(axis="y", lw=0)
    ax.set_title("Train on 750 days, trade the next 250, step forward and refit",
                 color=TINTA, fontsize=13, loc="left")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=AZUL, alpha=0.85, label="750 days of training"),
                       Patch(color=AMBAR, label="250 days traded, the only part ever scored")],
              frameon=False, fontsize=10, labelcolor=TINTA, loc="lower left")
    salva(fig, "janelas.png")

    # ---------------------------------------------------------------- Step 6, a arvore
    from sklearn.tree import DecisionTreeClassifier, plot_tree
    ini = janelas[0]
    faixa = dias[ini:ini + K.TREINO]
    X = pd.concat({n: q.loc[faixa].stack() for n, q in quadros.items()}, axis=1)
    y = alvo.loc[faixa].stack().reindex(X.index)
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok].astype("float32"), y[ok].astype("int8")
    raso = DecisionTreeClassifier(max_depth=3, min_samples_leaf=500,
                                  random_state=0).fit(X.to_numpy("float32"), y.to_numpy())
    fig, ax = plt.subplots(figsize=(13.5, 5.4), facecolor=FUNDO)
    plot_tree(raso, feature_names=list(X.columns), class_names=["below", "above"],
              filled=False, impurity=False, proportion=True, rounded=False,
              fontsize=8, ax=ax)
    ax.set_title("First three levels of the fitted tree, window 1",
                 color=TINTA, fontsize=13, loc="left")
    salva(fig, "arvore.png")

    tab.to_json(AQUI / "janelas.json", orient="records", date_format="iso")
    print("    janelas.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
