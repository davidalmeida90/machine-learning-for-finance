# -*- coding: utf-8 -*-
"""Replica o metodo de Krauss, Do e Huck (2017) so com as tres arvores, em dado gratuito.

Paper: "Deep neural networks, gradient-boosted trees, random forests: Statistical arbitrage
on the S&P 500", European Journal of Operational Research 259(2). Eles reportam, antes de
custo, 0,43% ao dia para a floresta e 0,37% para o boosting, de 1992 a 2015.

O que muda em relacao ao que a pagina ja tem, e cada item importa:

  1. HORIZONTE DE UM DIA, nao cinco.
  2. 31 ATRIBUTOS, todos retorno acumulado: m em {1..20} e {40,60,...,240}. Nada de RSI,
     media movel ou volatilidade.
  3. JANELA DESLIZANTE: treina em 750 dias, opera nos 250 seguintes, anda 250 e refaz.
     A pagina treinava UMA vez e previa quatro anos com o modelo velho.
  4. AVALIACAO POR CARTEIRA, nao por AUC no painel inteiro. Compra as k melhores e vende
     as k piores probabilidades, jogando fora o meio incerto, que e onde nao ha sinal.

O ponto 4 e o que reconcilia tudo: o proprio paper diz que o acerto direcional deles fica
"sempre acima de 50 por cento", sem numero grande nenhum. A AUC baixa que vimos nao era
sinal de erro, era a metrica errada para uma estrategia que so opera as pontas.

  py -3 krauss.py
"""
from __future__ import annotations

import argparse
import gc
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
AZUL, TEAL, AMBAR, ROSA, UVA = "#2f5db0", "#0ca678", "#d97706", "#d6336c", "#7048e8"

TREINO, OPERA = 750, 250          # dias, exatamente como no paper
KS = [10, 50, 100]                # tamanho de cada perna
PERIODOS = ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
            + list(range(40, 241, 20)))   # 20 + 11 = 31 atributos


def eixo(larg=12.0, alt=5.0):
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


def dados(desde=None, ate=None):
    px = pd.read_parquet(AQUI / "precos.parquet").astype("float32")
    if desde:
        px = px.loc[px.index >= desde]
    if ate:
        px = px.loc[px.index <= ate]
    comp = json.loads((AQUI / "membership.json").read_text(encoding="utf-8"))
    marcos = sorted(comp)
    masc = pd.DataFrame(False, index=px.index, columns=px.columns)
    for k, d0 in enumerate(marcos):
        d1 = marcos[k + 1] if k + 1 < len(marcos) else "2100-01-01"
        faixa = (px.index >= d0) & (px.index < d1)
        masc.loc[faixa, [t for t in comp[d0] if t in masc.columns]] = True
    return px, masc


def atributos(px, masc):
    """31 retornos acumulados, cada um padronizado NA SECAO TRANSVERSAL do dia.

    Padronizar por dia, e nao pela serie inteira, e o que torna o atributo comparavel
    entre epocas: um retorno de 2% em 2008 e um de 2% em 2017 nao significam a mesma
    coisa, mas "acima da media do dia por 1,3 desvios" significa.
    """
    quadros = {}
    for m in PERIODOS:
        r = (px / px.shift(m) - 1).where(masc)
        mu = r.mean(axis=1)
        sd = r.std(axis=1).replace(0, np.nan)
        quadros[f"r_{m}"] = ((r.sub(mu, axis=0)).div(sd, axis=0)).astype("float32")
    # alvo: bate a mediana transversal do retorno de UM dia a frente
    r1 = (px.shift(-1) / px - 1).where(masc)
    alvo = (r1.rank(axis=1, pct=True) > 0.5).astype("float32").where(r1.notna())
    return quadros, alvo, r1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default=None, help="corta o inicio da amostra")
    ap.add_argument("--ate", default=None, help="corta o fim da amostra")
    ap.add_argument("--arvores", type=int, default=60, help="arvores da floresta")
    a = ap.parse_args()

    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    import xgboost as xgb

    px, masc = dados(a.desde, a.ate)
    print(f"painel: {px.shape[1]} acoes, {px.shape[0]} pregoes "
          f"({px.index[0].date()} a {px.index[-1].date()})")
    quadros, alvo, r1 = atributos(px, masc)
    dias = px.index
    print(f"atributos: {len(quadros)} retornos acumulados, padronizados por dia")

    janelas = list(range(max(PERIODOS), len(dias) - TREINO - OPERA, OPERA))
    print(f"janelas deslizantes: {len(janelas)}  "
          f"(treina {TREINO}, opera {OPERA}, anda {OPERA})\n")

    modelos = {
        "arvore": lambda: DecisionTreeClassifier(max_depth=8, min_samples_leaf=500,
                                                 random_state=0),
        "floresta": lambda: RandomForestClassifier(n_estimators=a.arvores, max_depth=12,
                                                   min_samples_leaf=200, max_features="sqrt",
                                                   n_jobs=3, random_state=0),
        # hist com poucas caixas: o modo exato guarda a matriz ordenada inteira e
        # estourou a memoria nesta maquina. 100 iteracoes e o valor do paper.
        "boosting": lambda: xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                              max_depth=3, subsample=0.8,
                                              colsample_bytree=0.8, n_jobs=3,
                                              tree_method="hist", max_bin=96,
                                              eval_metric="logloss"),
    }
    retornos = {n: {k: [] for k in KS} for n in modelos}
    acertos = {n: [] for n in modelos}
    datas_op = []

    def empilha(faixa):
        X = pd.concat({n: q.loc[faixa].stack() for n, q in quadros.items()}, axis=1)
        y = alvo.loc[faixa].stack().reindex(X.index)
        ok = y.notna() & X.notna().all(axis=1)
        return X[ok].astype("float32"), y[ok].astype("int8")

    for j, ini in enumerate(janelas, 1):
        f_tr = dias[ini:ini + TREINO]
        f_op = dias[ini + TREINO:ini + TREINO + OPERA]
        Xtr, ytr = empilha(f_tr)
        Xop, yop = empilha(f_op)
        if len(ytr) < 5000 or len(yop) < 1000:
            continue

        linha = f"  janela {j:>2}/{len(janelas)}  opera {f_op[0].date()} a {f_op[-1].date()}"
        Atr, btr = Xtr.to_numpy(dtype="float32"), ytr.to_numpy()
        Aop = Xop.to_numpy(dtype="float32")
        for nome, fab in modelos.items():
            m = fab().fit(Atr, btr)
            p = pd.Series(m.predict_proba(Aop)[:, 1], index=Xop.index)
            del m; gc.collect()
            acertos[nome].append(float(((p > 0.5).astype(int) == yop).mean()))
            # carteira: compra as k maiores probabilidades do dia, vende as k menores
            for k in KS:
                for d, grupo in p.groupby(level=0):
                    if len(grupo) < 2 * k:
                        continue
                    ordem = grupo.sort_values(ascending=False)
                    longos = ordem.index[:k].get_level_values(1)
                    curtos = ordem.index[-k:].get_level_values(1)
                    rr = r1.loc[d]
                    retornos[nome][k].append(
                        (d, j, float(rr[longos].mean() - rr[curtos].mean())))
        datas_op.append(f_op[-1])
        print(linha, flush=True)
        del Xtr, ytr, Xop, yop, Atr, btr, Aop
        gc.collect()

    print("\n" + "=" * 68)
    print("RESULTADO, retorno diario da carteira comprada e vendida, ANTES de custo")
    print("=" * 68)
    print(f"{'modelo':10} {'k':>5} {'ret/dia':>9} {'anual':>9} {'desvio':>8} "
          f"{'sharpe':>7} {'acerto':>8}")
    print("-" * 62)
    saida = {}
    for nome in modelos:
        for k in KS:
            reg = retornos[nome][k]
            if not reg:
                continue
            v = np.array([x[2] for x in reg], dtype=float)
            md, sd = v.mean(), v.std(ddof=1)
            sharpe = (md / sd) * np.sqrt(252) if sd else 0.0
            anual = (1 + md) ** 252 - 1
            ac = float(np.mean(acertos[nome]))
            saida[f"{nome}_{k}"] = {"ret_dia": round(float(md), 6),
                                    "anual": round(float(anual), 4),
                                    "desvio": round(float(sd), 5),
                                    "sharpe": round(float(sharpe), 2),
                                    "acerto": round(ac, 4), "dias": int(v.size)}
            print(f"{nome:10} {k:>5} {md*100:>8.3f}% {anual*100:>8.1f}% {sd*100:>7.2f}% "
                  f"{sharpe:>7.2f} {ac*100:>7.2f}%")

    # ------------------------------------------------------------------ grafico
    # quebra por janela, que e o teste de honestidade do resultado agregado
    print("\nretorno medio por dia, k=10, JANELA A JANELA")
    print(f"  {'janela':>7} {'periodo':>26} {'arvore':>9} {'floresta':>10} {'boosting':>10}")
    js = sorted({x[1] for x in retornos['floresta'][10]})
    for jj in js:
        cel, per = [], ""
        for nome in modelos:
            vv = [x[2] for x in retornos[nome][10] if x[1] == jj]
            ds = [x[0] for x in retornos[nome][10] if x[1] == jj]
            if ds:
                per = f"{min(ds).date()} a {max(ds).date()}"
            cel.append(np.mean(vv) * 100 if vv else float('nan'))
        print(f"  {jj:>7} {per:>26} {cel[0]:>8.3f}% {cel[1]:>9.3f}% {cel[2]:>9.3f}%")
    saida["por_janela"] = {
        nome: {str(jj): round(float(np.mean([x[2] for x in retornos[nome][10]
                                             if x[1] == jj])), 6) for jj in js}
        for nome in modelos}

    fig, ax = eixo(12, 5.2)
    for nome, cor in zip(modelos, (AMBAR, TEAL, UVA)):
        v = np.array([x[2] for x in retornos[nome][10]], dtype=float)
        if v.size:
            ax.plot(np.cumprod(1 + v), color=cor, lw=1.8, label=f"{nome}, k=10")
    ax.axhline(1, color="#9aa1ae", lw=1.2)
    ax.set_yscale("log")
    ax.set_title("Long short portfolio, k=10, before transaction costs",
                 color=TINTA, fontsize=13, loc="left")
    ax.set_xlabel("trading days out of sample", color="#6b7280")
    ax.set_ylabel("growth of 1", color="#6b7280")
    ax.legend(frameon=False, fontsize=10, labelcolor=TINTA)
    fig.savefig(AQUI / "krauss_equity.png", facecolor=FUNDO, dpi=100,
                bbox_inches="tight", pad_inches=0.28)
    plt.close(fig)
    print("\n  krauss_equity.png")
    (AQUI / "krauss.json").write_text(json.dumps(saida, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
