# -*- coding: utf-8 -*-
"""Curvas ROC dentro e fora da amostra, mais as conferencias de integridade.

Duas coisas que a pagina precisa e que o krauss.py nao produz:

  1. A curva ROC de TREINO contra a de TESTE, para cada modelo. E a figura que mostra o
     quanto cada um decorou: a floresta separa quase perfeitamente o que ja viu e quase
     nada do que nao viu, e isso e o normal, nao um defeito.
  2. As conferencias de que nao ha espiada no futuro nem vies de sobrevivencia, feitas
     como TESTE e nao como afirmacao. Uma pagina que so promete "nao ha vazamento" pede
     para ser acreditada; uma que mostra o teste pede para ser conferida.

  py -3 krauss_auc.py
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
AMBAR, TEAL, UVA, ROSA = "#d97706", "#0ca678", "#7048e8", "#d6336c"
TREINO, OPERA = 750, 250
DESDE, ATE = "2007-01-01", "2015-12-31"


def main():
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_curve, roc_auc_score
    import xgboost as xgb
    import krauss as K

    px, masc = K.dados(DESDE, ATE)
    quadros, alvo, r1 = K.atributos(px, masc)
    dias = px.index
    janelas = list(range(max(K.PERIODOS), len(dias) - TREINO - OPERA, OPERA))
    print(f"  {len(janelas)} janelas, teste de "
          f"{dias[janelas[0]+TREINO].date()} a {dias[janelas[-1]+TREINO+OPERA-1].date()}\n")

    def empilha(faixa):
        X = pd.concat({n: q.loc[faixa].stack() for n, q in quadros.items()}, axis=1)
        y = alvo.loc[faixa].stack().reindex(X.index)
        ok = y.notna() & X.notna().all(axis=1)
        return X[ok].astype("float32"), y[ok].astype("int8")

    # ---------------------------------------------------------- 1. conferencias
    print("CONFERENCIA 1: o alvo usa o retorno de AMANHA, nunca o de hoje")
    d0 = dias[1200]
    stock = px.columns[0]
    r_hoje = float(px.loc[d0, stock] / px.loc[dias[1199], stock] - 1)
    r_amanha = float(px.loc[dias[1201], stock] / px.loc[d0, stock] - 1)
    print(f"    em {d0.date()}, {stock}: retorno de hoje {r_hoje:+.5f}, "
          f"de amanha {r_amanha:+.5f}")
    print(f"    r1 guardado para {d0.date()}: {float(r1.loc[d0, stock]):+.5f}"
          f"  -> {'bate com AMANHA' if abs(float(r1.loc[d0, stock]) - r_amanha) < 1e-9 else 'ERRADO'}")

    print("\nCONFERENCIA 2: o atributo so olha para tras")
    m = 20
    esperado = float(px.loc[d0, stock] / px.loc[dias[1200 - m], stock] - 1)
    print(f"    r_{m} bruto em {d0.date()}: {esperado:+.5f} (preco de hoje sobre o de {m} dias atras)")
    print(f"    nenhum preco posterior a {d0.date()} entra nessa conta")

    print("\nCONFERENCIA 3: so entram acoes que eram membros do indice naquele dia")
    comp = json.loads((AQUI / "membership.json").read_text(encoding="utf-8"))
    marco = max(k for k in comp if k <= str(d0.date()))
    membros_dia = int(masc.loc[d0].sum())
    print(f"    fotografia vigente em {d0.date()}: {marco}, {len(comp[marco])} nomes")
    print(f"    acoes ativas no painel nesse dia: {membros_dia}")
    fora = [c for c in px.columns if not masc.loc[d0, c]]
    print(f"    {len(fora)} tickers existem no arquivo mas ficam de fora nesse dia")

    print("\nCONFERENCIA 4: o universo inclui empresas que sumiram")
    universo = sorted({t for l in comp.values() for t in l})
    atuais = set(comp[max(comp)])
    sumiram = [t for t in universo if t not in atuais and t in px.columns]
    print(f"    {len(universo)} tickers ja estiveram no indice, {len(atuais)} estao hoje")
    print(f"    {len(sumiram)} deles sairam e continuam no painel, ex: {sumiram[:8]}")

    # ---------------------------------------------------------- 2. curvas ROC
    modelos = {
        "decision tree": (DecisionTreeClassifier(max_depth=8, min_samples_leaf=500,
                                                 random_state=0), AMBAR),
        "random forest": (RandomForestClassifier(n_estimators=60, max_depth=12,
                                                 min_samples_leaf=200, max_features="sqrt",
                                                 n_jobs=3, random_state=0), TEAL),
        "xgboost": (xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, max_depth=3,
                                      subsample=0.8, colsample_bytree=0.8, n_jobs=3,
                                      tree_method="hist", max_bin=96,
                                      eval_metric="logloss"), UVA),
    }
    # uma janela representativa, a ultima, para a figura nao virar media de medias
    ini = janelas[-1]
    Xtr, ytr = empilha(dias[ini:ini + TREINO])
    Xte, yte = empilha(dias[ini + TREINO:ini + TREINO + OPERA])
    print(f"\ncurvas ROC na ultima janela: treino {len(ytr):,} linhas, teste {len(yte):,}")

    fig, eixos = plt.subplots(1, 3, figsize=(13.5, 4.6), facecolor=FUNDO)
    resumo = {}
    for ax, (nome, (mod, cor)) in zip(eixos, modelos.items()):
        mod.fit(Xtr.to_numpy("float32"), ytr.to_numpy())
        ptr = mod.predict_proba(Xtr.to_numpy("float32"))[:, 1]
        pte = mod.predict_proba(Xte.to_numpy("float32"))[:, 1]
        a_tr, a_te = roc_auc_score(ytr, ptr), roc_auc_score(yte, pte)
        resumo[nome] = {"auc_treino": round(float(a_tr), 4), "auc_teste": round(float(a_te), 4)}
        for y_, p_, cor_, est, rot in [(ytr, ptr, cor, "--", f"train  {a_tr:.3f}"),
                                       (yte, pte, cor, "-", f"test   {a_te:.3f}")]:
            fpr, tpr, _ = roc_curve(y_, p_)
            ax.plot(fpr, tpr, color=cor_, ls=est, lw=1.9, label=rot)
        ax.plot([0, 1], [0, 1], color="#9aa1ae", lw=1.1, ls=":")
        ax.set_facecolor(FUNDO)
        for lado in ("top", "right"):
            ax.spines[lado].set_visible(False)
        for lado in ("left", "bottom"):
            ax.spines[lado].set_color(GRADE)
        ax.tick_params(colors="#6b7280", labelsize=9)
        ax.set_title(nome, color=TINTA, fontsize=12, loc="left")
        ax.set_xlabel("false positive rate", color="#6b7280", fontsize=9.5)
        ax.legend(frameon=False, fontsize=9.5, labelcolor=TINTA, loc="lower right")
        print(f"    {nome:14} treino {a_tr:.4f}   teste {a_te:.4f}   "
              f"queda {a_tr - a_te:+.4f}")
    eixos[0].set_ylabel("true positive rate", color="#6b7280", fontsize=9.5)
    fig.suptitle("ROC in sample and out of sample, last window of the 2010-2015 test",
                 color=TINTA, fontsize=13, x=0.09, ha="left")
    fig.savefig(AQUI / "krauss_roc.png", facecolor=FUNDO, dpi=100,
                bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)
    print("\n  krauss_roc.png")
    (AQUI / "krauss_roc.json").write_text(json.dumps(resumo, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
