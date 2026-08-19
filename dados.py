# -*- coding: utf-8 -*-
"""Monta os dois caches que todo o resto le: a composicao do indice e os precos.

Sem isto o `krauss.py` nao roda numa copia limpa do repositorio, porque ele LE o parquet e
nunca o cria. Aqui esta a parte que o notebook faz nos passos 2 e 3, extraida para um script,
para que os motores funcionem sem abrir o Jupyter.

  membership.json   39 fotografias semestrais da lista do S&P 500, tiradas do historico de
                    revisoes da Wikipedia. E o que remove o vies de sobrevivencia: a lista de
                    hoje aplicada a 2007 apaga toda empresa que quebrou ou foi comprada.
  precos.parquet    fechamento ajustado de todo ticker que ja apareceu em alguma fotografia,
                    inclusive os que nao negociam mais.

A primeira rodada demora, quase tudo baixando. Depois os dois arquivos ficam no disco e todo
o resto le deles.

  py -3 dados.py
  py -3 dados.py --ate 2026-08-19     # painel mais longo, para a varredura de 15 janelas
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import pandas as pd
import requests

AQUI = Path(__file__).parent
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

CABECALHO = {"User-Agent": "Mozilla/5.0 (research)"}
WIKI = "https://en.wikipedia.org/w/api.php"
COMPOSICAO = AQUI / "membership.json"
PRECOS = AQUI / "precos.parquet"


def membros_em(data: str) -> list[str]:
    """A lista do S&P 500 como estava viva naquela data, nao como esta hoje."""
    p = {"action": "query", "prop": "revisions", "titles": "List of S&P 500 companies",
         "rvlimit": 1, "rvstart": f"{data}T00:00:00Z", "rvdir": "older",
         "rvprop": "ids|timestamp", "format": "json", "formatversion": 2}
    for tentativa in range(4):
        resp = requests.get(WIKI, params=p, headers=CABECALHO, timeout=90)
        try:
            j = resp.json()
            break
        except ValueError:                        # limite de taxa, recua e tenta de novo
            time.sleep(3 * (tentativa + 1))
    else:
        return []
    rev = j["query"]["pages"][0].get("revisions")
    if not rev:
        return []
    html = requests.get("https://en.wikipedia.org/w/index.php",
                        params={"oldid": rev[0]["revid"]},
                        headers=CABECALHO, timeout=90).text
    try:
        grandes = [t for t in pd.read_html(io.StringIO(html)) if t.shape[0] > 300]
    except ValueError:
        return []
    if not grandes:
        return []
    col = [c for c in grandes[0].columns
           if str(c).lower().startswith(("symbol", "ticker"))]
    if not col:
        return []
    # fillna importa: algumas revisoes trazem NaN, que sobrevive ao astype(str) e quebra o isascii
    tick = (grandes[0][col[0]].astype("string").fillna("").str.strip().str.upper()
            .str.replace(".", "-", regex=False))          # BRK.B vira BRK-B, formato do Yahoo
    return sorted({x for x in tick.tolist()
                   if isinstance(x, str) and x.isascii()
                   and 1 <= len(x) <= 6 and x.replace("-", "").isalpha()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desde", default="2007-01-01")
    ap.add_argument("--ate", default="2015-12-31",
                    help="fim do painel; use 2026-08-19 para a varredura completa")
    a = ap.parse_args()

    import yfinance as yf

    # ------------------------------------------------------------------ composicao
    comp = json.loads(COMPOSICAO.read_text(encoding="utf-8")) if COMPOSICAO.exists() else {}
    querido = [f"{ano}-{mes}-01" for ano in range(2007, 2027) for mes in ("01", "07")]
    faltando = [d for d in querido if d <= a.ate and d not in comp]
    if faltando:
        print(f"buscando {len(faltando)} fotografias na Wikipedia, uma a cada 1,2 s")
        for d in faltando:
            achou = membros_em(d)
            if achou:
                comp[d] = achou
                COMPOSICAO.write_text(json.dumps(comp, indent=1), encoding="utf-8")
                print(f"  {d}: {len(achou)} nomes")
            time.sleep(1.2)
    universo = sorted({t for l in comp.values() for t in l})
    print(f"composicao: {len(comp)} fotografias, {len(universo)} tickers ja no indice")

    # ------------------------------------------------------------------ precos
    if PRECOS.exists():
        px = pd.read_parquet(PRECOS)
        print(f"precos ja no disco: {px.shape[1]} tickers, "
              f"{px.index[0].date()} a {px.index[-1].date()}")
        if str(px.index[-1].date()) >= a.ate[:10] or px.shape[1] >= len(universo) * 0.6:
            print("  nada a fazer. apague precos.parquet para baixar de novo")
            return 0

    print(f"baixando {len(universo)} tickers do Yahoo, em lotes de 120")
    partes = []
    for i in range(0, len(universo), 120):
        lote = universo[i:i + 120]
        d = yf.download(lote, start=a.desde, end=a.ate, auto_adjust=True, progress=False)
        if isinstance(d.columns, pd.MultiIndex):
            d = d["Close"]
        partes.append(d)
        print(f"  {i + len(lote)}/{len(universo)}")
    px = pd.concat(partes, axis=1).sort_index()
    px = px.loc[:, ~px.columns.duplicated()].dropna(axis=1, how="all")
    px.to_parquet(PRECOS)
    print(f"precos: {px.shape[1]} de {len(universo)} tickers com historico "
          f"({px.shape[1] / len(universo):.0%})")
    print(f"  {px.index[0].date()} a {px.index[-1].date()}, {px.shape[0]} pregoes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
