"""
07_plot_results.py
==================
Gera os gráficos em report/figures/, lendo os valores dos CSVs
e JSONs de outputs/ (nenhum valor escrito à mão).
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FIG_DIR = config.BASE_DIR / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

# 1) Métricas médias dos quatro sistemas
met = pd.read_csv(config.OUTPUT_DIR / "metrics_per_query.csv")
order = [("lexical", "Lexical (BM25)"), ("semantic", "Semântico"),
         ("hybrid", "Híbrido (RRF)"), ("reranking", "Híbrido + reranking")]
means = met.groupby("model")[["P10", "R10", "AP10", "NDCG10"]].mean()
labels = [lab for _, lab in order]
metric_names = {"P10": "P@10", "R10": "R@10", "AP10": "MAP@10", "NDCG10": "NDCG@10"}
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
w = 0.2
for i, (col, name) in enumerate(metric_names.items()):
    vals = [means.loc[key, col] for key, _ in order]
    ax.bar([xi + i * w - 1.5 * w for xi in range(len(order))], vals, w, label=name, color=colors[i])
ax.set_xticks(range(len(order)))
ax.set_xticklabels(labels, rotation=15, ha="right")
ax.set_ylabel("Métrica (@10)")
ax.set_ylim(0, 0.40)
ax.legend(ncol=4, frameon=False, fontsize=7.5)
ax.grid(axis="y", alpha=0.3, linestyle=":")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_metricas.png", dpi=200)
plt.close(fig)

# 2) Tuning de k do RRF no dev
times = json.load(open(config.OUTPUT_DIR / "retrieval_times.json", encoding="utf-8"))
kdev = times["rrf_k_dev_ndcg"]
ks = list(kdev.keys())
vals = [kdev[k] for k in ks]
best = ks[vals.index(max(vals))]
fig, ax = plt.subplots(figsize=(3.6, 2.2))
bars = ax.bar(ks, vals, width=0.5, color="#4C72B0")
bars[ks.index(best)].set_color("#C44E52")
ax.set_xlabel("k")
ax.set_ylabel("NDCG@10 (dev)")
lo, hi = min(vals), max(vals)
ax.set_ylim(lo - 0.01, hi + 0.01)
ax.grid(axis="y", alpha=0.3, linestyle=":")
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_rrf_k.png", dpi=200)
plt.close(fig)

# 3) RAG: notas médias (condições A e B), só com notas preenchidas
rag = pd.read_csv(config.OUTPUT_DIR / "rag_experiment.csv")
crit = ["fidelidade", "completude", "utilidade", "rastreabilidade"]
crit_lab = ["Fidelidade", "Completude", "Utilidade", "Rastreabilidade"]
def mean_note(cond, c):
    s = pd.to_numeric(rag[rag.condition == cond][c], errors="coerce").dropna()
    return s.mean() if len(s) else None
A = [mean_note("A", c) for c in crit]
B = [mean_note("B", c) for c in crit]
if any(v is not None for v in A + B):
    fig, ax = plt.subplots(figsize=(5.2, 2.8))
    for i in range(len(crit)):
        if A[i] is not None:
            ax.bar(i - 0.18, A[i], 0.34, color="#55A868", label="Sem RAG (A)" if i == 0 else None)
        else:
            ax.text(i - 0.18, 0.1, "N/A", ha="center", fontsize=7, color="gray")
        if B[i] is not None:
            ax.bar(i + 0.18, B[i], 0.34, color="#4C72B0", label="Com RAG (B)" if i == 0 else None)
    ax.set_xticks(range(len(crit)))
    ax.set_xticklabels(crit_lab)
    ax.set_ylabel("Nota média (0–2)")
    ax.set_ylim(0, 2.1)
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_rag_ab.png", dpi=200)
    plt.close(fig)
else:
    print("RAG: nenhuma nota preenchida em rag_experiment.csv; fig_rag_ab.png não gerado.")

print("Gráficos gerados em:", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print("  -", f.name)
