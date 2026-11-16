"""
07_plot_results.py
==================
Gera os gráficos do relatório (figures/). Requer matplotlib.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

FIG_DIR = Path(__file__).resolve().parent.parent / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# ----------------------------------------------------------------------
# 1) Métricas dos 4 sistemas (P@10, R@10, MAP@10, NDCG@10)
# ----------------------------------------------------------------------
sistemas = ["Lexical (BM25)", "Semântico", "Híbrido (RRF)", "Híbrido + reranking"]
metricas = {
    "P@10":     [0.2412, 0.2433, 0.2601, 0.2508],
    "R@10":     [0.1551, 0.1550, 0.1740, 0.1674],
    "MAP@10":   [0.1225, 0.1105, 0.1365, 0.1388],
    "NDCG@10":  [0.3302, 0.3160, 0.3562, 0.3562],
}
x = range(len(sistemas))
w = 0.2
colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
fig, ax = plt.subplots(figsize=(6.5, 3.0))
for i, (nome, vals) in enumerate(metricas.items()):
    ax.bar([xi + i*w - 1.5*w for xi in x], vals, w, label=nome, color=colors[i])
ax.set_xticks(list(x))
ax.set_xticklabels(sistemas, rotation=15, ha="right")
ax.set_ylabel("Métrica (@10)")
ax.set_ylim(0, 0.40)
ax.legend(ncol=4, frameon=False, fontsize=7.5)
ax.grid(axis="y", alpha=0.3, linestyle=":")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_metricas.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 2) Tuning de k do RRF no dev
# ----------------------------------------------------------------------
ks = ["10", "60", "100"]
ndcg = [0.3326, 0.3304, 0.3309]
fig, ax = plt.subplots(figsize=(3.6, 2.2))
bars = ax.bar(ks, ndcg, width=0.5, color="#4C72B0")
bars[0].set_color("#C44E52")
ax.set_xlabel("k")
ax.set_ylabel("NDCG@10 (dev)")
ax.set_ylim(0.30, 0.34)
ax.grid(axis="y", alpha=0.3, linestyle=":")
for b, v in zip(bars, ndcg):
    ax.text(b.get_x() + b.get_width()/2, v + 0.001, f"{v:.4f}", ha="center", fontsize=8)
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_rrf_k.png", dpi=200)
plt.close(fig)

# ----------------------------------------------------------------------
# 3) RAG: sem RAG (A) x com RAG (B)
# ----------------------------------------------------------------------
crit = ["Fidelidade", "Completude", "Utilidade", "Rastreabilidade"]
A = [1.40, 1.90, 1.80, None]   # rastreabilidade A = N/A
B = [1.60, 1.60, 1.50, 1.30]
x = range(len(crit))
fig, ax = plt.subplots(figsize=(5.2, 2.8))
for i in x:
    if A[i] is not None:
        ax.bar(i - 0.18, A[i], 0.34, color="#55A868", label="Sem RAG (A)" if i == 0 else None)
    ax.bar(i + 0.18, B[i], 0.34, color="#4C72B0", label="Com RAG (B)" if i == 0 else None)
ax.set_xticks(list(x))
ax.set_xticklabels(crit)
ax.set_ylabel("Nota média (0–2)")
ax.set_ylim(0, 2.1)
ax.legend(frameon=False, fontsize=8)
ax.grid(axis="y", alpha=0.3, linestyle=":")
ax.text(3, 0.25, "A: N/A", ha="center", fontsize=7, color="gray")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig_rag_ab.png", dpi=200)
plt.close(fig)

print("Gráficos gerados em:", FIG_DIR)
for f in sorted(FIG_DIR.glob("*.png")):
    print("  -", f.name)
