"""
analysis.py
===========
Funções auxiliares para as análises por consulta e de falhas (requisitos 7,
10 e 14 do enunciado). Estas funções não executam experimentos; apenas
organizam resultados já calculados para a redação do relatório.

  * requisito 7  : comparação lexical x semântico (diferença de NDCG@10)
  * requisito 10 : análise do reranking (ganhos, estabilidade e pioras)
  * requisito 14 : análise de falhas ponta a ponta
"""

from __future__ import annotations

from typing import Dict, List, Tuple


# ----------------------------------------------------------------------
# Requisito 7 — comparação lexical x semântico
# ----------------------------------------------------------------------
def ndcg_difference(
    metrics_a: Dict[str, Dict[str, float]],
    metrics_b: Dict[str, Dict[str, float]],
    metric: str = "NDCG",
) -> List[Tuple[str, float]]:
    """
    Calcula a diferença (b - a) da métrica `metric` para cada consulta.
    Retorna lista ordenada por diferença decrescente [(query_id, diff)].
    """
    diffs = []
    for qid in metrics_a:
        if qid in metrics_b:
            diffs.append((qid, metrics_b[qid][metric] - metrics_a[qid][metric]))
    return sorted(diffs, key=lambda x: x[1], reverse=True)


def extreme_queries(
    diffs: List[Tuple[str, float]], n: int = 3
) -> Tuple[List[Tuple[str, float]], List[Tuple[str, float]]]:
    """
    Devolve as `n` consultas com maior vantagem de B (positivas) e as `n`
    com maior vantagem de A (negativas).
    """
    positives = [d for d in diffs if d[1] > 0][:n]
    negatives = sorted([d for d in diffs if d[1] < 0], key=lambda x: x[1])[:n]
    return positives, negatives


def format_top_k(
    query_id: str,
    ranking: List[Tuple[str, float]],
    qrels: Dict[str, int],
    top_k: int = 5,
) -> str:
    """
    Formata o Top-k de um método para o relatório, marcando relevância.
    Retorna uma string multilinha pronta para exibição/tabela.
    """
    lines = []
    for i, (doc_id, score) in enumerate(ranking[:top_k], start=1):
        rel = qrels.get(doc_id, 0)
        flag = "*" if rel > 0 else " "
        lines.append(f"{i:2d}. {doc_id:15s} score={score:.4f} rel={rel} {flag}")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Requisito 10 — análise do reranking
# ----------------------------------------------------------------------
def rerank_analysis(
    hybrid_ndcg: Dict[str, float],
    reranked_ndcg: Dict[str, float],
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Identifica os casos pedidos no requisito 10:
      * "gains"   : dois maiores ganhos de NDCG@10 após o reranking
      * "stable"  : caso de pouca/nula alteração (|Δ| mínimo)
      * "losses"  : caso em que o reranking PIOROU o NDCG@10 (maior queda)

    Retorna {"gains": [...], "stable": [...], "losses": [...]}, cada item
    [(query_id, delta)].
    """
    deltas = []
    for qid in hybrid_ndcg:
        if qid in reranked_ndcg:
            deltas.append((qid, reranked_ndcg[qid] - hybrid_ndcg[qid]))

    gains = sorted([d for d in deltas if d[1] > 0], key=lambda x: x[1], reverse=True)[:2]
    losses = sorted([d for d in deltas if d[1] < 0], key=lambda x: x[1])[:1]
    # "Pouca ou nenhuma alteração" = menor |Δ| dentre todos.
    stable = sorted(deltas, key=lambda x: abs(x[1]))[:1]
    return {"gains": gains, "stable": stable, "losses": losses}


# ----------------------------------------------------------------------
# Requisito 14 — análise de falhas ponta a ponta
# ----------------------------------------------------------------------
def moved_documents(
    ranking_a: List[str], ranking_b: List[str], n: int = 10
) -> List[Tuple[str, int, int]]:
    """
    Devolve os documentos que mudaram de posição entre dois rankings
    (dentro das `n` primeiras posições), como (doc_id, pos_a, pos_b),
    ordenados pela maior mudança absoluta.
    """
    pos_a = {d: i for i, d in enumerate(ranking_a[:n], start=1)}
    pos_b = {d: i for i, d in enumerate(ranking_b[:n], start=1)}
    moved = []
    for doc_id in set(pos_a) | set(pos_b):
        pa = pos_a.get(doc_id)
        pb = pos_b.get(doc_id)
        if pa is not None and pb is not None and pa != pb:
            moved.append((doc_id, pa, pb))
    return sorted(moved, key=lambda x: abs(x[2] - x[1]), reverse=True)
