"""
hybrid_search.py
================
Busca híbrida por Reciprocal Rank Fusion (RRF) — requisito 8 do enunciado.

RRF combina os rankings de dois ou mais sistemas sem comparar diretamente
scores de escalas diferentes. O score de um documento d é:

    RRF(d) = Σ_s 1 / (k + rank_s(d))

onde rank_s(d) é a posição de d (começando em 1) no ranking do sistema s;
documentos ausentes de um dos rankings não recebem contribuição daquele
sistema. O parâmetro k é ajustado no conjunto dev (k ∈ {10, 60, 100}).

São fundidos os 100 primeiros documentos de cada sistema.
"""

from __future__ import annotations

from typing import Dict, List, Tuple


def reciprocal_rank_fusion(
    rankings: List[List[Tuple[str, float]]], k: int = 60
) -> List[Tuple[str, float]]:
    """
    Aplica RRF a uma lista de rankings.

    Parâmetros
    ----------
    rankings : list de listas de (doc_id, score). Cada ranking já vem truncado
               (ex.: top-100 de cada sistema).
    k        : parâmetro do RRF.

    Retorno
    -------
    Lista de (doc_id, rrf_score) em ordem decrescente de score RRF.
    """
    fused: Dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _score) in enumerate(ranking, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + 1.0 / (k + rank)

    return sorted(fused.items(), key=lambda kv: kv[1], reverse=True)


def tune_k(
    lexical_rankings: Dict[str, List[Tuple[str, float]]],
    semantic_rankings: Dict[str, List[Tuple[str, float]]],
    qrels: Dict[str, Dict[str, int]],
    k_candidates: List[int],
    metric_fn,
    cutoff: int = 10,
) -> Tuple[int, Dict[int, float]]:
    """
    Ajusta o parâmetro k do RRF no conjunto dev, avaliando NDCG@10 médio
    para cada k candidato. Retorna (melhor_k, {k: ndcg_medio}).

    `metric_fn(ranked_doc_ids, qrels_for_query, cutoff)` deve devolver um
    dicionário contendo "NDCG" (função `evaluation.evaluate_query`).
    """
    results: Dict[int, float] = {}
    for k in k_candidates:
        ndcgs = []
        for qid, lex_rank in lexical_rankings.items():
            if qid not in qrels:
                continue
            sem_rank = semantic_rankings.get(qid, [])
            fused = reciprocal_rank_fusion([lex_rank, sem_rank], k=k)
            doc_ids = [d for d, _ in fused]
            metrics = metric_fn(doc_ids, qrels[qid], cutoff)
            ndcgs.append(metrics["NDCG"])
        results[k] = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0

    best_k = max(results, key=results.get)
    return best_k, results
