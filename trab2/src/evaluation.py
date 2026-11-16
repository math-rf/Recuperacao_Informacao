"""
evaluation.py
=============
Métricas de avaliação da recuperação (requisito 6 do enunciado).

Métricas implementadas (corte k = 10 por padrão):
  * Precision@10 : #relevantes_no_top10 / 10
  * Recall@10    : #relevantes_no_top10 / total_de_relevantes_da_consulta
  * AP@10        : (Σ_{posições k<=10 com relevante} Precision@k) / total_relevantes
                   -> convenção do trec_eval (denominador = total
                      de relevantes da consulta nos qrels)
  * MAP@10       : média dos AP@10 sobre todas as consultas
  * NDCG@10      : DCG com ganho LINEAR (ganho = grau de relevância), ideal
                   calculado a partir de TODOS os julgamentos da consulta
                   (não apenas dos documentos recuperados)

Convenção de relevância:
  * Para P/R/AP: documento é relevante se grau de relevância > 0.
  * Para NDCG: os graus fornecidos pelos qrels são preservados.

Todas as métricas também podem ser calculadas por consulta.
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple, Union


def _to_doc_ids(ranked: List[Union[str, Tuple[str, float]]]) -> List[str]:
    """
    Normaliza uma lista de doc_ids (str) OU de pares (doc_id, score) para uma
    lista apenas de doc_ids (str). Isso evita que, ao passar rankings no formato
    [(doc_id, score)], a tupla seja comparada contra o doc_id (string) nas
    métricas — um erro que zeraria todas as métricas silenciosamente.
    """
    return [item[0] if isinstance(item, (tuple, list)) else item for item in ranked]


def _relevant_set(qrels_for_query: Dict[str, int]) -> set:
    """Conjunto de doc_ids com grau de relevância > 0."""
    return {doc_id for doc_id, rel in qrels_for_query.items() if rel > 0}


def _total_relevant(qrels_for_query: Dict[str, int]) -> int:
    """Número total de documentos relevantes da consulta nos qrels."""
    return len(_relevant_set(qrels_for_query))


def precision_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Precision@k: fração de relevantes entre os k primeiros."""
    if k <= 0:
        return 0.0
    rel = _relevant_set(qrels)
    top = ranked_doc_ids[:k]
    return sum(1 for d in top if d in rel) / k


def recall_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """Recall@k: relevantes no top-k / total de relevantes da consulta."""
    total = _total_relevant(qrels)
    if total == 0:
        return 0.0
    rel = _relevant_set(qrels)
    top = ranked_doc_ids[:k]
    return sum(1 for d in top if d in rel) / total


def average_precision_at_k(
    ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10
) -> float:
    """
    AP@k (Average Precision truncada em k).

    AP@k = ( Σ_{i=1..k, d_i relevante} Precision@i ) / total_relevantes.

    O denominador é o total de relevantes da consulta nos qrels (convenção do
    trec_eval), e NÃO o mínimo entre k e esse total.
    """
    total = _total_relevant(qrels)
    if total == 0:
        return 0.0
    rel = _relevant_set(qrels)
    num_hits = 0
    prec_sum = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        if doc_id in rel:
            num_hits += 1
            prec_sum += num_hits / i  # Precision@i
    return prec_sum / total


def dcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """
    DCG@k com ganho LINEAR (ganho = grau de relevância) e desconto logarítmico
    na base 2: DCG = Σ_{i=1..k} rel_i / log2(i+1).
    """
    dcg = 0.0
    for i, doc_id in enumerate(ranked_doc_ids[:k], start=1):
        rel = qrels.get(doc_id, 0)
        if rel > 0:
            dcg += rel / math.log2(i + 1)
    return dcg


def idcg_at_k(qrels: Dict[str, int], k: int = 10) -> float:
    """
    IDCG@k: DCG do ranking ideal. Calculado a partir de TODOS os julgamentos
    da consulta (não apenas dos recuperados), ordenando por relevância
    decrescente e truncando em k.
    """
    # Ordena todos os graus de relevância em ordem decrescente.
    sorted_rels = sorted((rel for rel in qrels.values() if rel > 0), reverse=True)[:k]
    return sum(rel / math.log2(i + 1) for i, rel in enumerate(sorted_rels, start=1))


def ndcg_at_k(ranked_doc_ids: List[str], qrels: Dict[str, int], k: int = 10) -> float:
    """NDCG@k = DCG@k / IDCG@k. Retorna 0.0 se IDCG for 0."""
    ideal = idcg_at_k(qrels, k)
    if ideal == 0.0:
        return 0.0
    return dcg_at_k(ranked_doc_ids, qrels, k) / ideal


def evaluate_query(
    ranked_doc_ids: List[Union[str, Tuple[str, float]]], qrels: Dict[str, int], k: int = 10
) -> Dict[str, float]:
    """Calcula P@k, R@k, AP@k e NDCG@k para UMA consulta."""
    # Aceita tanto [doc_id, ...] quanto [(doc_id, score), ...].
    ranked_doc_ids = _to_doc_ids(ranked_doc_ids)
    return {
        "P": precision_at_k(ranked_doc_ids, qrels, k),
        "R": recall_at_k(ranked_doc_ids, qrels, k),
        "AP": average_precision_at_k(ranked_doc_ids, qrels, k),
        "NDCG": ndcg_at_k(ranked_doc_ids, qrels, k),
    }


def evaluate_ranking(
    rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int = 10
) -> Dict[str, Dict[str, float]]:
    """
    Avalia um dicionário {query_id: [doc_ids ordenados]} contra os qrels.

    Retorna {query_id: {"P": .., "R": .., "AP": .., "NDCG": ..}}.
    """
    per_query: Dict[str, Dict[str, float]] = {}
    for qid, ranked in rankings.items():
        if qid in qrels:
            per_query[qid] = evaluate_query(ranked, qrels[qid], k)
    return per_query


def mean_metrics(per_query: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Calcula as médias globais de cada métrica (MAP@k = média dos AP@k)."""
    n = len(per_query)
    if n == 0:
        return {"P": 0.0, "R": 0.0, "AP": 0.0, "NDCG": 0.0}
    sums = {"P": 0.0, "R": 0.0, "AP": 0.0, "NDCG": 0.0}
    for m in per_query.values():
        for key in sums:
            sums[key] += m[key]
    return {key: val / n for key, val in sums.items()}
