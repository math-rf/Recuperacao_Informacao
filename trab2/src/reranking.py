"""
reranking.py
============
Reranking com cross-encoder.

O sistema híbrido (RRF) é usado como recuperador inicial. Os 50 primeiros
documentos do ranking híbrido são reordenados por um modelo mais preciso que
considera consulta e documento conjuntamente — um cross-encoder pré-treinado
(Sentence Transformers, modelo cross-encoder/ms-marco-MiniLM-L6-v2).

Os documentos abaixo da posição 50 permanecem, após o reranking, na mesma
ordem relativa do ranking híbrido, preservando um ranking completo para o
cálculo das métricas.
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import config


class Reranker:
    """Reranking neural com cross-encoder."""

    def __init__(self, model_name: str = None) -> None:
        from sentence_transformers import CrossEncoder

        self.model_name = model_name or config.RERANK_MODEL
        self.model = CrossEncoder(self.model_name, max_length=512)
        # Comprimento máximo de entrada.
        self.max_length = getattr(self.model, "max_length", None) or 512

        self.total_query_time: float = 0.0
        self.num_queries: int = 0

    def rerank(
        self,
        query: str,
        candidate_rank: List[Tuple[str, float]],
        corpus: Dict[str, str],
        top_k: int = None,
    ) -> List[Tuple[str, float]]:
        """
        Reordena os `top_k` primeiros documentos de `candidate_rank` com o
        cross-encoder; o restante permanece na ordem original.

        Parâmetros
        ----------
        query           : texto da consulta.
        candidate_rank  : ranking inicial [(doc_id, score)] (ex.: híbrido).
        corpus          : {doc_id: texto} para obter o texto dos documentos.
        top_k           : quantos documentos reordenar (default RERANK_TOP_K).

        Retorno
        -------
        Novo ranking completo [(doc_id, score)], onde o score das posições
        reordenadas é o do cross-encoder e o das demais é o original.
        """
        top_k = top_k or config.RERANK_TOP_K
        start = time.perf_counter()

        head = candidate_rank[:top_k]
        tail = candidate_rank[top_k:]

        # Pares (consulta, documento) para o cross-encoder.
        pairs = [(query, corpus.get(doc_id, "")) for doc_id, _ in head]
        ce_scores = self.model.predict(
            pairs, batch_size=config.RERANK_BATCH_SIZE, show_progress_bar=False
        )

        # Reordena o "head" pelo score do cross-encoder (decrescente).
        head_reranked = sorted(
            zip(head, ce_scores), key=lambda x: x[1], reverse=True
        )
        new_head = [(doc_id, float(score)) for (doc_id, _), score in head_reranked]

        self.total_query_time += time.perf_counter() - start
        self.num_queries += 1

        return new_head + tail

    @property
    def avg_query_time(self) -> float:
        """Tempo médio de reranking por consulta (segundos)."""
        return self.total_query_time / self.num_queries if self.num_queries else 0.0
