"""
lexical_search.py
=================
Busca lexical com BM25 sobre o índice invertido.

    score(q, d) = Σ_{t ∈ q} IDF(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 - b + b·|d|/avgdl))
    IDF(t)      = ln((N - df(t) + 0,5) / (df(t) + 0,5) + 1)

Só os documentos das posting lists dos termos da consulta recebem score.
Documentos sem nenhum termo em comum têm score 0; para que o ranking tenha
sempre `TOP_K_RETRIEVAL` documentos, ele é completado com
documentos de score 0 na ordem de inserção do índice.
"""

from __future__ import annotations

import math
import time
from typing import Dict, List, Tuple

import config
from src.inverted_index import InvertedIndex


class BM25:
    """BM25 sobre um índice invertido."""

    def __init__(self, index: InvertedIndex, k1: float = None, b: float = None) -> None:
        self.index = index
        self.k1 = k1 if k1 is not None else config.BM25_K1
        self.b = b if b is not None else config.BM25_B
        self.N = index.num_docs
        self.avgdl = index.avg_doc_len or 1.0

        self.total_query_time: float = 0.0
        self.num_queries: int = 0

    def _idf(self, df: int) -> float:
        """IDF de um termo a partir da sua document frequency."""
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str]) -> Dict[str, float]:
        """Score BM25 dos documentos que contêm ao menos um termo da consulta."""
        scores: Dict[str, float] = {}
        for term in set(query_tokens):
            postings = self.index.get_postings(term)
            if not postings:
                continue
            idf = self._idf(self.index.get_doc_freq(term))
            for doc_id, tf in postings.items():
                dl = self.index.get_doc_length(doc_id)
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * (tf * (self.k1 + 1.0)) / denom
        return scores

    def search(self, query_tokens: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """Top-k documentos como [(doc_id, score)]. Mede o tempo de processamento."""
        top_k = top_k or config.TOP_K_RETRIEVAL
        start = time.perf_counter()
        scores = self.score(query_tokens)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]

        # Completa com documentos de score 0 até top_k. Percorre apenas o
        # necessário (no máximo top_k + |documentos pontuados| entradas).
        if len(ranked) < top_k:
            for doc_id in self.index.doc_order:
                if len(ranked) >= top_k:
                    break
                if doc_id not in scores:
                    ranked.append((doc_id, 0.0))

        self.total_query_time += time.perf_counter() - start
        self.num_queries += 1
        return ranked

    @property
    def avg_query_time(self) -> float:
        """Tempo médio de processamento por consulta (segundos)."""
        return self.total_query_time / self.num_queries if self.num_queries else 0.0
