"""
lexical_search.py
=================
Busca lexical com BM25 (requisito 3 do enunciado).

Implementação do BM25 (Okapi BM25) SOBRE o índice invertido construído do
zero. A pontuação de um documento d para uma consulta q é:

    score(d, q) = Σ_{t ∈ q} IDF(t) * [ tf(t,d) * (k1 + 1) ] /
                                      [ tf(t,d) + k1 * (1 - b + b * |d|/avgdl) ]

    IDF(t) = ln( (N - df(t) + 0.5) / (df(t) + 0.5) + 1 )

onde:
    tf(t,d) : frequência bruta do termo t no documento d
    df(t)   : número de documentos que contêm t
    N       : total de documentos
    |d|     : comprimento do documento d (nº de tokens)
    avgdl   : comprimento médio dos documentos
    k1, b   : parâmetros livres (defaults clássicos: k1 = 1.5, b = 0.75)

A consulta acessa apenas as posting lists dos termos presentes na consulta
(cumprindo a restrição de não percorrer toda a coleção a cada consulta).
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

        # Para medir tempo médio de processamento de consultas.
        self.total_query_time: float = 0.0
        self.num_queries: int = 0

    def _idf(self, df: int) -> float:
        """Calcula o IDF de um termo a partir da sua document frequency."""
        return math.log((self.N - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: List[str]) -> Dict[str, float]:
        """
        Calcula o score BM25 de TODOS os documentos que contêm ao menos um
        termo da consulta. Retorna {doc_id: score}.

        Usa um acumulador (`scores`) alimentado percorrendo apenas as posting
        lists dos termos da consulta — nunca a coleção inteira.
        """
        scores: Dict[str, float] = {}
        # Reusa o IDF por termo (evita recalcular para termos repetidos).
        for term in set(query_tokens):
            postings = self.index.get_postings(term)
            if not postings:
                continue
            df = self.index.get_doc_freq(term)
            idf = self._idf(df)

            for doc_id, tf in postings.items():
                dl = self.index.get_doc_length(doc_id)
                # Normalização pelo comprimento do documento.
                denom = tf + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                term_score = idf * (tf * (self.k1 + 1.0)) / denom
                scores[doc_id] = scores.get(doc_id, 0.0) + term_score

        return scores

    def search(self, query_tokens: List[str], top_k: int = None) -> List[Tuple[str, float]]:
        """
        Devolve os `top_k` documentos ordenados por score decrescente como
        lista de (doc_id, score). Mede e acumula o tempo de processamento.
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        start = time.perf_counter()
        scores = self.score(query_tokens)
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        self.total_query_time += time.perf_counter() - start
        self.num_queries += 1
        return ranked[:top_k]

    @property
    def avg_query_time(self) -> float:
        """Tempo médio de processamento por consulta (segundos)."""
        return self.total_query_time / self.num_queries if self.num_queries else 0.0
