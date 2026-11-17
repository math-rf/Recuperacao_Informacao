"""
inverted_index.py
=================
Índice invertido construído do zero, com dicionários nativos do Python.

Estruturas de dados utilizadas (apenas dicionários nativos do Python):

  * `self.postings`   : dict  termo -> {doc_id: tf}
                        posting list como dicionário doc_id -> frequência do
                        termo no documento (TF bruto, não normalizado).
  * `self.doc_lengths`: dict  doc_id -> número total de tokens do documento
  * `self.doc_freq`   : dict  termo -> DF (número de documentos que contêm o termo)
  * `self.num_docs`   : int   total de documentos indexados
  * `self.total_tokens`: int  total de tokens da coleção
  * `self.avg_doc_len`: float comprimento médio dos documentos

O índice armazena TF, DF, tamanho dos documentos e estatísticas globais,
que são exatamente as informações necessárias ao BM25. NÃO é permitido
percorrer toda a coleção a cada consulta; aqui a consulta acessa apenas as
posting lists dos termos presentes na consulta.

Também são registradas as estatísticas de construção: nº de termos do
vocabulário, nº total de postings e tempo de construção.
"""

from __future__ import annotations

import time
from typing import Dict, Iterable, List, Tuple


class InvertedIndex:
    """Índice invertido simples e eficiente para busca lexical."""

    def __init__(self) -> None:
        # termo -> {doc_id: tf}
        self.postings: Dict[str, Dict[str, int]] = {}
        # doc_id -> número de tokens
        self.doc_lengths: Dict[str, int] = {}
        # Ordem de inserção dos documentos (usada para completar rankings com score 0).
        self.doc_order: List[str] = []
        # termo -> document frequency
        self.doc_freq: Dict[str, int] = {}
        self.num_docs: int = 0
        self.total_tokens: int = 0
        self.avg_doc_len: float = 0.0

        # Estatísticas de construção.
        self.build_time: float = 0.0

    # Construção
    def add_document(self, doc_id: str, tokens: List[str]) -> None:
        """
        Indexa um documento a partir da sua lista de tokens pré-processados.

        `tokens` deve vir já pré-processado (lowercase, sem stopwords, com
        stemming, etc.), pois a indexação e a consulta precisam usar a MESMA
        representação.
        """
        # Comprimento do documento (número de tokens).
        length = len(tokens)
        self.doc_lengths[doc_id] = length
        self.doc_order.append(doc_id)
        self.total_tokens += length
        self.num_docs += 1

        # Conta a frequência de cada termo no documento.
        tf_counter: Dict[str, int] = {}
        for term in tokens:
            tf_counter[term] = tf_counter.get(term, 0) + 1

        # Atualiza as posting lists.
        for term, tf in tf_counter.items():
            self.postings.setdefault(term, {})[doc_id] = tf
            self.doc_freq[term] = self.doc_freq.get(term, 0) + 1

    def build(self, tokenized_docs: Iterable[Tuple[str, List[str]]]) -> "InvertedIndex":
        """
        Constrói o índice a partir de (doc_id, tokens) e registra o tempo.
        """
        start = time.perf_counter()
        for doc_id, tokens in tokenized_docs:
            self.add_document(doc_id, tokens)
        # Finaliza a estatística de comprimento médio.
        if self.num_docs > 0:
            self.avg_doc_len = self.total_tokens / self.num_docs
        self.build_time = time.perf_counter() - start
        return self

    # Consulta
    def get_postings(self, term: str) -> Dict[str, int]:
        """Devolve a posting list {doc_id: tf} de um termo (vazio se ausente)."""
        return self.postings.get(term, {})

    def get_doc_freq(self, term: str) -> int:
        """Devolve a document frequency (DF) de um termo."""
        return self.doc_freq.get(term, 0)

    def get_doc_length(self, doc_id: str) -> int:
        """Devolve o comprimento (nº de tokens) de um documento."""
        return self.doc_lengths.get(doc_id, 0)

    # Estatísticas
    @property
    def vocabulary_size(self) -> int:
        """Número de termos distintos no vocabulário."""
        return len(self.postings)

    @property
    def total_postings(self) -> int:
        """
        Número total de postings = soma, sobre todos os termos, do nº de
        documentos em que o termo aparece (i.e., soma dos DF).
        """
        return sum(self.doc_freq.values())

    def stats(self) -> Dict[str, float | int]:
        """Devolve um dicionário com as estatísticas do índice."""
        return {
            "num_docs": self.num_docs,
            "vocabulary_size": self.vocabulary_size,
            "total_postings": self.total_postings,
            "total_tokens": self.total_tokens,
            "avg_doc_len": self.avg_doc_len,
            "build_time_sec": self.build_time,
        }

    # Persistência em pickle
    def save(self, path) -> None:
        """Serializa o índice em disco usando pickle."""
        import pickle

        path = str(path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "InvertedIndex":
        """Carrega um índice serializado com `save`."""
        import pickle

        with open(str(path), "rb") as f:
            return pickle.load(f)
