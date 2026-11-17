"""
semantic_search.py
==================
Recuperação semântica por embeddings densos.

Utiliza um modelo pré-treinado com arquitetura bi-encoder (Sentence
Transformers). Os embeddings dos documentos são calculados previamente e
armazenados em disco (arquivo .npy + lista de doc_ids em .json); para cada
consulta, gera-se o embedding e calcula-se a similaridade (cosseno, via
produto interno de vetores normalizados) com todos os documentos — busca
exaustiva, permitida porque a coleção é pequena (3.633 documentos).

Modelo padrão: sentence-transformers/all-MiniLM-L6-v2 (dimensão 384).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np

import config


class SemanticSearch:
    """Busca semântica com bi-encoder + similaridade de cosseno exaustiva."""

    def __init__(self, model_name: str = None) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name or config.SEMANTIC_MODEL
        self.model = SentenceTransformer(self.model_name)
        # Dimensão dos embeddings.
        self.dim = self.model.get_sentence_embedding_dimension()
        # Comprimento máximo de entrada do tokenizador.
        self.max_seq_length = getattr(self.model, "max_seq_length", None) or \
            self.model.tokenizer.model_max_length

        self.doc_ids: List[str] = []
        self.doc_embeddings: np.ndarray | None = None

        # Estatísticas de tempo.
        self.embedding_time: float = 0.0
        self.total_query_time: float = 0.0
        self.num_queries: int = 0

    # Embeddings de documentos
    def encode_documents(self, docs: dict[str, str], batch_size: int = None) -> None:
        """
        Calcula e armazena os embeddings de todos os documentos.

        `docs` é {doc_id: texto}. A ordem de `self.doc_ids` fica determinada
        pela ordem de iteração do dicionário (preservada na gravação em disco).
        """
        batch_size = batch_size or config.SEMANTIC_BATCH_SIZE
        self.doc_ids = list(docs.keys())
        texts = [docs[d] for d in self.doc_ids]

        start = time.perf_counter()
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=config.SEMANTIC_NORMALIZE,
            convert_to_numpy=True,
        )
        self.doc_embeddings = np.asarray(embeddings, dtype=np.float32)
        self.embedding_time = time.perf_counter() - start

    # Persistência (para não recalcular embeddings toda vez)
    def save_embeddings(self, embeddings_dir: Path) -> None:
        """Salva doc_ids (.json) e a matriz de embeddings (.npy)."""
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        ids_file = embeddings_dir / "doc_ids.json"
        mat_file = embeddings_dir / "doc_embeddings.npy"
        meta_file = embeddings_dir / "meta.json"

        with open(ids_file, "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)
        np.save(mat_file, self.doc_embeddings)
        with open(meta_file, "w", encoding="utf-8") as f:
            json.dump({"model": self.model_name, "dim": self.dim,
                       "normalized": config.SEMANTIC_NORMALIZE,
                       "embedding_time_sec": self.embedding_time}, f)

    def load_embeddings(self, embeddings_dir: Path) -> bool:
        """
        Carrega doc_ids e matriz de embeddings do disco. Retorna False se os
        arquivos não existirem.
        """
        ids_file = embeddings_dir / "doc_ids.json"
        mat_file = embeddings_dir / "doc_embeddings.npy"
        if not ids_file.exists() or not mat_file.exists():
            return False

        with open(ids_file, "r", encoding="utf-8") as f:
            self.doc_ids = json.load(f)
        self.doc_embeddings = np.load(mat_file)
        meta_file = embeddings_dir / "meta.json"
        if meta_file.exists():
            with open(meta_file, "r", encoding="utf-8") as f:
                self.embedding_time = json.load(f).get("embedding_time_sec", 0.0)
        return True

    # Consulta
    def search(self, query: str, top_k: int = None) -> List[Tuple[str, float]]:
        """
        Calcula a similaridade (cosseno) entre a consulta e todos os
        documentos e devolve os `top_k` como [(doc_id, score)].
        """
        top_k = top_k or config.TOP_K_RETRIEVAL
        start = time.perf_counter()

        q_emb = self.model.encode(
            [query],
            normalize_embeddings=config.SEMANTIC_NORMALIZE,
            convert_to_numpy=True,
        )[0]

        # Produto interno == cosseno, pois os embeddings estão normalizados.
        scores = self.doc_embeddings @ q_emb
        order = np.argsort(-scores)[:top_k]
        ranked = [(self.doc_ids[i], float(scores[i])) for i in order]

        self.total_query_time += time.perf_counter() - start
        self.num_queries += 1
        return ranked

    @property
    def avg_query_time(self) -> float:
        """Tempo médio de processamento por consulta (segundos)."""
        return self.total_query_time / self.num_queries if self.num_queries else 0.0
