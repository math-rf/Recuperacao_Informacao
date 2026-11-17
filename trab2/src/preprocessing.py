"""
Pré-processamento de documentos e consultas.

Ordem das etapas: minúsculas, remoção de pontuação, tokenização por espaço,
remoção de stopwords (NLTK, inglês) e stemming de Porter (NLTK).
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import List

import config


@lru_cache(maxsize=None)
def _english_stopwords() -> frozenset:
    from nltk.corpus import stopwords

    return frozenset(stopwords.words("english"))


@lru_cache(maxsize=None)
def _porter_stemmer():
    from nltk.stem import PorterStemmer

    return PorterStemmer()


def ensure_nltk_data() -> None:
    """Baixa a lista de stopwords do NLTK, se ainda não estiver disponível."""
    import nltk

    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)


def preprocess(text: str, cfg: dict | None = None) -> List[str]:
    """Converte um texto em lista de tokens processados."""
    cfg = cfg or config.PREPROCESSING

    if cfg.get("lowercase", True):
        text = text.lower()

    if cfg.get("remove_punctuation", True):
        pattern = r"[^a-z0-9 ]" if not cfg.get("remove_digits", False) else r"[^a-z ]"
        text = re.sub(pattern, " ", text)

    tokens = text.split()

    if cfg.get("remove_stopwords", True):
        stop = _english_stopwords()
        tokens = [t for t in tokens if t not in stop]

    if cfg.get("stem", True):
        stemmer = _porter_stemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens
