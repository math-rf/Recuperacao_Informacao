"""
preprocessing.py
================
Pré-processamento de documentos e consultas (requisito 1 do enunciado).

Etapas implementadas (todas controláveis via config.PREPROCESSING):
  1. Lowercasing                       -> minúsculas
  2. Remoção de pontuação              -> expressão regular [^a-z0-9 ]
  3. Tokenização                       -> split por espaço (após normalização)
  4. Remoção de stopwords              -> lista NLTK do inglês
  5. Stemming (Porter) ou Lemmatização -> NLTK

Bibliotecas adotadas: NLTK (stopwords, PorterStemmer, WordNetLemmatizer).
A tokenização usa regex simples por robustez; o NLTK é usado apenas para os
recursos lexicais (stopwords/stemmer/lematizador), que são baixados na
primeira execução via `ensure_nltk_data()`.
"""

from __future__ import annotations

import re
from typing import List

import config


def ensure_nltk_data() -> None:
    """
    Garante que os recursos do NLTK necessários estejam baixados.
    Chamar uma vez no início do pipeline.

    Baixa apenas o que é realmente usado: `stopwords` (sempre) e, somente se a
    lematização estiver ativa no config, `wordnet` e `omw-1.4`. O stemmer
    (Porter) e a tokenização por regex não exigem download.
    """
    import nltk

    needed = ["stopwords"]
    if config.PREPROCESSING.get("lemmatize", False):
        needed += ["wordnet", "omw-1.4"]

    for resource in needed:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)


def _tokenize(text: str) -> List[str]:
    """Divide o texto normalizado em tokens (semântica de `str.split`)."""
    return text.split()


def preprocess(text: str, cfg: dict | None = None) -> List[str]:
    """
    Aplica o pipeline de pré-processamento ao `text` e devolve a lista de
    tokens finais.

    Parâmetros
    ----------
    text : str
        Texto bruto (título + conteúdo, ou a consulta).
    cfg  : dict | None
        Dicionário de configuração (usa config.PREPROCESSING se None).

    Retorno
    -------
    list[str] : tokens pré-processados (já com stemming/lematização, se ativo).
    """
    cfg = cfg or config.PREPROCESSING

    # 1) Minúsculas.
    if cfg.get("lowercase", True):
        text = text.lower()

    # 2) Remove pontuação (e, opcionalmente, dígitos).
    if cfg.get("remove_punctuation", True):
        # Mantém apenas letras minúsculas, espaços e (se permitido) dígitos.
        keep_digits = not cfg.get("remove_digits", False)
        pattern = r"[^a-z0-9 ]" if keep_digits else r"[^a-z ]"
        text = re.sub(pattern, " ", text)

    # 3) Tokenização.
    tokens = _tokenize(text)

    # 4) Remoção de stopwords.
    if cfg.get("remove_stopwords", True):
        from nltk.corpus import stopwords

        _stop = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in _stop]

    # 5) Stemming ou lematização.
    if cfg.get("lemmatize", False):
        from nltk.stem import WordNetLemmatizer

        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]
    elif cfg.get("stem", True):
        from nltk.stem import PorterStemmer

        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return tokens
