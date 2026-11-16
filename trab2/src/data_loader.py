"""
data_loader.py
==============
Leitura dos arquivos padronizados do BEIR/NFCorpus:

  * corpus.jsonl   -> documentos (campos `_id`, `title`, `text`)
  * queries.jsonl  -> consultas (campos `_id`, `text`)
  * qrels/dev.tsv  -> julgamentos de relevância do conjunto dev
  * qrels/test.tsv -> julgamentos de relevância do conjunto test

O enunciado exige que `title` e `text` sejam concatenados para formar a
representação de cada documento em TODOS os sistemas (lexical, semântico e
reranking). Isso é feito na função `load_corpus`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_corpus(corpus_file: Path) -> Dict[str, str]:
    """
    Lê o corpus.jsonl e devolve um dicionário {doc_id: "title text"}.

    `title` e `text` são concatenados (com espaço) conforme exigido pelo
    enunciado. O doc_id usado é o campo `_id` do BEIR.
    """
    docs: Dict[str, str] = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj["_id"]
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            # Concatenação title + text (exigência do enunciado).
            docs[doc_id] = f"{title} {text}".strip()
    return docs


def load_queries(queries_file: Path) -> Dict[str, str]:
    """
    Lê o queries.jsonl e devolve {query_id: texto da consulta}.

    Obs.: queries.jsonl contém as consultas de TODOS os conjuntos
    (3.237 no total), e não apenas as do test. Quem define quais consultas
    pertencem a dev/test são os respectivos arquivos de qrels.
    """
    queries: Dict[str, str] = {}
    with open(queries_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            queries[obj["_id"]] = obj["text"].strip()
    return queries


def load_qrels(qrels_file: Path) -> Dict[str, Dict[str, int]]:
    """
    Lê um arquivo de qrels (TSV) e devolve:
        {query_id: {doc_id: grau_de_relevância}}.

    Formato BEIR: `query-id \t corpus-id \t score`, com cabeçalho opcional
    na primeira linha. O grau de relevância é preservado como inteiro
    (necessário para o NDCG com ganho linear).
    """
    qrels: Dict[str, Dict[str, int]] = {}
    with open(qrels_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 3:
                continue
            qid, doc_id, score = parts[0], parts[1], parts[2]
            # Ignora a linha de cabeçalho, se existir.
            if qid.lower() in {"query-id", "query_id", "queryid"}:
                continue
            try:
                rel = int(score)
            except ValueError:
                # Alguns qrels usam score float (ex.: 1.0); tenta float -> int.
                rel = int(float(score))
            # Guarda apenas julgamentos com relevância > 0? Não: preserva todos,
            # mas a convenção do BEIR é só listar relevantes. Mantemos todos.
            qrels.setdefault(qid, {})[doc_id] = rel
    return qrels


def get_query_ids_from_qrels(qrels: Dict[str, Dict[str, int]]) -> List[str]:
    """Devolve a lista (ordenada) de query_ids presentes em um qrels."""
    return sorted(qrels.keys())


def load_all(
    corpus_file: Path,
    queries_file: Path,
    qrels_dev_file: Path,
    qrels_test_file: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """
    Carrega todos os dados de uma vez.

    Retorna (corpus, queries, qrels_dev, qrels_test).
    """
    corpus = load_corpus(corpus_file)
    queries = load_queries(queries_file)
    qrels_dev = load_qrels(qrels_dev_file)
    qrels_test = load_qrels(qrels_test_file)
    return corpus, queries, qrels_dev, qrels_test
