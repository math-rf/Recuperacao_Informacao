"""
Leitura dos arquivos do BEIR/NFCorpus: corpus.jsonl, queries.jsonl e os qrels
de dev e test (formato TSV: query-id, corpus-id, score).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


def load_corpus(corpus_file: Path) -> Dict[str, str]:
    """Devolve {doc_id: título + texto}."""
    docs: Dict[str, str] = {}
    with open(corpus_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            title = obj.get("title", "") or ""
            text = obj.get("text", "") or ""
            docs[obj["_id"]] = f"{title} {text}".strip()
    return docs


def load_queries(queries_file: Path) -> Dict[str, str]:
    """Devolve {query_id: texto}. O arquivo contém consultas de todos os conjuntos."""
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
    """Devolve {query_id: {doc_id: grau de relevância}}."""
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
            if qid.lower() in {"query-id", "query_id", "queryid"}:
                continue
            qrels.setdefault(qid, {})[doc_id] = int(float(score))
    return qrels


def get_query_ids_from_qrels(qrels: Dict[str, Dict[str, int]]) -> List[str]:
    """IDs das consultas presentes em um arquivo de qrels, em ordem."""
    return sorted(qrels.keys())


def load_all(
    corpus_file: Path,
    queries_file: Path,
    qrels_dev_file: Path,
    qrels_test_file: Path,
) -> Tuple[Dict[str, str], Dict[str, str], Dict[str, Dict[str, int]], Dict[str, Dict[str, int]]]:
    """Devolve (corpus, consultas, qrels_dev, qrels_test)."""
    return (
        load_corpus(corpus_file),
        load_queries(queries_file),
        load_qrels(qrels_dev_file),
        load_qrels(qrels_test_file),
    )
