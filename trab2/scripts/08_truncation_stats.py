"""
08_truncation_stats.py
======================
Conta quantos textos excedem o comprimento máximo de entrada dos modelos
(truncamento), para o bi-encoder e o cross-encoder.

Não executa LLM e não recalcula rankings: apenas tokeniza. Requer os dados
(data/nfcorpus) e os rankings de outputs/results_hybrid.csv.

    python scripts/08_truncation_stats.py

Saída:
    outputs/truncation_stats.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_corpus, load_queries


def bi_encoder_stats(corpus: dict) -> dict:
    """Documentos mais longos que o limite do bi-encoder (todos os documentos)."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.SEMANTIC_MODEL)
    max_len = model.max_seq_length
    tokenizer = model.tokenizer

    lengths = [
        len(tokenizer(text, add_special_tokens=True)["input_ids"])
        for text in corpus.values()
    ]
    truncated = sum(1 for n in lengths if n > max_len)
    return {
        "model": config.SEMANTIC_MODEL,
        "max_seq_length": max_len,
        "documents_total": len(lengths),
        "documents_truncated": truncated,
        "percent_truncated": round(100 * truncated / len(lengths), 2),
        "median_tokens": int(pd.Series(lengths).median()),
        "max_tokens": max(lengths),
    }


def cross_encoder_stats(corpus: dict, queries: dict) -> dict:
    """Pares (consulta, documento) do top-50 híbrido no test que excedem o limite."""
    from sentence_transformers import CrossEncoder

    model = CrossEncoder(config.RERANK_MODEL, max_length=512)
    max_len = getattr(model, "max_length", None) or 512
    tokenizer = model.tokenizer

    hyb = pd.read_csv(config.OUTPUT_DIR / "results_hybrid.csv")
    test_ids = set(
        pd.read_csv(config.OUTPUT_DIR / "metrics_per_query.csv")
        .query_id.unique()
    )
    hyb = hyb[hyb.query_id.isin(test_ids) & (hyb["rank"] <= config.RERANK_TOP_K)]

    lengths = []
    for row in hyb.itertuples():
        enc = tokenizer(queries[row.query_id], corpus.get(row.doc_id, ""),
                        add_special_tokens=True)
        lengths.append(len(enc["input_ids"]))

    truncated = sum(1 for n in lengths if n > max_len)
    return {
        "model": config.RERANK_MODEL,
        "max_length": max_len,
        "pairs_total": len(lengths),
        "pairs_truncated": truncated,
        "percent_truncated": round(100 * truncated / len(lengths), 2),
        "median_tokens": int(pd.Series(lengths).median()),
        "max_tokens": max(lengths),
        "source": f"outputs/results_hybrid.csv, rank <= {config.RERANK_TOP_K}, consultas do test",
    }


def main() -> None:
    corpus = load_corpus(config.CORPUS_FILE)
    queries = load_queries(config.QUERIES_FILE)

    print("[1/2] Bi-encoder (todos os documentos) ...")
    bi = bi_encoder_stats(corpus)
    print(f"      {bi['documents_truncated']}/{bi['documents_total']} documentos "
          f"truncados ({bi['percent_truncated']}%) com limite {bi['max_seq_length']}")

    print("[2/2] Cross-encoder (pares do top-50 híbrido, test) ...")
    ce = cross_encoder_stats(corpus, queries)
    print(f"      {ce['pairs_truncated']}/{ce['pairs_total']} pares truncados "
          f"({ce['percent_truncated']}%) com limite {ce['max_length']}")

    out = config.OUTPUT_DIR / "truncation_stats.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"bi_encoder": bi, "cross_encoder": ce}, f, indent=2, ensure_ascii=False)
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
