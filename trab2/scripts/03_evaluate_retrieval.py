"""
03_evaluate_retrieval.py
========================
Executa o pipeline de recuperação completo e salva os resultados padronizados.

Passo 3 do pipeline:

    python scripts/03_evaluate_retrieval.py

O que faz:
  1. Recuperação lexical (BM25) para dev e test.
  2. Recuperação semântica (bi-encoder) para dev e test.
  3. Ajuste do k do RRF no dev (k ∈ {10, 60, 100}) e fusão híbrida.
  4. Reranking (cross-encoder) do top-50 híbrido no test.
  5. Cálculo de P@10, R@10, AP@10 e NDCG@10 (todas as métricas por consulta).

Arquivos gerados em outputs/:
    results_lexical.csv, results_semantic.csv, results_hybrid.csv,
    results_reranking.csv, metrics_per_query.csv, retrieval_times.json
"""

import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_all, get_query_ids_from_qrels
from src.evaluation import evaluate_query, evaluate_ranking, mean_metrics
from src.hybrid_search import reciprocal_rank_fusion, tune_k
from src.inverted_index import InvertedIndex
from src.lexical_search import BM25
from src.preprocessing import ensure_nltk_data, preprocess
from src.reranking import Reranker
from src.semantic_search import SemanticSearch


# Helpers de saída
def save_ranking_csv(filename: str, rankings: dict, qrels: dict) -> None:
    """
    Salva um arquivo de ranking no formato padronizado:
    query_id, doc_id, rank, score, relevance (top-100 por consulta).
    """
    rows = []
    for qid in sorted(rankings):
        for rank, (doc_id, score) in enumerate(rankings[qid][:config.TOP_K_RETRIEVAL], start=1):
            rows.append({
                "query_id": qid,
                "doc_id": doc_id,
                "rank": rank,
                "score": score,
                "relevance": qrels.get(qid, {}).get(doc_id, 0),
            })
    df = pd.DataFrame(rows)
    path = config.OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    print(f"      -> {path} ({len(df)} linhas)")


def main() -> None:
    ensure_nltk_data()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Carregando dados ...")
    corpus, queries, qrels_dev, qrels_test = load_all(
        config.CORPUS_FILE, config.QUERIES_FILE,
        config.QRELS_DEV_FILE, config.QRELS_TEST_FILE,
    )
    dev_ids = get_query_ids_from_qrels(qrels_dev)
    test_ids = get_query_ids_from_qrels(qrels_test)
    print(f"      dev: {len(dev_ids)} consultas | test: {len(test_ids)} consultas")

    # Índice invertido + BM25
    print("[2/6] Busca lexical (BM25) ...")
    index_path = config.INDEX_DIR / "inverted_index.pkl"
    if index_path.exists():
        index = InvertedIndex.load(index_path)
    else:
        print("      Índice não encontrado; execute 02_build_index.py antes.")
        return
    bm25 = BM25(index)

    lex_dev = {qid: bm25.search(preprocess(queries[qid])) for qid in dev_ids}
    lex_test = {qid: bm25.search(preprocess(queries[qid])) for qid in test_ids}
    print(f"      índice: {index.vocabulary_size} termos, "
          f"{index.total_postings} postings, "
          f"construção {index.build_time:.3f}s, "
          f"consulta média {bm25.avg_query_time * 1000:.2f}ms")

    # Recuperação semântica (bi-encoder)
    print("[3/6] Recuperação semântica (bi-encoder) ...")
    sem = SemanticSearch(config.SEMANTIC_MODEL)
    if not sem.load_embeddings(config.EMBEDDINGS_DIR):
        print(f"      Gerando embeddings de {len(corpus)} documentos "
              f"(modelo {config.SEMANTIC_MODEL}) ...")
        sem.encode_documents(corpus)
        sem.save_embeddings(config.EMBEDDINGS_DIR)
    else:
        print("      Embeddings carregados do disco.")
    print(f"      dim={sem.dim}, max_seq_length={sem.max_seq_length}, "
          f"tempo de embedding {sem.embedding_time:.1f}s")

    sem_dev = {qid: sem.search(queries[qid]) for qid in dev_ids}
    sem_test = {qid: sem.search(queries[qid]) for qid in test_ids}
    print(f"      consulta média semântica {sem.avg_query_time * 1000:.2f}ms")

    # Busca híbrida (RRF) — ajuste de k no dev
    print("[4/6] Busca híbrida (RRF) — tuning de k no dev ...")
    best_k, k_results = tune_k(
        lex_dev, sem_dev, qrels_dev, config.RRF_K_CANDIDATES,
        metric_fn=evaluate_query,
    )
    print(f"      NDCG@10 médio por k no dev: {k_results}")
    print(f"      Melhor k = {best_k}")

    hyb_test = {
        qid: reciprocal_rank_fusion([lex_test[qid], sem_test[qid]], k=best_k)
        for qid in test_ids
    }

    # Reranking (cross-encoder) no test
    print("[5/6] Reranking (cross-encoder) no test ...")
    reranker = Reranker(config.RERANK_MODEL)
    print(f"      modelo {config.RERANK_MODEL}, max_length={reranker.max_length}")
    rerank_test = {}
    for qid in test_ids:
        rerank_test[qid] = reranker.rerank(queries[qid], hyb_test[qid], corpus)
    print(f"      reranking médio por consulta {reranker.avg_query_time * 1000:.2f}ms")

    # Avaliação
    print("[6/6] Avaliação (P@10, R@10, AP@10, NDCG@10) ...")
    systems = {
        "lexical": lex_test,
        "semantic": sem_test,
        "hybrid": hyb_test,
        "reranking": rerank_test,
    }
    metrics_per_model = {}
    for name, rankings in systems.items():
        per_query = evaluate_ranking(rankings, qrels_test, config.METRIC_CUTOFF)
        metrics_per_model[name] = per_query
        mean = mean_metrics(per_query)
        print(f"      {name:10s}: P@10={mean['P']:.4f} R@10={mean['R']:.4f} "
              f"MAP@10={mean['AP']:.4f} NDCG@10={mean['NDCG']:.4f}")

    # Salvando rankings e métricas
    print("\nSalvando CSVs padronizados ...")
    save_ranking_csv("results_lexical.csv", lex_test, qrels_test)
    save_ranking_csv("results_semantic.csv", sem_test, qrels_test)
    save_ranking_csv("results_hybrid.csv", hyb_test, qrels_test)
    save_ranking_csv("results_reranking.csv", rerank_test, qrels_test)

    # metrics_per_query.csv
    mrows = []
    for model, per_query in metrics_per_model.items():
        for qid, m in per_query.items():
            mrows.append({
                "query_id": qid,
                "model": model,
                "P10": m["P"],
                "R10": m["R"],
                "AP10": m["AP"],
                "NDCG10": m["NDCG"],
            })
    pd.DataFrame(mrows).to_csv(config.OUTPUT_DIR / "metrics_per_query.csv", index=False)
    print(f"      -> {config.OUTPUT_DIR / 'metrics_per_query.csv'}")

    # Tempos de execução.
    times = {
        "index_build_sec": index.build_time,
        "index_vocabulary_size": index.vocabulary_size,
        "index_total_postings": index.total_postings,
        "bm25_avg_query_ms": bm25.avg_query_time * 1000,
        "semantic_embedding_sec": sem.embedding_time,
        "semantic_avg_query_ms": sem.avg_query_time * 1000,
        "semantic_model": config.SEMANTIC_MODEL,
        "semantic_dim": sem.dim,
        "semantic_max_seq_length": sem.max_seq_length,
        "rerank_avg_query_ms": reranker.avg_query_time * 1000,
        "rerank_model": config.RERANK_MODEL,
        "rrf_k": best_k,
        "rrf_k_dev_ndcg": k_results,
    }
    with open(config.OUTPUT_DIR / "retrieval_times.json", "w", encoding="utf-8") as f:
        json.dump(times, f, indent=2)
    print(f"      -> {config.OUTPUT_DIR / 'retrieval_times.json'}")
    print("\nConcluído.")


if __name__ == "__main__":
    main()
