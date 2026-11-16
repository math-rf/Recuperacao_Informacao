"""
06_analysis.py
==============
Gera artefatos de análise por consulta para o relatório (requisitos 7, 10 e 14).

Executa (requer os passos 03 e, opcionalmente, 05):

    python scripts/06_analysis.py

Saídas:
  * outputs/analysis_lexical_vs_semantic.txt  -> 6 consultas (Top-3 de cada lado)
  * outputs/analysis_reranking.txt            -> casos de ganho/estabilidade/perda
  * console                                    -> resumo + orientação p/ requisito 14
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.analysis import (  # noqa: E402
    extreme_queries,
    format_top_k,
    moved_documents,
    ndcg_difference,
    rerank_analysis,
)
from src.data_loader import load_all  # noqa: E402


def load_rankings_dict(csv_path):
    """Lê um results_*.csv e devolve {query_id: [(doc_id, score), ...]}."""
    df = pd.read_csv(csv_path)
    out = {}
    for qid, sub in df.groupby("query_id", sort=True):
        sub = sub.sort_values("rank")
        out[qid] = [(row.doc_id, row.score) for row in sub.itertuples()]
    return out


def main() -> None:
    corpus, queries, qrels_dev, qrels_test = load_all(
        config.CORPUS_FILE, config.QUERIES_FILE,
        config.QRELS_DEV_FILE, config.QRELS_TEST_FILE,
    )
    metrics = pd.read_csv(config.OUTPUT_DIR / "metrics_per_query.csv")

    # ------------------------------------------------------------------
    # Requisito 7 — lexical x semântico
    # ------------------------------------------------------------------
    print("=== Requisito 7: comparação lexical x semântico (NDCG@10) ===")
    lex_ndcg = metrics[metrics.model == "lexical"].set_index("query_id")["NDCG10"]
    sem_ndcg = metrics[metrics.model == "semantic"].set_index("query_id")["NDCG10"]

    diffs = ndcg_difference(
        {q: {"NDCG": v} for q, v in lex_ndcg.items()},
        {q: {"NDCG": v} for q, v in sem_ndcg.items()},
    )
    sem_adv, lex_adv = extreme_queries(diffs, n=3)

    lex_rank = load_rankings_dict(config.OUTPUT_DIR / "results_lexical.csv")
    sem_rank = load_rankings_dict(config.OUTPUT_DIR / "results_semantic.csv")

    lines = ["ANÁLISE LEXICAL x SEMÂNTICO (diferença de NDCG@10)\n"]
    lines.append(f"Maior vantagem do SEMÂNTICO (3 consultas): {[q for q, _ in sem_adv]}")
    lines.append(f"Maior vantagem do LEXICAL   (3 consultas): {[q for q, _ in lex_adv]}\n")

    for qid, diff in sem_adv + sorted(lex_adv, key=lambda x: x[1]):
        lines.append("=" * 78)
        lines.append(f"Consulta: {qid}  (ΔNDCG@10 semântico-lexical = {diff:+.4f})")
        lines.append(f"Texto: {queries[qid]}")
        lines.append("-" * 78)
        lines.append("Top-5 LEXICAL:")
        lines.append(format_top_k(qid, lex_rank[qid], qrels_test[qid]))
        lines.append("Top-5 SEMÂNTICO:")
        lines.append(format_top_k(qid, sem_rank[qid], qrels_test[qid]))
        lines.append("")

    with open(config.OUTPUT_DIR / "analysis_lexical_vs_semantic.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  -> {config.OUTPUT_DIR / 'analysis_lexical_vs_semantic.txt'}")

    # ------------------------------------------------------------------
    # Requisito 10 — análise do reranking
    # ------------------------------------------------------------------
    print("\n=== Requisito 10: análise do reranking ===")
    hyb_ndcg = metrics[metrics.model == "hybrid"].set_index("query_id")["NDCG10"].to_dict()
    rer_ndcg = metrics[metrics.model == "reranking"].set_index("query_id")["NDCG10"].to_dict()
    cases = rerank_analysis(hyb_ndcg, rer_ndcg)

    hyb_rank = load_rankings_dict(config.OUTPUT_DIR / "results_hybrid.csv")
    rer_rank = load_rankings_dict(config.OUTPUT_DIR / "results_reranking.csv")

    rlines = ["ANÁLISE DO RERANKING (ΔNDCG@10 = reranking - híbrido)\n"]
    for qid, delta in cases["gains"]:
        rlines.append("=" * 78)
        rlines.append(f"GANHO ({qid}): ΔNDCG@10 = {delta:+.4f}")
        rlines.append(f"Texto: {queries[qid]}")
        moved = moved_documents([d for d, _ in hyb_rank[qid]],
                                [d for d, _ in rer_rank[qid]], n=10)
        rlines.append("Principais documentos que mudaram de posição (doc_id, antes, depois):")
        for doc_id, pa, pb in moved:
            rlines.append(f"  {doc_id}: {pa} -> {pb} (Δ={pb - pa:+d})")
        rlines.append("")

    for qid, delta in cases["stable"]:
        rlines.append("=" * 78)
        rlines.append(f"POUCA/NENHUMA ALTERAÇÃO ({qid}): ΔNDCG@10 = {delta:+.4f}")
        rlines.append(f"Texto: {queries[qid]}")
        moved = moved_documents([d for d, _ in hyb_rank[qid]],
                                [d for d, _ in rer_rank[qid]], n=10)
        rlines.append("Documentos que mudaram de posição:")
        for doc_id, pa, pb in moved:
            rlines.append(f"  {doc_id}: {pa} -> {pb} (Δ={pb - pa:+d})")
        rlines.append("")

    for qid, delta in cases["losses"]:
        rlines.append("=" * 78)
        rlines.append(f"PIORA ({qid}): ΔNDCG@10 = {delta:+.4f}")
        rlines.append(f"Texto: {queries[qid]}")
        moved = moved_documents([d for d, _ in hyb_rank[qid]],
                                [d for d, _ in rer_rank[qid]], n=10)
        rlines.append("Principais documentos que mudaram de posição:")
        for doc_id, pa, pb in moved:
            rlines.append(f"  {doc_id}: {pa} -> {pb} (Δ={pb - pa:+d})")
        rlines.append("")

    with open(config.OUTPUT_DIR / "analysis_reranking.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(rlines))
    print(f"  -> {config.OUTPUT_DIR / 'analysis_reranking.txt'}")

    # ------------------------------------------------------------------
    # Requisito 14 — orientação p/ análise de falhas do RAG
    # ------------------------------------------------------------------
    print("\n=== Requisito 14: análise de falhas ponta a ponta ===")
    print("  Após preencher as notas em rag_experiment.csv, compare a média das")
    print("  notas A x B por consulta para escolher: (i) RAG melhorou claramente,")
    print("  (ii) pouca diferença, (iii) RAG não ajudou/prejudicou. Em seguida,")
    print("  investigue em qual estágio ocorreu o problema (lexical, semântica,")
    print("  fusão, reranking, seleção de contexto ou geração).")


if __name__ == "__main__":
    main()
