"""
05_rag_experiment.py
====================
Experimento RAG: sem RAG (A) x com RAG (B) — requisitos 11, 12 e 13.

Executa (requer ter rodado o passo 03, que gera results_reranking.csv):

    python scripts/05_rag_experiment.py

Procedimento:
  1. Seleciona as 10 consultas fixas do test (seed 42, entre as terminadas em
     "?"). Confere se a lista bate com a do enunciado:
        PLAIN-68, PLAIN-2450, PLAIN-153, PLAIN-2850, PLAIN-2750,
        PLAIN-270, PLAIN-2470, PLAIN-2430, PLAIN-238, PLAIN-3342
  2. Para cada consulta, obtém o Top-5 após o reranking (contexto).
  3. Gera duas respostas com a MESMA LLM e MESMOS parâmetros:
       (A) sem RAG  : só a pergunta.
       (B) com RAG  : pergunta + Top-5 documentos (com doc_ids).
  4. Calcula Recall@5, Precision@5 e Hit@5 do contexto (recuperação).
  5. Salva rag_experiment.csv com colunas para as notas humanas (0/1/2).

Arquivo gerado:
    outputs/rag_experiment.csv
"""

import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_loader import load_all, get_query_ids_from_qrels  # noqa: E402
from src.rag import (  # noqa: E402
    TransformersLLM,
    build_context,
    build_prompt_no_rag,
    build_prompt_rag,
)

# Lista oficial do enunciado (para conferência).
EXPECTED_QUERIES = [
    "PLAIN-68", "PLAIN-2450", "PLAIN-153", "PLAIN-2850", "PLAIN-2750",
    "PLAIN-270", "PLAIN-2470", "PLAIN-2430", "PLAIN-238", "PLAIN-3342",
]


def select_rag_queries(test_ids, queries):
    """
    Reproduz exatamente o sorteio do enunciado (seed 42) entre as consultas
    do test cujo texto termina em "?".
    """
    ids = [q for q in sorted(test_ids) if queries[q].strip().endswith("?")]
    return random.Random(42).sample(ids, 10)


def retrieval_metrics(top_k_doc_ids, qrels_for_query):
    """Recall@5, Precision@5 e Hit@5 do contexto enviado à LLM."""
    relevant = {d for d, rel in qrels_for_query.items() if rel > 0}
    total_rel = len(relevant)
    hits = [d for d in top_k_doc_ids if d in relevant]
    recall5 = len(hits) / total_rel if total_rel else 0.0
    precision5 = len(hits) / len(top_k_doc_ids) if top_k_doc_ids else 0.0
    hit5 = 1 if hits else 0
    return recall5, precision5, hit5


def main() -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Carregando dados ...")
    corpus, queries, qrels_dev, qrels_test = load_all(
        config.CORPUS_FILE, config.QUERIES_FILE,
        config.QRELS_DEV_FILE, config.QRELS_TEST_FILE,
    )
    test_ids = get_query_ids_from_qrels(qrels_test)

    rag_queries = select_rag_queries(test_ids, queries)
    print(f"      Consultas sorteadas (seed 42): {rag_queries}")
    assert rag_queries == EXPECTED_QUERIES, \
        f"Divergência na seleção: {rag_queries} != {EXPECTED_QUERIES}"
    print("      Lista confere com o enunciado.")

    print("[2/5] Carregando ranking reranqueado (results_reranking.csv) ...")
    rerank_df = pd.read_csv(config.OUTPUT_DIR / "results_reranking.csv")
    reranked = {}
    for qid in rag_queries:
        sub = rerank_df[rerank_df["query_id"] == qid].sort_values("rank")
        reranked[qid] = [(row.doc_id, row.score) for row in sub.itertuples()]

    print("[3/5] Preparando LLM ...")
    llm = TransformersLLM()
    print(f"      modelo={llm.model_name} temp={llm.temperature} "
          f"max_tokens={llm.max_tokens} seed={llm.seed}")

    print("[4/5] Gerando respostas (A e B) ...")
    rows = []
    for i, qid in enumerate(rag_queries, start=1):
        query = queries[qid]
        context = build_context(reranked[qid], corpus, top_k=config.RAG_TOP_K)
        context_ids = [c["doc_id"] for c in context]
        recall5, precision5, hit5 = retrieval_metrics(context_ids, qrels_test[qid])

        prompt_a = build_prompt_no_rag(query)
        prompt_b = build_prompt_rag(query, context)

        print(f"      [{i}/10] {qid} (A: sem RAG)")
        answer_a = llm.generate(prompt_a)
        print(f"      [{i}/10] {qid} (B: com RAG)")
        answer_b = llm.generate(prompt_b)

        # Condição A.
        rows.append({
            "query_id": qid, "query": query, "condition": "A",
            "context_doc_ids": "", "prompt": prompt_a, "answer": answer_a,
            "recall5": None, "precision5": None, "hit5": None,
            "fidelidade": "", "completude": "", "utilidade": "",
            "rastreabilidade": "N/A",
            "just_fidelidade": "", "just_completude": "",
            "just_utilidade": "", "just_rastreabilidade": "",
        })
        # Condição B.
        rows.append({
            "query_id": qid, "query": query, "condition": "B",
            "context_doc_ids": ";".join(context_ids),
            "prompt": prompt_b, "answer": answer_b,
            "recall5": recall5, "precision5": precision5, "hit5": hit5,
            "fidelidade": "", "completude": "", "utilidade": "",
            "rastreabilidade": "",
            "just_fidelidade": "", "just_completude": "",
            "just_utilidade": "", "just_rastreabilidade": "",
        })

    print("[5/5] Salvando rag_experiment.csv ...")
    df = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "rag_experiment.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print(f"      -> {out}")
    print("\nPróximo passo: preencher manualmente as notas (fidelidade, "
          "completude, utilidade, rastreabilidade) e justificativas no CSV, "
          "preferencialmente com as respostas embaralhadas (sem saber se é A ou B).")


if __name__ == "__main__":
    main()
