"""
Pacote `src` do Trabalho Prático 2 (SCC0282 - Recuperação de Informação).

Módulos:
  data_loader     -> leitura de corpus.jsonl, queries.jsonl e qrels (BEIR/NFCorpus)
  preprocessing   -> tokenização, normalização, stopwords e stemming
  inverted_index  -> índice invertido implementado do zero
  lexical_search  -> BM25 sobre o índice invertido
  evaluation      -> P@10, R@10, AP@10/MAP@10 e NDCG@10
  spell_correction-> correção ortográfica (k-gramas + Jaccard + distância de edição)
  semantic_search -> recuperação por embeddings densos (bi-encoder)
  hybrid_search   -> Reciprocal Rank Fusion (RRF)
  reranking       -> reordenação com cross-encoder
  rag             -> Retrieval-Augmented Generation via LLM (Qwen/Transformers)
  analysis        -> análises por consulta e de falhas
"""

__all__ = [
    "data_loader",
    "preprocessing",
    "inverted_index",
    "lexical_search",
    "evaluation",
    "spell_correction",
    "semantic_search",
    "hybrid_search",
    "reranking",
    "rag",
    "analysis",
]
