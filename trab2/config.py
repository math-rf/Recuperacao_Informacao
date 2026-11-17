"""
Configurações do projeto. Todos os parâmetros dos experimentos estão aqui.
"""

from pathlib import Path

# Caminhos
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "nfcorpus"
OUTPUT_DIR = BASE_DIR / "outputs"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
INDEX_DIR = BASE_DIR / "index"

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_DEV_FILE = DATA_DIR / "qrels" / "dev.tsv"
QRELS_TEST_FILE = DATA_DIR / "qrels" / "test.tsv"

NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"

# Pré-processamento: minúsculas, pontuação, stopwords e Porter stemmer
PREPROCESSING = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_digits": False,
    "remove_stopwords": True,
    "stem": True,
}

# BM25
BM25_K1 = 1.5
BM25_B = 0.75
TOP_K_RETRIEVAL = 100

# Correção ortográfica
SPELL = {
    "kgram_size": 3,
    "jaccard_threshold": 0.2,
    "max_candidates": 50,
    "max_edit_distance": 3,
}

# Bi-encoder
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SEMANTIC_BATCH_SIZE = 128
SEMANTIC_NORMALIZE = True

# Fusão híbrida (RRF): k escolhido no dev entre os valores abaixo
RRF_K_CANDIDATES = [10, 60, 100]

# Cross-encoder
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_TOP_K = 50
RERANK_BATCH_SIZE = 32

# LLM do RAG (HuggingFace Transformers, execução local)
LLM_MODEL = "Qwen/Qwen3-1.7B"  # quantizado em 4 bits (NF4) para caber em 4 GB de VRAM
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 512
LLM_SEED = 42
LLM_LANGUAGE = "português"
RAG_TOP_K = 5

# Corte das métricas
METRIC_CUTOFF = 10
