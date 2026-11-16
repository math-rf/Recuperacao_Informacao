"""
config.py
=========
Configurações globais do Trabalho Prático 2 (SCC0282 - Recuperação de Informação).

Todas as decisões experimentais ficam concentradas aqui, de modo que a
reprodução dos experimentos depende apenas deste arquivo + dos dados + das
bibliotecas listadas no requirements.txt.

Decisões adotadas (justificadas no relatório):
  * Baseline lexical ............ BM25 (k1=1.5, b=0.75, valores usuais da literatura)
  * Modelo de embeddings ........ sentence-transformers/all-MiniLM-L6-v2 (bi-encoder)
  * Modelo de reranking ......... cross-encoder/ms-marco-MiniLM-L6-v2
  * Fusão híbrida ............... Reciprocal Rank Fusion (RRF), k ajustado no dev
  * LLM (RAG) ................... local (HuggingFace Transformers, Qwen)
"""

from pathlib import Path

# --------------------------------------------------------------
# Caminhos do projeto
# --------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "nfcorpus"          # dataset BEIR/NFCorpus
OUTPUT_DIR = BASE_DIR / "outputs"                  # CSVs padronizados + rankings
EMBEDDINGS_DIR = BASE_DIR / "embeddings"           # embeddings .npy e ids .json
INDEX_DIR = BASE_DIR / "index"                     # índice invertido serializado (opcional)

CORPUS_FILE = DATA_DIR / "corpus.jsonl"
QUERIES_FILE = DATA_DIR / "queries.jsonl"
QRELS_DEV_FILE = DATA_DIR / "qrels" / "dev.tsv"
QRELS_TEST_FILE = DATA_DIR / "qrels" / "test.tsv"

# URL oficial do BEIR para download do NFCorpus (corpus + queries + qrels).
NFCORPUS_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nfcorpus.zip"

# --------------------------------------------------------------
# Pré-processamento
# --------------------------------------------------------------
PREPROCESSING = {
    "lowercase": True,        # converter para minúsculas
    "remove_punctuation": True,
    "remove_digits": False,   # números podem importar em domínio médico
    "remove_stopwords": True, # stopwords do inglês (NLTK)
    "stem": True,             # Porter stemmer (NLTK)
    "lemmatize": False,       # usa stemming como padrão
}

# --------------------------------------------------------------
# Busca lexical (BM25)
# --------------------------------------------------------------
BM25_K1 = 1.5   # parâmetro k1 (saturação da frequência do termo)
BM25_B = 0.75   # parâmetro b (normalização pelo comprimento do documento)

# Quantos documentos cada sistema devolve no ranking usado na fusão.
TOP_K_RETRIEVAL = 100

# --------------------------------------------------------------
# Tolerância a erros (correção ortográfica)
# --------------------------------------------------------------
SPELL = {
    "kgram_size": 3,          # tamanho dos k-gramas de caracteres
    "jaccard_threshold": 0.2, # limiar do coeficiente de Jaccard na filtragem
    "max_candidates": 50,     # máx. de candidatos considerados por termo
    "max_edit_distance": 3,   # distância de edição máxima aceitável
}

# --------------------------------------------------------------
# Recuperação semântica (bi-encoder)
# --------------------------------------------------------------
SEMANTIC_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Alternativa orientada a QA (caso se queira comparar no relatório):
#   "sentence-transformers/multi-qa-MiniLM-L6-cos-v1"
SEMANTIC_BATCH_SIZE = 128     # tamanho do lote ao gerar embeddings
SEMANTIC_NORMALIZE = True     # normaliza embeddings (similaridade = cosseno)

# --------------------------------------------------------------
# Busca híbrida (RRF)
# --------------------------------------------------------------
RRF_K_CANDIDATES = [10, 60, 100]   # valores de k testados no conjunto dev
RRF_K = 60                          # valor escolhido após tuning no dev (será preenchido)

# --------------------------------------------------------------
# Reranking (cross-encoder)
# --------------------------------------------------------------
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RERANK_TOP_K = 50   # os 50 primeiros do ranking híbrido são reordenados
RERANK_BATCH_SIZE = 32

# --------------------------------------------------------------
# RAG (LLM local via HuggingFace Transformers)
# --------------------------------------------------------------
# Modelo causal (id de repositório HuggingFace), baixado na primeira execução.
#   Qwen/Qwen2.5-1.5B-Instruct  -> ~1.5B, ~3 GB de RAM (bom equilíbrio)
#   Qwen/Qwen2.5-0.5B-Instruct  -> ~0.5B, ~1 GB de RAM (mais leve, se faltar memória)
LLM_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

LLM_TEMPERATURE = 0.0                     # temperatura 0 p/ reprodutibilidade
LLM_MAX_TOKENS = 512
LLM_SEED = 42
LLM_LANGUAGE = "Portuguese"               # idioma das respostas geradas

# Número de documentos enviados como contexto no RAG (Top-5 após reranking).
RAG_TOP_K = 5

# --------------------------------------------------------------
# Avaliação
# --------------------------------------------------------------
METRIC_CUTOFF = 10   # todas as métricas são @10 (P10, R10, AP10/MAP10, NDCG10)
