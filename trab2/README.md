# Trabalho Prático 2 — Recuperação de Informação (SCC0282)

**Universidade de São Paulo — ICMC — Bacharelado em Sistemas de Informação**
Prof. Marcelo Manzato · 2º Semestre/2026

Sistema completo de Recuperação de Informação que combina **busca lexical**
(BM25 sobre índice invertido implementado do zero), **busca semântica**
(bi-encoder), **busca híbrida** (RRF), **reranking** (cross-encoder),
**tolerância a erros** (k-gramas + Jaccard + distância de edição) e **RAG**
(LLM local, Qwen), sobre o dataset **NFCorpus (BEIR)**.

**Integrantes:** Gustavo Cardozo de Moraes Moreira (5244057) · Matheus Rodrigues Ferreira (14762149).

---

## 1. Pipeline

```
documentos → pré-processamento → índice invertido
                                        │
              ┌─────────────────────────┼─────────────────────────┐
              ▼                         ▼                         │
       busca lexical (BM25)     busca semântica (bi-encoder)      │
              │                         │                         │
              └────────────┬────────────┘                         │
                           ▼                                      │
                   busca híbrida (RRF)                            │
                           │                                      │
                           ▼                                      │
                    reranking (cross-encoder)                     │
                           │                                      │
                           ▼                                      │
              seleção de contexto (Top-5) → LLM (RAG) → resposta  │
```

---

## 2. Estrutura do projeto

```
tp2-ri/
├── config.py                 # todas as decisões experimentais
├── requirements.txt          # dependências
├── src/                      # módulos reutilizáveis
│   ├── data_loader.py        # leitura de corpus/queries/qrels
│   ├── preprocessing.py      # tokenização/normalização/stopwords/stemming
│   ├── inverted_index.py     # índice invertido (do zero)
│   ├── lexical_search.py     # BM25
│   ├── evaluation.py         # P@10, R@10, AP@10, NDCG@10
│   ├── spell_correction.py   # k-gramas + Jaccard + distância de edição
│   ├── semantic_search.py    # bi-encoder (Sentence Transformers)
│   ├── hybrid_search.py      # RRF
│   ├── reranking.py          # cross-encoder
│   ├── rag.py                # LLM (Qwen/Transformers) + prompts RAG
│   └── analysis.py           # análises por consulta/falhas
├── scripts/                  # executáveis (ordem numérica)
│   ├── 01_download_data.py
│   ├── 02_build_index.py
│   ├── 03_evaluate_retrieval.py
│   ├── 04_typo_experiment.py
│   ├── 05_rag_experiment.py
│   ├── 06_analysis.py
│   └── 07_plot_results.py     # gráficos do relatório (report/figures/)
├── data/nfcorpus/            # dataset (gerado pelo passo 01)
├── index/                    # índice + vocabulário (gerado no passo 02)
├── embeddings/               # embeddings dos documentos (gerado no passo 03)
├── outputs/                  # CSVs de resultados + análises
└── report/relatorio.tex      # relatório em LaTeX
```

---

## 3. Instalação

**Linguagem:** Python 3.11+ (testado em 3.14).

**Bibliotecas e versões principais:**

| Biblioteca | Versão |
|---|---|
| numpy | 2.4 |
| pandas | 3.0 |
| nltk | 3.10 |
| sentence-transformers | 5.3 |
| torch | 2.12 (CPU) |
| requests | 2.32 |
| tqdm | 4.67 |

**Hardware empregado:** Windows 10, CPU (12 threads), 7,4 GB de RAM.

```powershell
# Na raiz do projeto (ambiente virtual opcional):
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

O próprio código baixa os recursos do NLTK necessários (`stopwords`, e
`wordnet`/`omw-1.4` apenas se a lematização estiver ativa) na primeira execução.

### LLM para o RAG (passo 05)

A LLM roda **localmente, sem login e sem instalar programas extras**, via
HuggingFace Transformers. Na primeira execução o modelo é baixado
automaticamente. O modelo padrão é `Qwen/Qwen2.5-1.5B-Instruct`
(`config.LLM_MODEL`).

**Pouca memória?** Troque `LLM_MODEL` para `Qwen/Qwen2.5-0.5B-Instruct`
(~1 GB) — o modelo 1.5B usa ~3 GB.

---

## 4. Execução

Execute os scripts **na ordem**, a partir da raiz do projeto:

```powershell
python scripts/01_download_data.py        # baixa o NFCorpus (data/nfcorpus/)
python scripts/02_build_index.py          # pré-processa e constrói o índice (index/)
python scripts/03_evaluate_retrieval.py   # lexical + semântica + híbrida + reranking (outputs/)
python scripts/04_typo_experiment.py      # tolerância a erros (outputs/typo_experiment.csv)
python scripts/05_rag_experiment.py       # RAG sem RAG (A) x com RAG (B) (outputs/rag_experiment.csv)
python scripts/06_analysis.py             # análises por consulta (outputs/analysis_*.txt)
python scripts/07_plot_results.py         # gráficos do relatório (report/figures/)
```

Para rodar offline (tudo já cacheado), adicione `set HF_HUB_OFFLINE=1`.

---

## 5. Decisões adotadas

| Componente | Decisão | Onde mudar |
|---|---|---|
| Baseline lexical | **BM25** (k1=1.5, b=0.75) | `config.py` (`BM25_K1`, `BM25_B`) |
| Modelo de embeddings | `sentence-transformers/all-MiniLM-L6-v2` (dim 384) | `config.SEMANTIC_MODEL` |
| Modelo de reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` | `config.RERANK_MODEL` |
| Fusão híbrida | RRF, k ajustado no dev ∈ {10,60,100} | `config.RRF_K_CANDIDATES` |
| LLM (RAG) | local via Transformers, `Qwen/Qwen2.5-1.5B-Instruct` | `config.LLM_MODEL` |
| Pré-processamento | lower + pontuação + stopwords + Porter stemming | `config.PREPROCESSING` |
| Corte das métricas | @10 | `config.METRIC_CUTOFF` |

---

## 6. Arquivos de resultados

| Arquivo | Conteúdo |
|---|---|
| `outputs/results_lexical.csv` | ranking BM25 (top-100) — `query_id, doc_id, rank, score, relevance` |
| `outputs/results_semantic.csv` | ranking bi-encoder (top-100) |
| `outputs/results_hybrid.csv` | ranking RRF (top-100) |
| `outputs/results_reranking.csv` | ranking pós cross-encoder (top-100) |
| `outputs/metrics_per_query.csv` | `query_id, model, P10, R10, AP10, NDCG10` |
| `outputs/typo_experiment.csv` | experimento de tolerância a erros |
| `outputs/rag_experiment.csv` | respostas A/B + notas e justificativas |
| `outputs/index_stats.json` | nº de termos, postings, tempo de construção |
| `outputs/retrieval_times.json` | tempos das etapas |
| `outputs/analysis_*.txt` | análises por consulta (lexical×semântico, reranking) |

---

## 7. Solução de problemas

- **`UnicodeEncodeError` no console**: rode com `python -X utf8 script.py` ou
  defina `$env:PYTHONUTF8 = '1'`.
- **Erro de instalação do torch no Python 3.14**: use um venv com Python 3.11
  (`py -3.11 -m venv venv`).
- **Memória baixa no RAG**: feche programas ou use `Qwen/Qwen2.5-0.5B-Instruct`.
- **LLM não responde**: confirme o modelo em `config.LLM_MODEL` (o padrão é
  `Qwen/Qwen2.5-1.5B-Instruct`, baixado automaticamente na 1ª execução).
- **NLTK não baixa recursos**: rode manualmente `nltk.download("stopwords")`.

---

## 8. Relatório

O relatório completo em LaTeX está em `report/relatorio.tex` (duas colunas, 9pt).
