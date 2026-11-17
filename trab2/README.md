# Trabalho Prático 2 — Recuperação de Informação (SCC0282)

**Universidade de São Paulo — ICMC — Bacharelado em Sistemas de Informação**
Prof. Marcelo Manzato · 2º Semestre/2026

Sistema completo de Recuperação de Informação que combina **busca lexical**
(BM25 sobre índice invertido implementado do zero), **busca semântica**
(bi-encoder), **busca híbrida** (RRF), **reranking** (cross-encoder),
**tolerância a erros** (k-gramas + Jaccard + distância de edição) e **RAG**
(LLM local quantizada, Qwen3), sobre o dataset **NFCorpus (BEIR)**.

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
│   ├── rag.py                # LLM quantizada (bitsandbytes/Transformers) + prompts RAG
│   └── analysis.py           # análises por consulta/falhas
├── tests/                    # testes do corretor ortográfico (unittest)
├── scripts/                  # executáveis (ordem numérica)
│   ├── 01_download_data.py
│   ├── 02_build_index.py
│   ├── 03_evaluate_retrieval.py
│   ├── 04_typo_experiment.py
│   ├── 05_rag_experiment.py
│   ├── 06_analysis.py
│   ├── 07_plot_results.py     # gráficos (report/figures/)
│   └── 08_truncation_stats.py # truncamento dos modelos (outputs/truncation_stats.json)
├── data/nfcorpus/            # dataset (gerado pelo passo 01)
├── index/                    # índice + vocabulário (gerado no passo 02)
├── embeddings/               # embeddings dos documentos (gerado no passo 03)
├── outputs/                  # CSVs de resultados + análises
└── report/                   # gerado pelo passo 07 (figuras)
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
| torch | versão com CUDA (ex.: cu126) |
| bitsandbytes | >= 0.45 (quantização 4 bits) |
| accelerate | >= 0.30 |
| requests | 2.32 |
| tqdm | 4.67 |

**Hardware empregado:** Windows 10, CPU (12 threads), 7,4 GB de RAM; GPU NVIDIA GeForce GTX 1650 (4 GB de VRAM, driver 591.86) usada pelo LLM do passo 05.

```powershell
# Na raiz do projeto (ambiente virtual opcional):
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

O próprio código baixa o recurso do NLTK necessário (`stopwords`) na primeira execução.

### LLM para o RAG (passo 05)

A LLM roda **localmente, na GPU**, via HuggingFace Transformers com quantização
**4 bits (NF4, bitsandbytes)**, para caber em 4 GB de VRAM. O modelo padrão é
`Qwen/Qwen3-1.7B` (`config.LLM_MODEL`), baixado automaticamente na
1ª execução. Sem GPU CUDA, o script para com erro explicando o motivo.

Modo de pensamento desativado (`enable_thinking=False`), para que a resposta
saia direta, sem bloco de raciocínio.

**Parâmetros do LLM (passo 05)**

| Parâmetro | Valor |
|---|---|
| Modelo | `Qwen/Qwen3-1.7B` (Hugging Face) |
| Versão (commit) | `<PREENCHER: hash do snapshot>` (comando abaixo) |
| Forma de execução | local, GPU via Transformers + bitsandbytes |
| Quantização | 4 bits, NF4, com dupla quantização; compute em float16 |
| Temperatura | 0.0 (decodificação gulosa, `do_sample=False`) |
| Máximo de tokens gerados | 512 (`max_new_tokens`) |
| Seed | 42 |
| Idioma das respostas | português (condições A e B) |
| Custo | R$ 0 (sem API; execução local) |
| Hardware | GPU NVIDIA GeForce GTX 1650, 4 GB de VRAM; CPU e 7,4 GB de RAM para o restante |
| Biblioteca | transformers, bitsandbytes >= 0.45, torch com CUDA |

Para obter o commit do modelo:

```powershell
Get-ChildItem "$env:USERPROFILE\.cache\huggingface\hub\models--Qwen--Qwen3-1.7B\snapshots"
```

O nome da pasta que aparece é o commit. O aviso "generation flags ... temperature ... may be ignored"
do log é esperado: com temperatura 0 a geração é gulosa, e os parâmetros de amostragem não são usados.

---

## 4. Execução

Execute os scripts **na ordem**, a partir da raiz do projeto:

```powershell
python scripts/01_download_data.py        # baixa o NFCorpus (data/nfcorpus/)
python scripts/02_build_index.py          # pré-processa e constrói o índice (index/)
python scripts/03_evaluate_retrieval.py   # lexical + semântica + híbrida + reranking (outputs/)
python scripts/04_typo_experiment.py      # tolerância a erros: escolha no dev + teste (outputs/typo_*.csv)
python scripts/05_rag_experiment.py       # RAG sem RAG (A) x com RAG (B) (outputs/rag_experiment.csv)
python scripts/06_analysis.py             # análises por consulta (outputs/analysis_*.txt)
python scripts/07_plot_results.py         # gráficos do relatório (report/figures/)
python scripts/08_truncation_stats.py     # contagem de truncamento (outputs/truncation_stats.json)
```

Testes do corretor ortográfico (não precisam de dados nem de GPU):

```powershell
python -X utf8 -m unittest discover -s tests -v
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
| LLM (RAG) | local na GPU, `Qwen/Qwen3-1.7B` em 4 bits (NF4) | `config.LLM_MODEL` |
| Pré-processamento | lower + pontuação + stopwords + Porter stemming | `config.PREPROCESSING` |
| Corte das métricas | @10 | `config.METRIC_CUTOFF` |
| Idioma das respostas do RAG | português (condições A e B) | `config.LLM_LANGUAGE` |

---

## 6. Arquivos de resultados

| Arquivo | Conteúdo |
|---|---|
| `outputs/results_lexical.csv` | ranking BM25 (top-100; documentos sem termo em comum entram com score 0) — `query_id, doc_id, rank, score, relevance` |
| `outputs/results_semantic.csv` | ranking bi-encoder (top-100) |
| `outputs/results_hybrid.csv` | ranking RRF (top-100) |
| `outputs/results_reranking.csv` | ranking pós cross-encoder (top-100) |
| `outputs/metrics_per_query.csv` | `query_id, model, P10, R10, AP10, NDCG10` |
| `outputs/typo_dev_selection.csv` | varredura no dev: variante (distância × dicionário) × limiar de Jaccard, com recuperação média e taxa de acerto exato |
| `outputs/typo_experiment.csv` | experimento de tolerância a erros no test (configuração escolhida no dev); `correcao_exata` indica se o termo corrigido é o original; `observacao` marca os casos em que o erro não reduziu o desempenho |
| `outputs/typo_summary.json` | configuração escolhida no dev e médias no test (recuperação, casos válidos, acerto exato) |
| `outputs/rag_experiment.csv` | respostas A/B, contexto (doc_ids), recall/precision/hit@5, notas 0–2 e justificativas da avaliação humana; `status_revisao` marca notas ainda a confirmar (`a confirmar`) |
| `outputs/index_stats.json` | nº de termos, postings, tempo de construção |
| `outputs/retrieval_times.json` | tempos das etapas |
| `outputs/truncation_stats.json` | documentos e pares truncados pelo limite de entrada de cada modelo |
| `outputs/analysis_*.txt` | análises por consulta (lexical×semântico, reranking) |

---

## 6.1 Limitações e decisões que o relatório deve declarar

- **Completar o ranking lexical com score 0.** Documentos sem nenhum termo em comum são
  colocados depois dos pontuados, em ordem de inserção, até somar 100. Isso ocorre em 127 das
  323 consultas. Em P@10 o efeito é pequeno (1 relevante entre 550 posições preenchidas no top-10),
  mas esses documentos entram na fusão RRF.
- **Coluna `score` do `results_reranking.csv`.** Nas 50 primeiras posições é o logit do
  cross-encoder; na cauda, é o score RRF. Não compare esses valores entre si.
- **Consultas do experimento de typo.** São as 10 primeiras por ID com NDCG@10 original > 0,
  não um sorteio.
- **Empate no corretor.** Termos à mesma distância de edição são desempatados pela frequência
  documental (DF). Ex.: `cemo` tem `ecmo` e `chemo` à distância 1, e vence `chemo` (DF maior).
  Esse é o caso de falha `ecmo → cemo → chemo` para a análise de falhas.
- **Truncamento.** 78,8% dos documentos são truncados pelo bi-encoder (256 tokens) e 8,1% dos pares
  pelo cross-encoder (512 tokens); ver `outputs/truncation_stats.json`.

## 7. Solução de problemas

- **`UnicodeEncodeError` no console**: rode com `python -X utf8 script.py` ou
  defina `$env:PYTHONUTF8 = '1'`.
- **Erro de instalação do torch no Python 3.14**: use um venv com Python 3.11
  (`py -3.11 -m venv venv`).
- **Falta de VRAM no RAG**: feche programas que usam a GPU e rode o passo 05 novamente.
- **LLM não responde**: confirme o modelo em `config.LLM_MODEL` (o padrão é
  `Qwen/Qwen3-1.7B`, baixado automaticamente na 1ª execução) e confira
  se `torch.cuda.is_available()` retorna `True`.
- **NLTK não baixa recursos**: rode manualmente `nltk.download("stopwords")`.

---

