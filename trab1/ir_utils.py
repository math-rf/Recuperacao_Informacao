"""
Funções utilitárias do trabalho de Recuperação da Informação (Cranfield).

Organização do módulo:
  1. Pré-processamento de texto (stopwords, stemming)
  2. Carregamento do dataset Cranfield (genérico, usado por todos os scripts)
  3. Modelos de recuperação (Vetorial, BM25) + dispatcher genérico
  4. Métricas de avaliação (P@k, R@k, F1@k, AP, MAP)
  5. Pipeline de experimentos (usado pelo main.py)
"""
import math

import ir_datasets
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Stopwords carregadas uma única vez (evita reconstruir o set a cada chamada
# de remove_stopwords, que antes era chamada centenas de vezes por execução).
_STOPWORDS = set(stopwords.words('english'))


# ---------------------------------------------------------------------------
# 1. Pré-processamento de texto
# ---------------------------------------------------------------------------

def remove_stopwords(text):
    """Remove stopwords (idioma detectado pelo conjunto padrão do NLTK)."""
    words = word_tokenize(text)
    new_text = [word for word in words if word.lower() not in _STOPWORDS]
    return ' '.join(new_text)  # non-tokenized sentence


def stemming(text):
    """Aplica stemming (Snowball, inglês) a cada token do texto."""
    stemmer = SnowballStemmer('english')
    words = word_tokenize(text)
    stemmed_text = [stemmer.stem(word) for word in words]
    return ' '.join(stemmed_text)


def preprocessar_texto(texto, del_stopwords=False, use_stemming=False):
    """Aplica remoção de stopwords e/ou stemming conforme as flags.

    Centraliza o padrão `if del_stopwords: ... / if use_stemming: ...` que
    antes estava repetido em quase todo script do projeto.
    """
    if del_stopwords:
        texto = remove_stopwords(texto)
    if use_stemming:
        texto = stemming(texto)
    return texto


# ---------------------------------------------------------------------------
# 2. Carregamento do dataset Cranfield
# ---------------------------------------------------------------------------

def carregar_cranfield(del_stopwords=False, use_stemming=False):
    """Carrega o Cranfield já pré-processado e pronto para uso pelos modelos.

    Substitui o bloco de ~25 linhas (docs_iter + qrels_iter + queries_iter)
    que antes era copiado em ir_utils.executar_experimentos,
    bm25_params_test.preparar_dados, error_analysis.preparar e
    custom_queries.preparar.

    Retorna:
      doc_ids               -- lista de doc_id, na ordem do corpus
      corpus_strings         -- textos de documento pré-processados (mesma ordem de doc_ids)
      textos_doc_originais    -- dict doc_id -> texto original (sem pré-processamento)
      qrels                   -- dict query_id -> lista de doc_ids relevantes (relevance >= 1)
      gabarito                -- dict query_id -> dict doc_id -> relevance (todos os julgamentos)
      consultas                -- lista de (query_id, texto_original, texto_processado),
                                   já filtrada para conter só queries com >=1 doc relevante
    """
    dataset = ir_datasets.load("cranfield")

    doc_ids, corpus_strings, textos_doc_originais = [], [], {}
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        textos_doc_originais[doc.doc_id] = doc.text
        corpus_strings.append(preprocessar_texto(doc.text, del_stopwords, use_stemming))

    qrels, gabarito = {}, {}
    for qrel in dataset.qrels_iter():
        qrels.setdefault(qrel.query_id, [])
        gabarito.setdefault(qrel.query_id, {})[qrel.doc_id] = qrel.relevance
        if qrel.relevance >= 1:
            qrels[qrel.query_id].append(qrel.doc_id)

    consultas = []
    for query in dataset.queries_iter():
        if qrels.get(query.query_id):  # ignora queries sem nenhum doc relevante
            texto_original = query.text.strip()
            texto_processado = preprocessar_texto(query.text, del_stopwords, use_stemming)
            consultas.append((query.query_id, texto_original, texto_processado))

    return doc_ids, corpus_strings, textos_doc_originais, qrels, gabarito, consultas


# ---------------------------------------------------------------------------
# 3. Modelos de recuperação
# ---------------------------------------------------------------------------

def modelo_vetorial(corpus, query):
    """Modelo vetorial (TF-IDF + similaridade de cosseno).

    Retorna (ranking, representacao_query), onde ranking é uma lista de
    (indice_no_corpus, score) ordenada por score decrescente.
    """
    vectorizer = TfidfVectorizer()
    corpus_matrix = vectorizer.fit_transform(corpus)
    query_array = vectorizer.transform([query])

    similarity = cosine_similarity(query_array, corpus_matrix).flatten()

    vocabulary = vectorizer.get_feature_names_out()
    weight_query = query_array.toarray()[0]
    representacao_query = {
        vocabulary[i]: round(peso, 4)
        for i, peso in enumerate(weight_query) if peso > 0
    }

    ranking = list(enumerate(similarity))
    return sorted(ranking, key=lambda x: x[1], reverse=True), representacao_query


def modelo_bm25(corpus_strings, consulta_string, k1=1.5, b=0.75):
    """
    Implementação do BM25 usando CountVectorizer para otimização das frequências,
    mas mantendo o cálculo do score explícito conforme exigido no trabalho.

    Retorna uma lista de (indice_no_corpus, score) ordenada por score decrescente.
    """
    vectorizer = CountVectorizer()
    matriz_tf = vectorizer.fit_transform(corpus_strings)  # Matriz Documento x Termo
    vocabulario = vectorizer.vocabulary_

    tamanhos_docs = matriz_tf.sum(axis=1).A1  # Soma de palavras por documento
    N = len(corpus_strings)
    avgdl = np.mean(tamanhos_docs) if N > 0 else 0

    # Document Frequency (DF): quantos documentos têm > 0 ocorrências de cada termo
    df = np.array((matriz_tf > 0).sum(axis=0))[0]

    tokens_consulta = vectorizer.build_analyzer()(consulta_string)

    ranking = []
    for i in range(N):
        score_doc = 0
        tamanho_doc = tamanhos_docs[i]

        for termo in tokens_consulta:
            if termo not in vocabulario:
                continue
            idx_termo = vocabulario[termo]
            tf = matriz_tf[i, idx_termo]
            if tf <= 0:
                continue

            # IDF (Robertson-Spärck Jones)
            freq_doc = df[idx_termo]
            idf = math.log(((N - freq_doc + 0.5) / (freq_doc + 0.5)) + 1)

            # Saturação do TF e penalização de comprimento
            numerador = tf * (k1 + 1)
            denominador = tf + k1 * (1 - b + b * (tamanho_doc / avgdl))
            score_doc += idf * (numerador / denominador)

        ranking.append((i, score_doc))

    return sorted(ranking, key=lambda x: x[1], reverse=True)


def executar_modelo(nome_modelo, corpus_strings, query_texto, **kwargs):
    """Dispatcher genérico: roda o modelo pelo nome ('Vetorial' ou 'BM25').

    Sempre devolve uma lista de (indice_no_corpus, score) ordenada por score
    decrescente, escondendo a diferença de assinatura entre modelo_vetorial
    (que também devolve os pesos da query) e modelo_bm25.
    """
    if nome_modelo == "Vetorial":
        ranking, _ = modelo_vetorial(corpus_strings, query_texto)
        return ranking
    if nome_modelo == "BM25":
        return modelo_bm25(corpus_strings, query_texto, **kwargs)
    raise ValueError(f"Modelo desconhecido: {nome_modelo!r} (use 'Vetorial' ou 'BM25')")


def ranking_para_doc_ids(ranking, doc_ids, k=None):
    """Converte uma lista de (indice_no_corpus, score) em uma lista de doc_ids,
    na mesma ordem. Use k para limitar ao Top-k (k=10 -> Top-10)."""
    pares = ranking[:k] if k else ranking
    return [doc_ids[idx] for idx, _ in pares]


def ranking_com_detalhes(ranking, doc_ids, gabarito, q_id, k=10):
    """Top-k como lista de (doc_id, score arredondado, relevância real),
    útil para inspeção/relatórios (mesmo formato salvo em Top_10 pelo main.py)."""
    return [
        (doc_ids[idx], round(score, 4), gabarito.get(q_id, {}).get(doc_ids[idx], 0))
        for idx, score in ranking[:k]
    ]


# ---------------------------------------------------------------------------
# 4. Métricas de avaliação
# ---------------------------------------------------------------------------

def precision_at_k(recuperados, relevantes, k=10):
    recuperados_k = recuperados[:k]
    relevantes_set = set(relevantes)
    acertos = sum(1 for doc in recuperados_k if doc in relevantes_set)
    return acertos / k if k > 0 else 0.0


def recall_at_k(recuperados, relevantes, k=10):
    recuperados_k = recuperados[:k]
    relevantes_set = set(relevantes)
    acertos = sum(1 for doc in recuperados_k if doc in relevantes_set)
    return acertos / len(relevantes_set) if len(relevantes_set) > 0 else 0.0


def average_precision(recuperados, relevantes):
    relevantes_set = set(relevantes)
    acertos = 0
    soma_precisoes = 0.0

    for i, doc in enumerate(recuperados):
        if doc in relevantes_set:
            acertos += 1
            soma_precisoes += acertos / (i + 1.0)

    return soma_precisoes / len(relevantes_set) if len(relevantes_set) > 0 else 0.0


def f1_at_k(recuperados, relevantes, k=10):
    p = precision_at_k(recuperados, relevantes, k)
    r = recall_at_k(recuperados, relevantes, k)
    return 2 * (p * r) / (p + r) if (p + r) > 0 else 0.0


def avaliar_sistema(consultas, k=10):
    """
    'consultas' deve ser um dicionário no formato:
    {
      'id_consulta': {
          'recuperados': ['doc1', 'doc2', 'doc3', ...],
          'relevantes': ['doc2', 'doc5', ...]
      }
    }

    Retorna (resultados_por_consulta, map_score, mean_f1_score):
      resultados_por_consulta -- dict query_id -> {P@k, R@k, F1@k, AP}
      map_score                -- MAP (média das AP de todas as consultas)
      mean_f1_score             -- média do F1@k de todas as consultas
    """
    resultados_por_consulta = {}
    soma_ap = 0.0
    soma_f1 = 0.0

    for q_id, dados in consultas.items():
        rec = dados['recuperados']
        rel = dados['relevantes']

        ap = average_precision(rec, rel)
        f1_k = f1_at_k(rec, rel, k)

        resultados_por_consulta[q_id] = {
            f'P@{k}': precision_at_k(rec, rel, k),
            f'R@{k}': recall_at_k(rec, rel, k),
            f'F1@{k}': f1_k,
            'AP': ap,
        }
        soma_ap += ap
        soma_f1 += f1_k

    n = len(consultas)
    map_score = soma_ap / n if n else 0.0
    mean_f1_score = soma_f1 / n if n else 0.0

    return resultados_por_consulta, map_score, mean_f1_score


# ---------------------------------------------------------------------------
# 5. Pipeline de experimentos (main.py)
# ---------------------------------------------------------------------------

def executar_experimentos(del_stopwords=False, use_stemming=False):
    """Roda Vetorial e BM25 em todas as consultas do Cranfield para uma dada
    configuração de pré-processamento, e avalia os resultados (MAP e F1@10).

    Retorna (df_experimentos, df_map):
      df_experimentos -- 1 linha por (consulta, modelo): métricas + Top-10 detalhado
      df_map            -- 1 linha por modelo: MAP e F1@10 médios da configuração
    """
    doc_ids, corpus_strings, _, qrels, gabarito, consultas = carregar_cranfield(
        del_stopwords, use_stemming
    )

    resultados = {"Vetorial": {}, "BM25": {}}
    for q_id, _texto_original, q_texto in consultas:
        for nome_modelo in resultados:
            ranking = executar_modelo(nome_modelo, corpus_strings, q_texto)
            resultados[nome_modelo][q_id] = {
                "recuperados": ranking_para_doc_ids(ranking, doc_ids),
                "relevantes": qrels[q_id],
                "top_k": ranking_com_detalhes(ranking, doc_ids, gabarito, q_id, k=10),
            }

    # Avaliação final (MAP + F1@10 médio) por modelo
    dados_map, linhas = [], []
    for nome_modelo, resultados_modelo in resultados.items():
        metricas_por_query, map_score, f1_medio = avaliar_sistema(resultados_modelo, k=10)

        dados_map.append({
            'Modelo': nome_modelo,
            'Stopwords_Removidas': del_stopwords,
            'Stemming_Aplicado': use_stemming,
            'MAP': map_score,
            'F1@10': f1_medio,
        })
        print(f"Modelo: {nome_modelo}\nStopwords: {del_stopwords}\nStemming: {use_stemming}\n"
              f"MAP: {map_score}\nF1@10: {f1_medio}")

        for q_id, metricas in metricas_por_query.items():
            linha = {
                'Query_ID': q_id,
                'Modelo': nome_modelo,
                'Stopwords_Removidas': int(del_stopwords),
                'Stemming_Aplicado': int(use_stemming),
                'Top_10': resultados_modelo[q_id]['top_k'],
            }
            linha.update(metricas)
            linhas.append(linha)

    return pd.DataFrame(linhas), pd.DataFrame(dados_map)