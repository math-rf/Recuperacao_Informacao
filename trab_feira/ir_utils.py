"""
Módulo utilitário de Recuperação de Informação (IR) integrando o mMARCO (PT).
"""

import math
import re
import ir_datasets
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Lista estendida de Stopwords em Português
STOPWORDS_PT = set([
    "a", "o", "e", "é", "de", "do", "da", "em", "um", "para", "com", "não", "uma", 
    "os", "no", "se", "na", "por", "mais", "as", "dos", "como", "mas", "foi", "ao", 
    "ele", "das", "seu", "sua", "ou", "quando", "muito", "nos", "já", "está", "eu", 
    "também", "só", "pelo", "pela", "até", "isso", "eles", "esta", "entre", "depois", 
    "sem", "mesmo", "aos", "suas", "quem", "nas", "me", "esse", "você", "faz", "fazer",
    "qual", "quais", "onde", "como", "porque", "porquê", "que", "sobre"
])

def carregar_corpus_mmarco(limite_docs=1000):
    """
    Carrega uma amostra do corpus mMARCO (Português) via ir_datasets
    para manter a aplicação web ágil e responsiva.
    """
    try:
        dataset = ir_datasets.load("mmarco/v2/pt")
        corpus_docs = []
        for i, doc in enumerate(dataset.docs_iter()):
            if i >= limite_docs:
                break
            corpus_docs.append({
                "id": doc.doc_id,
                "titulo": f"Passagem mMARCO {doc.doc_id[:8]}",
                "categoria": "mMARCO (Web)",
                "texto": doc.text
            })
        return corpus_docs
    except Exception as e:
        # Fallback caso ocorra algum problema de rede ou dependência no dataset
        print(f"Erro ao carregar ir_datasets: {e}. Usando corpus de fallback.")
        return [
            {
                "id": "fb1",
                "titulo": "Inteligência Artificial no Brasil",
                "categoria": "Tecnologia",
                "texto": "A inteligência artificial aplicada ao processamento de dados impulsiona a inovação tecnológica no Brasil."
            },
            {
                "id": "fb2",
                "titulo": "A Hidrografia da Amazônia",
                "categoria": "Geografia",
                "texto": "A bacia amazônica possui o maior volume de água doce do planeta, sendo vital para o clima mundial."
            }
        ]


def preprocessar_texto(texto, remover_stopwords=True):
    """Tokeniza e limpa um texto em português."""
    texto_minusculo = texto.lower()
    tokens = re.findall(r'\b\w+\b', texto_minusculo)
    if remover_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS_PT]
    return tokens


# ---------------------------------------------------------------------------
# MODELOS DE SIMILARIDADE
# ---------------------------------------------------------------------------

def modelo_vetorial_tfidf(corpus_strings, query_string):
    """Modelo Vetorial TF-IDF com Similaridade de Cosseno."""
    vectorizer = TfidfVectorizer(stop_words=list(STOPWORDS_PT), lowercase=True)
    matriz_tfidf = vectorizer.fit_transform(corpus_strings)
    vetor_query = vectorizer.transform([query_string])
    
    similaridades = cosine_similarity(vetor_query, matriz_tfidf).flatten()
    ranking = sorted(list(enumerate(similaridades)), key=lambda x: x[1], reverse=True)
    return ranking, vectorizer, matriz_tfidf


def modelo_bm25(corpus_strings, query_string, k1=1.5, b=0.75):
    """Modelo de Recuperação Probabilístico Okapi BM25."""
    vectorizer = CountVectorizer(stop_words=list(STOPWORDS_PT), lowercase=True)
    matriz_tf = vectorizer.fit_transform(corpus_strings)
    vocabulario = vectorizer.vocabulary_

    tamanhos_docs = matriz_tf.sum(axis=1).A1
    N = len(corpus_strings)
    avgdl = np.mean(tamanhos_docs) if N > 0 else 1.0

    df = np.array((matriz_tf > 0).sum(axis=0))[0]
    tokens_query = preprocessar_texto(query_string, remover_stopwords=True)

    ranking = []
    for i in range(N):
        score_doc = 0.0
        tamanho_doc = tamanhos_docs[i]

        for termo in tokens_query:
            if termo in vocabulario:
                idx_termo = vocabulario[termo]
                tf = matriz_tf[i, idx_termo]
                if tf > 0:
                    freq_doc = df[idx_termo]
                    idf = math.log(((N - freq_doc + 0.5) / (freq_doc + 0.5)) + 1)
                    numerador = tf * (k1 + 1)
                    denominador = tf + k1 * (1 - b + b * (tamanho_doc / avgdl))
                    score_doc += idf * (numerador / denominador)

        ranking.append((i, float(score_doc)))

    return sorted(ranking, key=lambda x: x[1], reverse=True)


def modelo_jaccard(corpus_strings, query_string):
    """Modelo de Similaridade Baseado em Sobreposição de Conjuntos (Jaccard)."""
    tokens_query = set(preprocessar_texto(query_string, remover_stopwords=True))
    ranking = []

    for i, doc in enumerate(corpus_strings):
        tokens_doc = set(preprocessar_texto(doc, remover_stopwords=True))
        uniao = tokens_query.union(tokens_doc)
        intersecao = tokens_query.intersection(tokens_doc)
        
        score = len(intersecao) / len(uniao) if uniao else 0.0
        ranking.append((i, score))

    return sorted(ranking, key=lambda x: x[1], reverse=True)


def executar_modelo_busca(nome_modelo, corpus_strings, query_string):
    """Dispatcher genérico para executar modelos no app.py."""
    if nome_modelo == "Vetorial (TF-IDF + Cosseno)":
        ranking, vectorizer, matriz = modelo_vetorial_tfidf(corpus_strings, query_string)
        return ranking, vectorizer, matriz
    elif nome_modelo == "BM25 (Probabilístico)":
        ranking = modelo_bm25(corpus_strings, query_string)
        return ranking, None, None
    elif nome_modelo == "Similaridade de Jaccard (Conjuntos)":
        ranking = modelo_jaccard(corpus_strings, query_string)
        return ranking, None, None
    else:
        raise ValueError(f"Modelo não suportado: {nome_modelo}")