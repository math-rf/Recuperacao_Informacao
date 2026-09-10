import ir_datasets
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
import math
import pandas as pd

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)


def remove_stopwords(text):
    words = word_tokenize(text)
    stop_words = set(stopwords.words())

    new_text = [word for word in words if word.lower() not in stop_words]

    return ' '.join(new_text) # non-tokenized sentence

def stemming(text):
    stemmer = SnowballStemmer('english')
    words = word_tokenize(text)

    stemmed_text = [stemmer.stem(word) for word in words]

    return ' '.join(stemmed_text)

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
    """
    resultados_por_consulta = {}
    soma_ap = 0.0
    
    for q_id, dados in consultas.items():
        rec = dados['recuperados']
        rel = dados['relevantes']
        
        p_k = precision_at_k(rec, rel, k)
        r_k = recall_at_k(rec, rel, k)
        ap = average_precision(rec, rel)
        f1_k = f1_at_k(rec, rel, k)
        
        resultados_por_consulta[q_id] = {
            f'P@{k}': p_k,
            f'R@{k}': r_k,
            f'F1@{k}': f1_k,
            'AP': ap,
        }
        soma_ap += ap
        
    map_score = soma_ap / len(consultas) if consultas else 0.0
    
    return resultados_por_consulta, map_score


def modelo_vetorial(corpus, query):
    vectorizer = TfidfVectorizer()
    corpus_matrix = vectorizer.fit_transform(corpus)

    query_array = vectorizer.transform([query])

    similarity = cosine_similarity(query_array, corpus_matrix).flatten()

    vocabulary = vectorizer.get_feature_names_out()
    weight_query = query_array.toarray()[0]

    representacao_query = {vocabulary[i]: round(peso, 4) 
                           for i, peso in enumerate(weight_query) if peso > 0}
                           
    ranking = [(i, sim) for i, sim in enumerate(similarity)]
    
    return sorted(ranking, key=lambda x: x[1], reverse=True), representacao_query


def modelo_bm25(corpus_strings, consulta_string, k1=1.5, b=0.75):
    """
    Implementação do BM25 usando CountVectorizer para otimização das frequências,
    mas mantendo o cálculo do score explícito conforme exigido no trabalho.
    """
    # 1. Tarefa Auxiliar: Extração de Frequências (Substitui o Counter)
    vectorizer = CountVectorizer()
    matriz_tf = vectorizer.fit_transform(corpus_strings) # Matriz Documento x Termo
    vocabulario = vectorizer.vocabulary_
    
    # Cálculos globais otimizados com arrays do NumPy
    tamanhos_docs = matriz_tf.sum(axis=1).A1 # Soma de palavras por documento
    N = len(corpus_strings)
    avgdl = np.mean(tamanhos_docs) if N > 0 else 0
    
    # Document Frequency (DF): Quantos documentos têm > 0 ocorrências de cada termo
    df = np.array((matriz_tf > 0).sum(axis=0))[0]
    
    # Tokeniza a consulta usando as mesmas regras do vectorizer
    tokens_consulta = vectorizer.build_analyzer()(consulta_string)
    
    # Cálculo do BM25
    ranking = []
    for i in range(N):
        score_doc = 0
        tamanho_doc = tamanhos_docs[i]
        
        for termo in tokens_consulta:
            if termo in vocabulario:
                idx_termo = vocabulario[termo]
                tf = matriz_tf[i, idx_termo] # Frequência da palavra neste documento
                
                if tf > 0:
                    # Cálculo do IDF (Robertson-Spärck Jones)
                    freq_doc = df[idx_termo]
                    idf = math.log(((N - freq_doc + 0.5) / (freq_doc + 0.5)) + 1)
                    
                    # Saturação do TF e penalização de comprimento
                    numerador = tf * (k1 + 1)
                    denominador = tf + k1 * (1 - b + b * (tamanho_doc / avgdl))
                    
                    score_doc += idf * (numerador / denominador)
                    
        ranking.append((i, score_doc))
        
    return sorted(ranking, key=lambda x: x[1], reverse=True)

def executar_experimentos(del_stopwords=False, use_stemming=False):
    dataset = ir_datasets.load("cranfield")
    df_map = pd.DataFrame()
    doc_ids = []
    corpus_strings = []
    
    for doc in dataset.docs_iter():
        doc_ids.append(doc.doc_id)
        text = doc.text
        if del_stopwords:
            text = remove_stopwords(text)
        if use_stemming:
            text = stemming(text)
        corpus_strings.append(text) 
        
    # Prepara os Julgamentos de Relevância (Qrels)
    qrels = {}
    gabarito = {}
    for qrel in dataset.qrels_iter():
        if qrel.query_id not in qrels:
            qrels[qrel.query_id] = []
            gabarito[qrel.query_id] = {}
            
        if qrel.relevance >= 1:
            qrels[qrel.query_id].append(qrel.doc_id)

        gabarito[qrel.query_id][qrel.doc_id] = qrel.relevance

    # Estruturas para guardar os resultados
    resultados_vetorial = {}
    resultados_bm25 = {}
    
    # Loop pelas Consultas
    for query in dataset.queries_iter():
        q_id = query.query_id
        q_texto = query.text
        if del_stopwords:
            q_texto = remove_stopwords(q_texto)
        if use_stemming:
            q_texto = stemming(q_texto)
        
        
        # Ignora consultas que não possuem documentos relevantes mapeados
        if q_id not in qrels or len(qrels[q_id]) == 0:
            continue
            
        # --- EXECUÇÃO MODELO VETORIAL ---
        # Retorna lista de tuplas (indice_doc, score)
        ranking_vetorial, pesos_query = modelo_vetorial(corpus_strings, q_texto)
        # Converte os índices de volta para os doc_ids originais
        docs_recuperados_vet = [doc_ids[idx] for (idx, score) in ranking_vetorial]

        detalhes_vet = [
            (doc_ids[idx], round(score, 4), gabarito.get(q_id, {}).get(doc_ids[idx], 0)) 
            for idx, score in ranking_vetorial[:10]
        ]
        
        resultados_vetorial[q_id] = {
            "recuperados": docs_recuperados_vet,
            "relevantes": qrels[q_id],
            "top_k": detalhes_vet
        }
        
        # --- EXECUÇÃO BM25 ---
        ranking_bm25 = modelo_bm25(corpus_strings, q_texto)
        docs_recuperados_bm25 = [doc_ids[idx] for idx, score in ranking_bm25]

        detalhes_bm25 = [
            (doc_ids[idx], round(score, 4), gabarito.get(q_id, {}).get(doc_ids[idx], 0)) 
            for idx, score in ranking_bm25[:10]
        ]
        
        resultados_bm25[q_id] = {
            "recuperados": docs_recuperados_bm25,
            "relevantes": qrels[q_id],
            "top_k": detalhes_bm25
        }
        
    # Avaliação Final
    metric_per_query_vet, map_vetorial = avaliar_sistema(resultados_vetorial, k=10)
    metric_per_query_bm25, map_bm25 = avaliar_sistema(resultados_bm25, k=10)

    # Substitua os dois .append() por uma criação direta:
    dados_map = [
    {'Modelo': 'Vetorial', 'Stopwords_Removidas': del_stopwords, 'Stemming_Aplicado': use_stemming, 'MAP': map_vetorial},
    {'Modelo': 'BM25', 'Stopwords_Removidas': del_stopwords, 'Stemming_Aplicado': use_stemming, 'MAP': map_bm25}
    ]
    df_map = pd.DataFrame(dados_map)

    print(f"Modelo: Vetorial\nStopwords: {del_stopwords}\nStemming: {use_stemming}\nMAP: {map_vetorial}")
    print(f"Modelo: BM25\nStopwords: {del_stopwords}\nStemming: {use_stemming}\nMAP: {map_bm25}")


    linhas = []
    for modelo, metricas_dict, results in [('Vetorial', metric_per_query_vet, resultados_vetorial), ('BM25', metric_per_query_bm25, resultados_bm25)]:
        for q_id, metricas in metricas_dict.items():
            linha = {
                'Query_ID': q_id,
                'Modelo': modelo,
                'Stopwords_Removidas': int(del_stopwords), # Salva como 0 ou 1
                'Stemming_Aplicado': int(use_stemming),     # Salva como 0 ou 1
                'Top_10': results[q_id]['top_k']
            }
            linha.update(metricas)
            linhas.append(linha)
            
    return pd.DataFrame(linhas), df_map