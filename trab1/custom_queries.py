import ir_datasets
import pandas as pd
from ir_utils import modelo_vetorial, modelo_bm25, remove_stopwords, stemming

dataset = ir_datasets.load("cranfield")
doc_ids = []
corpus_strings = []

for doc in dataset.docs_iter():
    doc_ids.append(doc.doc_id)
    # Aplica stopwords e depois stemming no documento
    texto_limpo = stemming(remove_stopwords(doc.text))
    corpus_strings.append(texto_limpo)

# Formato: 'ID_da_Consulta': ('Texto Original Exato', 'Texto Modificado')
consultas_customizadas = {
    '1': (
        "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .",
        "similarity laws aeroelastic models heated high speed aircraft" # Reduzido apenas para palavras-chave
    ),
    '2': (
        "what are the structural and aeroelastic problems associated with flight of high speed aircraft .",
        "structural aeroelastic problems flight supersonic aircraft" # Troca de "high speed" por sinônimo "supersonic"
    ),
    '3': (
        "can a criterion be developed to show empirically the validity of flow solutions for chemically reacting gas mixtures based on the assumption of thermodynamic equilibrium .",
        "validity of flow solutions chemically reacting gas mixtures thermodynamic equilibrium"
    ),
    '4': (
        "what is the present state of the art in fluid mechanics as applied to re-entry vehicles.",
        "fluid mechanics applied to re-entry vehicles spacecraft" # Expansão adicionando um termo específico
    ),
    '5': (
        "has anyone developed an analysis which accurately establishes the large deflection behavior of conical shells .",
        "analysis large deflection behavior conical shells"
    )
}

print("Executando consultas originais e modificadas...")
resultados = []

# Executa a busca comparativa
for q_id, (txt_original, txt_modificado) in consultas_customizadas.items():
    
    # Aplica pré-processamento nas consultas
    q_orig_limpa = stemming(remove_stopwords(txt_original))
    q_mod_limpa = stemming(remove_stopwords(txt_modificado))
    
    # --- MODELO VETORIAL ---
    rank_vet_orig, _ = modelo_vetorial(corpus_strings, q_orig_limpa)
    rank_vet_mod, _ = modelo_vetorial(corpus_strings, q_mod_limpa)
    
    # Extrai apenas o Top-10 (os IDs dos documentos)
    top10_vet_orig = [doc_ids[idx] for idx, score in rank_vet_orig[:10]]
    top10_vet_mod = [doc_ids[idx] for idx, score in rank_vet_mod[:10]]
    
    # --- MODELO BM25 ---
    rank_bm25_orig = modelo_bm25(corpus_strings, q_orig_limpa)
    rank_bm25_mod = modelo_bm25(corpus_strings, q_mod_limpa)
    
    # Extrai apenas o Top-10
    top10_bm25_orig = [doc_ids[idx] for idx, score in rank_bm25_orig[:10]]
    top10_bm25_mod = [doc_ids[idx] for idx, score in rank_bm25_mod[:10]]
    
    # Salva os dados na lista
    resultados.append({'Query_ID': q_id, 'Modelo': 'Vetorial', 'Versao': 'Original', 'Top_10': top10_vet_orig})
    resultados.append({'Query_ID': q_id, 'Modelo': 'Vetorial', 'Versao': 'Modificada', 'Top_10': top10_vet_mod})
    resultados.append({'Query_ID': q_id, 'Modelo': 'BM25', 'Versao': 'Original', 'Top_10': top10_bm25_orig})
    resultados.append({'Query_ID': q_id, 'Modelo': 'BM25', 'Versao': 'Modificada', 'Top_10': top10_bm25_mod})

# Converte para DataFrame e salva o CSV
df_comparacao = pd.DataFrame(resultados)
df_comparacao.to_csv('comparacao_queries_modificadas.csv', index=False)

print("\nProcesso finalizado!")
print("Arquivo 'comparacao_queries_modificadas.csv' gerado com sucesso.")