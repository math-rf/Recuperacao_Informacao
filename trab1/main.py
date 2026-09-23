"""
Bateria de experimentos: roda Vetorial e BM25 no Cranfield para as 4
combinações de pré-processamento (com/sem stopwords x com/sem stemming) e
salva os resultados em results/.
"""
import os

import pandas as pd

from ir_utils import executar_experimentos

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# (del_stopwords, use_stemming, rótulo para o log)
CONFIGURACOES = [
    (False, False, "Texto original (Baseline)"),
    (True, False, "Apenas remoção de stopwords"),
    (False, True, "Apenas stemming"),
    (True, True, "Stopwords + Stemming"),
]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("Iniciando bateria de experimentos...\n")
    dfs_experimentos, dfs_map = [], []

    for i, (del_stopwords, use_stemming, rotulo) in enumerate(CONFIGURACOES, start=1):
        print(f"{i}. Processando: {rotulo}...")
        df_exp, df_map = executar_experimentos(del_stopwords=del_stopwords, use_stemming=use_stemming)
        dfs_experimentos.append(df_exp)
        dfs_map.append(df_map)

    # Unindo tudo em um único DataFrame mestre
    df_final_exp = pd.concat(dfs_experimentos, ignore_index=True)
    df_final_map = pd.concat(dfs_map, ignore_index=True)

    # Salvando o resultado completo para análise
    df_final_exp.to_csv(os.path.join(RESULTS_DIR, "results_default_queries.csv"), index=False)
    df_final_map.to_csv(os.path.join(RESULTS_DIR, "results_default_queries_map.csv"), index=False)

    print("\nConcluído! Dimensões do DataFrame final (exp e map):", df_final_exp.shape, df_final_map.shape)


if __name__ == "__main__":
    main()