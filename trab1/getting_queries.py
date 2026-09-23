"""Salva o texto de todas as queries do Cranfield em CSV, para uso posterior
(ex.: query_analysis.py lê este arquivo para exibir o texto da consulta)."""
import os

import ir_datasets
import pandas as pd

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARQUIVO_SAIDA = os.path.join(RESULTS_DIR, "cranfield_queries_texts.csv")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    dataset = ir_datasets.load("cranfield")

    df_consultas = pd.DataFrame([
        {"Query_ID": query.query_id, "Texto_Consulta": query.text.strip()}
        for query in dataset.queries_iter()
    ])
    df_consultas.to_csv(ARQUIVO_SAIDA, index=False)

    print("Arquivo salvo com sucesso! Dimensões:", df_consultas.shape)
    print(df_consultas.head())


if __name__ == "__main__":
    main()