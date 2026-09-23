"""
Comparação entre modelos (Vetorial x BM25)..

Saídas em results/comparacao_modelos/:
- map_agregado.png       -> MAP de cada modelo, nas 4 configs de pré-processamento
- maiores_diferencas.csv -> AP por consulta (Vetorial x BM25) ordenado pela
                             maior diferença absoluta, para embasar a discussão
                             de hipóteses no relatório
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

DEL_STOPWORDS, USE_STEMMING = 1, 1  # melhor config de pré-processamento
RESULTS_DIR = "results/comparacao_modelos"


def gerar_grafico_map_agregado(df_map, path):
    """MAP de cada modelo nas 4 configurações de pré-processamento."""
    df_map = df_map.copy()
    df_map["Config"] = (
        df_map["Stopwords_Removidas"].map({True: "SW", False: "s/SW"})
        + " + "
        + df_map["Stemming_Aplicado"].map({True: "Stem", False: "s/Stem"})
    )
    pivot = df_map.pivot(index="Config", columns="Modelo", values="MAP")

    fig, ax = plt.subplots(figsize=(7, 5))
    pivot.plot(kind="bar", ax=ax)
    ax.set_title("MAP por modelo e configuração de pré-processamento")
    ax.set_ylabel("MAP")
    ax.set_xlabel("")
    ax.legend(title="Modelo")
    ax.grid(True, alpha=0.3, axis="y")
    plt.xticks(rotation=0)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gerar_maiores_diferencas(df_exp, path, top_n=10):
    """Para a melhor config, pivota AP por consulta (Vetorial x BM25) e
    ordena pela maior diferença absoluta."""
    df = df_exp[
        (df_exp["Stopwords_Removidas"] == DEL_STOPWORDS)
        & (df_exp["Stemming_Aplicado"] == USE_STEMMING)
    ]
    pivot = df.pivot(index="Query_ID", columns="Modelo", values="AP")
    pivot["Diferenca_BM25_menos_Vetorial"] = pivot["BM25"] - pivot["Vetorial"]
    pivot["Diferenca_Absoluta"] = pivot["Diferenca_BM25_menos_Vetorial"].abs()
    pivot = pivot.sort_values("Diferenca_Absoluta", ascending=False)

    pivot.to_csv(path)

    print(f"\n=== Top {top_n} consultas com maior diferença de AP (Vetorial x BM25) ===")
    print(pivot.head(top_n).to_string())


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    df_map = pd.read_csv("results/results_default_queries_map.csv")
    df_exp = pd.read_csv("results/results_default_queries.csv")

    print("=== MAP agregado (todas as configs) ===")
    print(df_map.to_string(index=False))

    gerar_grafico_map_agregado(df_map, os.path.join(RESULTS_DIR, "map_agregado.png"))
    gerar_maiores_diferencas(df_exp, os.path.join(RESULTS_DIR, "maiores_diferencas.csv"))

    print(f"\nArquivos salvos em {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
