"""
Métricas médias (MAP, F1@10, P@10, R@10) por modelo e configuração de
pré-processamento.

Não roda os modelos de novo: lê o CSV que main.py já gerou
(results/results_default_queries.csv), que tem P@10, R@10, F1@10 e AP por
consulta, e agrega (média) por Modelo x configuração de pré-processamento.
AP médio = MAP (mesma definição usada em ir_utils.avaliar_sistema).

Gera um único gráfico combinado (eixo x = configuração de pré-processamento)
com uma linha colorida por métrica (MAP, F1@10, P@10, R@10) e um estilo de
linha/marcador por modelo (contínua/círculo = BM25, tracejada/quadrado =
Vetorial), para comparar os dois modelos sem duplicar o gráfico nem multiplicar
o número de cores.

Saídas em results/metricas_medias/:
- metricas_medias.csv    -> tabela com MAP, F1@10, P@10 e R@10 médios,
                             por Modelo e configuração
- metricas_combinado.png -> MAP, F1@10, P@10 e R@10 x configuração,
                             Vetorial e BM25 no mesmo gráfico
"""
import os

import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PASTA_SAIDA = os.path.join(RESULTS_DIR, "metricas_medias")

# Ordem fixa das 4 configurações de pré-processamento (mesma de main.py),
# para as linhas do gráfico seguirem sempre a mesma ordem no eixo x.
ORDEM_CONFIGS = ["s/SW + s/Stem", "SW + s/Stem", "s/SW + Stem", "SW + Stem"]

# Métricas plotadas, cada uma com uma cor fixa.
METRICAS = ["MAP", "F1@10", "P@10", "R@10"]
CORES = {"MAP": "tab:blue", "F1@10": "tab:orange", "P@10": "tab:green", "R@10": "tab:red"}

# Modelo -> (estilo de linha, marcador): diferencia o modelo sem precisar de
# cores extras, já que a cor está reservada para a métrica.
ESTILOS_MODELO = {"BM25": ("-", "o"), "Vetorial": ("--", "s")}

NOME_ARQUIVO_SAIDA = "metricas_combinado.png"


def rotular_config(row):
    sw = "SW" if row["Stopwords_Removidas"] else "s/SW"
    stem = "Stem" if row["Stemming_Aplicado"] else "s/Stem"
    return f"{sw} + {stem}"


def calcular_medias(df_exp):
    """Agrupa por Modelo + configuração e calcula a média de P@10, R@10,
    F1@10 e AP (== MAP). Uma linha por Modelo x configuração."""
    df = df_exp.copy()
    df["Config"] = df.apply(rotular_config, axis=1)

    agregados = (
        df.groupby(["Modelo", "Config"])
        .agg(
            **{
                "MAP": ("AP", "mean"),
                "F1@10": ("F1@10", "mean"),
                "P@10": ("P@10", "mean"),
                "R@10": ("R@10", "mean"),
            }
        )
        .reset_index()
    )
    agregados["Config"] = pd.Categorical(agregados["Config"], categories=ORDEM_CONFIGS, ordered=True)
    return agregados.sort_values(["Modelo", "Config"])


def gerar_grafico_combinado(df_medias, path):
    """Um único gráfico: cor = métrica, estilo/marcador de linha = modelo.
    Duas legendas separadas (métrica e modelo) para não poluir com 8 rótulos
    misturados."""
    fig, ax = plt.subplots(figsize=(8, 5.5))

    for modelo, (linestyle, marker) in ESTILOS_MODELO.items():
        subset = df_medias[df_medias["Modelo"] == modelo].sort_values("Config")
        for metrica in METRICAS:
            ax.plot(
                subset["Config"].astype(str), subset[metrica],
                linestyle=linestyle, marker=marker, color=CORES[metrica],
                markersize=6, linewidth=1.8,
            )

    ax.set(title="Métricas médias por configuração de pré-processamento", xlabel="", ylabel="Score")
    ax.grid(True, alpha=0.3)

    # Legenda 1: cor -> métrica
    legenda_metricas = [
        plt.Line2D([0], [0], color=CORES[m], linewidth=2, label=m) for m in METRICAS
    ]
    # Legenda 2: estilo/marcador -> modelo (em preto, só para indicar o traço)
    legenda_modelos = [
        plt.Line2D([0], [0], color="black", linestyle=linestyle, marker=marker, label=modelo)
        for modelo, (linestyle, marker) in ESTILOS_MODELO.items()
    ]

    primeira_legenda = ax.legend(handles=legenda_metricas, title="Métrica", loc="upper left")
    ax.add_artist(primeira_legenda)
    ax.legend(handles=legenda_modelos, title="Modelo", loc="upper left", bbox_to_anchor=(0, 0.72), framealpha=1)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)

    df_exp = pd.read_csv(os.path.join(RESULTS_DIR, "results_default_queries.csv"))
    df_medias = calcular_medias(df_exp)

    print("=== Métricas médias por Modelo e configuração ===")
    print(df_medias.to_string(index=False))

    df_medias.to_csv(os.path.join(PASTA_SAIDA, "metricas_medias.csv"), index=False)

    gerar_grafico_combinado(df_medias, os.path.join(PASTA_SAIDA, NOME_ARQUIVO_SAIDA))

    print(f"\nArquivos salvos em {PASTA_SAIDA}/")


if __name__ == "__main__":
    main()