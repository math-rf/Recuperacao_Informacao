"""
Análise por consulta.

Identifica:
  i)   2 consultas em que o BM25 é claramente superior ao Modelo Vetorial
  ii)  2 consultas em que o Modelo Vetorial é claramente superior ao BM25
  iii) 2 consultas em que ambos têm desempenho insatisfatório

Critério: F1@10.

Não roda os modelos de novo: lê o CSV que main.py já gerou
(results/results_default_queries.csv), com métricas e Top-10 (doc_id,
score, relevância) por consulta/modelo/configuração.
"""
import ast
import os
import re
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
PASTA_SAIDA = os.path.join(RESULTS_DIR, "comparacao_casosModelos")

DEL_STOPWORDS, USE_STEMMING = 1, 1
METRICA = "F1@10"
MODELOS = ("Vetorial", "BM25")


def parse_top10(texto):
    """Top_10 foi salvo como str de tuplas com np.float64(...); remove
    esse wrapper e faz o parse com literal_eval."""
    return ast.literal_eval(re.sub(r"np\.float64\(([^)]+)\)", r"\1", texto))


def carregar_comparacao():
    """Monta uma linha por consulta com texto, F1@10 e Top-10 de cada
    modelo, já filtrando pela configuração escolhida."""

    df = pd.read_csv(os.path.join(RESULTS_DIR, "results_default_queries.csv"))
    df = df[(df["Stopwords_Removidas"] == DEL_STOPWORDS) & (df["Stemming_Aplicado"] == USE_STEMMING)]
    df["Top_10"] = df["Top_10"].apply(parse_top10)
    textos = pd.read_csv(os.path.join(RESULTS_DIR, "cranfield_queries_texts.csv")).set_index("Query_ID")["Texto_Consulta"]

    dados = {"Texto": textos}
    for modelo in MODELOS:
        por_modelo = df[df["Modelo"] == modelo].set_index("Query_ID")
        dados[f"F1_{modelo}"] = por_modelo[METRICA]
        dados[f"Top10_{modelo}"] = por_modelo["Top_10"]

    comparacao = pd.DataFrame(dados).dropna(subset=[f"F1_{m}" for m in MODELOS])
    comparacao["Diferenca_BM25_menos_Vetorial"] = comparacao["F1_BM25"] - comparacao["F1_Vetorial"]
    comparacao["Soma_F1"] = comparacao["F1_Vetorial"] + comparacao["F1_BM25"]
    return comparacao


def salvar_figura(numero_caso, q_id, row):
    """Gera uma imagem com o Top-10 de Vetorial e BM25 lado a lado para
    uma consulta, com os documentos relevantes destacados em verde """

    fig, eixos = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(f"Caso {numero_caso} - Query {q_id}: \"{row['Texto']}\"", fontsize=9, wrap=True)

    for eixo, modelo in zip(eixos, MODELOS):
        top10 = row[f"Top10_{modelo}"]
        celulas = [[str(pos), str(doc_id), f"{score:.4f}", str(rel)]
                   for pos, (doc_id, score, rel) in enumerate(top10, start=1)]

        eixo.axis("off")
        eixo.set_title(f"{modelo}  (F1@10={row[f'F1_{modelo}']:.4f})", fontsize=10)
        tabela = eixo.table(
            cellText=celulas,
            colLabels=["#", "Doc", "Score", "Rel"],
            loc="center", cellLoc="center",
        )
        tabela.auto_set_font_size(False)
        tabela.set_fontsize(8)
        tabela.scale(1, 1.3)

        # Pinta de verde claro as linhas de documentos relevantes (rel >= 1)
        for (linha, _), celula in tabela.get_celld().items():
            if linha == 0:
                continue
            rel = int(celulas[linha - 1][3])
            if rel >= 1:
                celula.set_facecolor("#d9f2d9")

    plt.tight_layout()
    caminho = f"{PASTA_SAIDA}/query_analysis_caso{numero_caso}_q{q_id}.png"
    plt.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def main():
    os.makedirs(PASTA_SAIDA, exist_ok=True)
    comparacao = carregar_comparacao()

    # Para i e ii, restringe às consultas em que NENHUM modelo teve F1=0 para casos de discordância mais interessantes
    ambos_relevantes = comparacao[(comparacao["F1_Vetorial"] > 0) & (comparacao["F1_BM25"] > 0)]

    casos = [
        (1, "BM25 claramente superior ao Vetorial",
         ambos_relevantes.sort_values("Diferenca_BM25_menos_Vetorial", ascending=False).head(2)),
        (2, "Vetorial claramente superior ao BM25",
         ambos_relevantes.sort_values("Diferenca_BM25_menos_Vetorial", ascending=True).head(2)),
        (3, "Ambos os modelos performam mal",
         comparacao.sort_values("Soma_F1", ascending=True).head(2)),
    ]

    linhas_csv = []
    for numero_caso, titulo, subset in casos:
        print("=" * 70, f"{numero_caso}) {titulo}", "=" * 70, sep="\n")
        for q_id, row in subset.iterrows():
            print(f"\nQuery {q_id}: \"{row['Texto']}\"")
            caminho_figura = salvar_figura(numero_caso, q_id, row)
            print(f"  (figura salva em {caminho_figura})")
            for modelo in MODELOS:
                print(f"  [{modelo}] F1@10={row[f'F1_{modelo}']:.4f}")
                for pos, (doc_id, score, rel) in enumerate(row[f"Top10_{modelo}"], start=1):
                    marca = "RELEVANTE" if rel >= 1 else "não relevante"
                    print(f"    {pos}. doc {doc_id} (score={score:.4f}) -> {marca} (rel={rel})")
                    linhas_csv.append({
                        "Caso": numero_caso, "Query_ID": q_id, "Modelo": modelo,
                        "F1@10": row[f"F1_{modelo}"],
                        "Posicao": pos, "Doc_ID": doc_id, "Score": score, "Relevancia": rel,
                        "Texto": row["Texto"],
                    })
        print()

    pd.DataFrame(linhas_csv).to_csv(f"{PASTA_SAIDA}/comparacao_vet_vs_bm25_por_query.csv", index=False)


if __name__ == "__main__":
    main()