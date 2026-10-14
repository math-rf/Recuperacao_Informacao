"""
Variação dos parâmetros do BM25 (k1 e b).

Usa a configuração de pré-processamento com melhor MAP encontrada antes
(stopwords removidas + stemming). Toda métrica vem de avaliar_sistema
(ir_utils.py) — nada é recalculado na mão.

Compara as configurações usando duas métricas: MAP e F1@10.

Saídas em results/bm25_params/:
- bm25_params_map_f1.csv        -> tabela com MAP e F1@10 de cada (k1, b)
- bm25_params_map.png           -> MAP x b, uma linha por k1
- bm25_params_f1.png            -> F1@10 x b, uma linha por k1
- bm25_params_map_vs_f1.png     -> MAP e F1@10 lado a lado por combinação (k1, b)
- bm25_confusion_matrix.png     -> matriz de confusão da melhor combinação
                                    (maior MAP; empate desempatado por F1@10)
- bm25_query_sensivel_a_b.txt   -> consulta cujo AP mais muda entre b=0 e b=1,
                                    com o Caso: FP em b=0 -> TN em b=1, contexto do documento
"""
import os
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ir_utils import carregar_cranfield, modelo_bm25, avaliar_sistema, ranking_para_doc_ids

DEL_STOPWORDS, USE_STEMMING = True, True  # melhor configuração encontrada em main.py
GRADE = {"k1": [0.5, 1.2, 2.0], "b": [0.0, 0.75, 1.0]}
K1_ANALISE_B, B_BASE, B_COMPARACAO = 1.2, 0.0, 1.0
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "bm25_params")


def rodar_configuracao(k1, b, doc_ids, corpus, qrels, consultas):
    """Roda o BM25 com (k1, b) em todas as consultas; devolve `resultados` no
    formato exigido por avaliar_sistema: {q_id: {recuperados, relevantes}}."""
    resultados = {}
    for q_id, _texto_original, q_texto in consultas:
        ranking = modelo_bm25(corpus, q_texto, k1=k1, b=b)
        resultados[q_id] = {
            "recuperados": ranking_para_doc_ids(ranking, doc_ids),
            "relevantes": qrels[q_id],
        }
    return resultados


def sweep(doc_ids, corpus, qrels, consultas):
    """MAP e F1@10 de cada combinação de GRADE, via avaliar_sistema."""
    linhas, resultados_por_combo = [], {}
    for k1, b in itertools.product(*GRADE.values()):
        print(f"Rodando BM25 com k1={k1}, b={b}...")
        resultados = rodar_configuracao(k1, b, doc_ids, corpus, qrels, consultas)
        _, map_score, f1_score = avaliar_sistema(resultados, k=10)
        linhas.append({"k1": k1, "b": b, "MAP": map_score, "F1@10": f1_score})
        resultados_por_combo[(k1, b)] = resultados
    return pd.DataFrame(linhas), resultados_por_combo


def gerar_grafico_linha(df_map, path, metrica="MAP"):
    fig, ax = plt.subplots(figsize=(7, 5))
    for k1 in sorted(df_map["k1"].unique()):
        subset = df_map[df_map["k1"] == k1].sort_values("b")
        ax.plot(subset["b"], subset[metrica], marker="o", label=f"k1={k1}")
    ax.set(title=f"{metrica} em função de b", xlabel="b", ylabel=metrica)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gerar_grafico_comparativo(df_map, path):
    """Compara MAP e F1@10 lado a lado para cada combinação (k1, b)."""
    df_ordenado = df_map.sort_values(["k1", "b"]).reset_index(drop=True)
    labels = [f"k1={row.k1}\nb={row.b}" for row in df_ordenado.itertuples()]
    x = np.arange(len(df_ordenado))
    largura = 0.35

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - largura / 2, df_ordenado["MAP"], largura, label="MAP")
    ax.bar(x + largura / 2, df_ordenado["F1@10"], largura, label="F1@10")
    ax.set(title="MAP vs F1@10 por combinação (k1, b)", ylabel="Score")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def gerar_matriz_confusao(df_map, resultados_por_combo, n_docs, path, k=10):
    """Matriz de confusão (só números, sem cor) da melhor combinação.

    A melhor combinação é escolhida pelo maior MAP; em caso de empate,
    desempata pelo maior F1@10 (segunda métrica de comparação)."""
    melhor = df_map.sort_values(["MAP", "F1@10"], ascending=False).iloc[0]
    resultados = resultados_por_combo[(melhor["k1"], melhor["b"])]

    tp = fp = fn = 0
    for dados in resultados.values():
        retrieved, relevantes = set(dados["recuperados"][:k]), set(dados["relevantes"])
        tp += len(retrieved & relevantes)
        fp += len(retrieved - relevantes)
        fn += len(relevantes - retrieved)
    tn = n_docs * len(resultados) - tp - fp - fn

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.axis("off")
    tabela = ax.table(
        cellText=[[str(tp), str(fp)], [str(fn), str(tn)]],
        rowLabels=["Real: Relevante", "Real: Nao relevante"],
        colLabels=["Predito: Recuperado@10", "Predito: Nao recuperado"],
        cellLoc="center", loc="center",
    )
    tabela.auto_set_font_size(False)
    tabela.set_fontsize(11)
    tabela.scale(1, 2.2)
    ax.set_title(
        f"Matriz de confusao - melhor config (k1={melhor['k1']}, b={melhor['b']}) "
        f"| MAP={melhor['MAP']:.4f}  F1@10={melhor['F1@10']:.4f}",
        pad=20,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def gerar_relatorio_query_sensivel(
    resultados_por_combo, doc_ids, corpus, gabarito, consultas, textos_doc, path
):
    """Consulta cujo AP (avaliar_sistema) mais varia entre B_BASE e B_COMPARACAO;
    imprime o Caso 1: doc falso positivo em base (Top-10, mas nao relevante)
    que vira verdadeiro negativo em comp (sai do Top-10), com o texto do doc."""
    resultados_base = resultados_por_combo[(K1_ANALISE_B, B_BASE)]
    resultados_comp = resultados_por_combo[(K1_ANALISE_B, B_COMPARACAO)]

    metricas_base, _, _ = avaliar_sistema(resultados_base, k=10)
    metricas_comp, _, _ = avaliar_sistema(resultados_comp, k=10)
    q_id = max(metricas_base, key=lambda q: abs(metricas_comp[q]["AP"] - metricas_base[q]["AP"]))

    textos_originais = {q_id_: texto for q_id_, texto, _texto_proc in consultas}
    q_texto_proc = next(texto_proc for q_id_, _texto, texto_proc in consultas if q_id_ == q_id)

    ranking_base = modelo_bm25(corpus, q_texto_proc, k1=K1_ANALISE_B, b=B_BASE)
    ranking_comp = modelo_bm25(corpus, q_texto_proc, k1=K1_ANALISE_B, b=B_COMPARACAO)

    rank_base = {doc_ids[i]: pos + 1 for pos, (i, _) in enumerate(ranking_base)}
    rank_comp = {doc_ids[i]: pos + 1 for pos, (i, _) in enumerate(ranking_comp)}
    top10_base = list(rank_base)[:10]
    top10_comp = list(rank_comp)[:10]
    relevantes = {d for d, rel in gabarito.get(q_id, {}).items() if rel >= 1}

    caso1 = next((d for d in top10_base if d not in relevantes and d not in top10_comp), None)

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Consulta mais sensivel a variacao de b (k1={K1_ANALISE_B})\n")
        f.write(f"Query_ID: {q_id}\n")
        f.write(f"Texto da consulta: {textos_originais[q_id]}\n")
        f.write(f"Config base: b={B_BASE}    Config comparacao: b={B_COMPARACAO}\n")

        f.write(f"\n--- Caso 1: Falso Positivo em b={B_BASE} -> Verdadeiro Negativo em b={B_COMPARACAO} ---\n")
        if caso1:
            f.write(f"Doc_ID: {caso1}\n")
            f.write(f"Rank em b={B_BASE}: {rank_base[caso1]} (no Top-10, mas NAO e relevante)\n")
            f.write(f"Rank em b={B_COMPARACAO}: {rank_comp.get(caso1, '> 10')} (fora do Top-10, corretamente)\n")
            f.write(f"Texto do documento: {textos_doc[caso1]}\n")
        else:
            f.write("Nao encontrado para esta consulta.\n")

        f.write(f"\nTop-10 com b={B_BASE}: {', '.join(top10_base)}\n")
        f.write(f"Top-10 com b={B_COMPARACAO}: {', '.join(top10_comp)}\n")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    doc_ids, corpus, textos_doc, qrels, gabarito, consultas = carregar_cranfield(
        DEL_STOPWORDS, USE_STEMMING
    )

    df_map, resultados_por_combo = sweep(doc_ids, corpus, qrels, consultas)
    print("\n=== MAP e F1@10 por combinação de parâmetros ===")
    print(df_map.sort_values("MAP", ascending=False).to_string(index=False))
    df_map.to_csv(os.path.join(RESULTS_DIR, "bm25_params_map_f1.csv"), index=False)

    gerar_grafico_linha(df_map, os.path.join(RESULTS_DIR, "bm25_params_map.png"), metrica="MAP")
    gerar_grafico_linha(df_map, os.path.join(RESULTS_DIR, "bm25_params_f1.png"), metrica="F1@10")
    gerar_grafico_comparativo(df_map, os.path.join(RESULTS_DIR, "bm25_params_map_vs_f1.png"))
    gerar_matriz_confusao(df_map, resultados_por_combo, len(doc_ids),
                           os.path.join(RESULTS_DIR, "bm25_confusion_matrix.png"))
    gerar_relatorio_query_sensivel(resultados_por_combo, doc_ids, corpus, gabarito, consultas,
                                    textos_doc, os.path.join(RESULTS_DIR, "bm25_query_sensivel_a_b.txt"))

    print(f"\n6 arquivos de saída salvos em {RESULTS_DIR}/")


if __name__ == "__main__":
    main()