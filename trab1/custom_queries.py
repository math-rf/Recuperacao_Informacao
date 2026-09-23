"""
Consultas customizadas (originais x modificadas).

Para cada par de consulta (texto original exato do Cranfield x texto
modificado manualmente), roda o Modelo Vetorial e o BM25 e analisa o que
muda no Top-10: quais documentos entram, quais saem, quais permanecem (e se
mudam de posição), e se essas mudanças aproximam ou afastam o ranking dos
documentos realmente relevantes (segundo os qrels do Cranfield).

Pré-processamento: stopwords removidas + stemming (mesma configuração de
melhor MAP usada no restante do trabalho).

Saídas em results/:
- comparacao_queries_modificadas.csv  -> Top-10 bruto (Query_ID, Modelo, Versao, Top_10)
- custom_queries_analysis.txt         -> análise legível das mudanças no Top-10
"""
import os

import pandas as pd

from ir_utils import carregar_cranfield, executar_modelo, preprocessar_texto, ranking_para_doc_ids

DEL_STOPWORDS, USE_STEMMING = True, True
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
ARQUIVO_CSV = os.path.join(RESULTS_DIR, "comparacao_queries_modificadas.csv")
ARQUIVO_ANALISE = os.path.join(RESULTS_DIR, "custom_queries_analysis.txt")
SEPARADOR = "=" * 78

# Formato: 'ID_da_Consulta': ('Texto Original Exato', 'Texto Modificado')
CONSULTAS_CUSTOMIZADAS = {
    '1': (
        "what similarity laws must be obeyed when constructing aeroelastic models of heated high speed aircraft .",
        "similarity laws aeroelastic models heated high speed aircraft"  # Reduzido apenas para palavras-chave
    ),
    '2': (
        "what are the structural and aeroelastic problems associated with flight of high speed aircraft .",
        "structural aeroelastic problems flight supersonic aircraft"  # Troca de "high speed" por sinônimo "supersonic"
    ),
    '3': (
        "can a criterion be developed to show empirically the validity of flow solutions for chemically reacting gas mixtures based on the assumption of thermodynamic equilibrium .",
        "validity of flow solutions chemically reacting gas mixtures thermodynamic equilibrium"
    ),
    '4': (
        "what is the present state of the art in fluid mechanics as applied to re-entry vehicles.",
        "fluid mechanics applied to re-entry vehicles spacecraft"  # Expansão adicionando um termo específico
    ),
    '5': (
        "has anyone developed an analysis which accurately establishes the large deflection behavior of conical shells .",
        "analysis large deflection behavior conical shells"
    ),
}


def rel_tag(gabarito, q_id, doc_id):
    """Marca um doc_id como Relevante/Nao relevante para a query q_id."""
    return "Relevante" if gabarito.get(q_id, {}).get(doc_id, 0) >= 1 else "Nao relevante"


def analisar_mudancas(q_id, top_orig, top_mod, gabarito):
    """Compara dois Top-10 e devolve um dicionário com o que mudou."""
    set_orig, set_mod = set(top_orig), set(top_mod)
    comuns = set_orig & set_mod

    entraram = [d for d in top_mod if d not in set_orig]
    sairam = [d for d in top_orig if d not in set_mod]

    mudancas_posicao = [
        (d, top_orig.index(d) + 1, top_mod.index(d) + 1)
        for d in top_orig
        if d in comuns and top_orig.index(d) != top_mod.index(d)
    ]

    jaccard = len(comuns) / len(set_orig | set_mod) if (set_orig | set_mod) else 0.0
    relevantes_entraram = sum(1 for d in entraram if gabarito.get(q_id, {}).get(d, 0) >= 1)
    relevantes_sairam = sum(1 for d in sairam if gabarito.get(q_id, {}).get(d, 0) >= 1)

    return {
        "jaccard": jaccard,
        "n_comuns": len(comuns),
        "entraram": entraram,
        "sairam": sairam,
        "mudancas_posicao": mudancas_posicao,
        "relevantes_entraram": relevantes_entraram,
        "relevantes_sairam": relevantes_sairam,
    }


def formatar_bloco(q_id, modelo, txt_original, txt_modificado, top_orig, top_mod, analise, gabarito):
    linhas = [
        SEPARADOR,
        f"Consulta {q_id} | Modelo: {modelo}",
        "",
        f"Texto original:   {txt_original}",
        f"Texto modificado: {txt_modificado}",
        "",
        f"Top-10 original:   {top_orig}",
        f"Top-10 modificado: {top_mod}",
        "",
        f"Similaridade Jaccard (Top-10 original x modificado): {analise['jaccard']:.2f}  "
        f"({analise['n_comuns']}/10 documentos em comum)",
        "",
    ]

    if analise["entraram"]:
        linhas.append("Documentos que ENTRARAM no Top-10 (só na versão modificada):")
        linhas += [f"  - {d} ({rel_tag(gabarito, q_id, d)})" for d in analise["entraram"]]
    else:
        linhas.append("Nenhum documento novo entrou no Top-10.")
    linhas.append("")

    if analise["sairam"]:
        linhas.append("Documentos que SAÍRAM do Top-10 (só na versão original):")
        linhas += [f"  - {d} ({rel_tag(gabarito, q_id, d)})" for d in analise["sairam"]]
    else:
        linhas.append("Nenhum documento saiu do Top-10.")
    linhas.append("")

    if analise["mudancas_posicao"]:
        linhas.append("Documentos que permaneceram, mas mudaram de posição:")
        for d, pos_orig, pos_mod in analise["mudancas_posicao"]:
            direcao = "subiu" if pos_mod < pos_orig else "desceu"
            linhas.append(f"  - {d}: posição {pos_orig} -> {pos_mod} ({direcao})")
    else:
        linhas.append("Os documentos em comum mantiveram a mesma posição.")
    linhas.append("")

    linhas.append(
        f"Resumo: {analise['relevantes_entraram']} documento(s) relevante(s) entraram no Top-10 e "
        f"{analise['relevantes_sairam']} documento(s) relevante(s) saíram do Top-10 "
        f"com a modificação da consulta."
    )
    linhas.append("")
    return "\n".join(linhas)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    doc_ids, corpus_strings, _textos_doc, _qrels, gabarito, _consultas = carregar_cranfield(
        DEL_STOPWORDS, USE_STEMMING
    )

    print("Executando consultas originais e modificadas...")
    linhas_csv, blocos_analise = [], []

    for q_id, (txt_original, txt_modificado) in CONSULTAS_CUSTOMIZADAS.items():
        q_orig_limpa = preprocessar_texto(txt_original, DEL_STOPWORDS, USE_STEMMING)
        q_mod_limpa = preprocessar_texto(txt_modificado, DEL_STOPWORDS, USE_STEMMING)

        for modelo in ["Vetorial", "BM25"]:
            top_orig = ranking_para_doc_ids(
                executar_modelo(modelo, corpus_strings, q_orig_limpa), doc_ids, k=10
            )
            top_mod = ranking_para_doc_ids(
                executar_modelo(modelo, corpus_strings, q_mod_limpa), doc_ids, k=10
            )

            linhas_csv.append({"Query_ID": q_id, "Modelo": modelo, "Versao": "Original", "Top_10": top_orig})
            linhas_csv.append({"Query_ID": q_id, "Modelo": modelo, "Versao": "Modificada", "Top_10": top_mod})

            analise = analisar_mudancas(q_id, top_orig, top_mod, gabarito)
            blocos_analise.append(
                formatar_bloco(q_id, modelo, txt_original, txt_modificado, top_orig, top_mod, analise, gabarito)
            )

    pd.DataFrame(linhas_csv).to_csv(ARQUIVO_CSV, index=False)

    with open(ARQUIVO_ANALISE, "w", encoding="utf-8") as f:
        for bloco in blocos_analise:
            f.write(bloco + "\n")
        f.write(SEPARADOR + "\n")

    for bloco in blocos_analise:
        print(bloco)

    print("\nProcesso finalizado!")
    print(f"Arquivo '{ARQUIVO_CSV}' gerado com sucesso.")
    print(f"Arquivo '{ARQUIVO_ANALISE}' gerado com sucesso.")


if __name__ == "__main__":
    main()