"""
Item 9: Análise de erros.

Para cada modelo (Vetorial e BM25), na melhor configuração (stopwords
removidas + stemming), procura UMA query que tenha, ao mesmo tempo:
  - pelo menos 2 documentos NÃO relevantes dentro do Top-K do ranking;
  - pelo menos 1 documento RELEVANTE fora do Top-K.

Para essa query, recupera os 3 documentos (2 não relevantes + 1 relevante
fora do topo), mostrando a posição de cada um no ranking. As queries podem
ser diferentes entre os modelos. No total são recuperados 6 documentos
(3 por modelo).

Para cada documento, imprime:
  - Modelo
  - Query (antes da limpeza)
  - Query limpa vetorizada (lista de termos após stopwords + stemming)
  - Documento (relevante ou não) com sua posição no ranking

Coloque este arquivo dentro da pasta trab1/ (junto de ir_utils.py).
"""
import os
from ir_utils import carregar_cranfield, executar_modelo

DEL_STOPWORDS = True
USE_STEMMING = True
TOP_K = 10
N_NAO_RELEVANTES_NECESSARIOS = 2
N_RELEVANTES_FORA_NECESSARIOS = 1
ARQUIVO_SAIDA = os.path.join(os.path.dirname(__file__), "results", "error_analysis_output.txt")
SEPARADOR = "=" * 78


def formatar_query_limpa(q_texto_proc):
    termos = q_texto_proc.split()
    return "[" + ", ".join(termos) + "]"


def buscar_query_com_os_3_casos(nome_modelo, doc_ids, corpus_strings, qrels, gabarito, consultas):
    """Procura, para um modelo, UMA query cujo ranking contenha ao mesmo
    tempo >=2 documentos não relevantes no Top-K e >=1 documento relevante
    fora do Top-K. Retorna os 3 casos encontrados para essa query."""
    for q_id, q_texto_original, q_texto_proc in consultas:
        ranking = executar_modelo(nome_modelo, corpus_strings, q_texto_proc)
        top_k_ids = [doc_ids[idx] for idx, _ in ranking[:TOP_K]]

        nao_relevantes_da_query = []
        for pos, doc_id in enumerate(top_k_ids, start=1):
            rel = gabarito.get(q_id, {}).get(doc_id, 0)
            if rel < 1:
                nao_relevantes_da_query.append({
                    "doc_id": doc_id, "relevancia_real": rel, "posicao_no_ranking": pos,
                })
            if len(nao_relevantes_da_query) >= N_NAO_RELEVANTES_NECESSARIOS:
                break

        if len(nao_relevantes_da_query) < N_NAO_RELEVANTES_NECESSARIOS:
            continue

        relevante_fora_da_query = None
        for doc_id in qrels.get(q_id, []):
            if doc_id not in top_k_ids:
                posicao_real = next(
                    (p + 1 for p, (idx, _) in enumerate(ranking) if doc_ids[idx] == doc_id), None
                )
                relevante_fora_da_query = {
                    "doc_id": doc_id,
                    "relevancia_real": gabarito.get(q_id, {}).get(doc_id, 0),
                    "posicao_no_ranking": posicao_real,
                }
                break

        if relevante_fora_da_query is None:
            continue

        casos = [dict(c, relevante=False) for c in nao_relevantes_da_query]
        casos.append(dict(relevante_fora_da_query, relevante=True))

        for caso in casos:
            caso["query_id"] = q_id
            caso["query_texto_original"] = q_texto_original
            caso["query_texto_limpo"] = q_texto_proc

        return casos

    return []  # nenhuma query do modelo atende às duas condições simultaneamente


def formatar_caso(nome_modelo, caso, doc_textos_originais):
    status = "RELEVANTE (fora do Top-K)" if caso["relevante"] else "NÃO relevante (dentro do Top-K)"
    return "\n".join([
        SEPARADOR,
        f"Modelo: {nome_modelo}",
        "",
        f"Query (antes da limpeza):",
        f"  {caso['query_texto_original']}",
        "",
        f"Query limpa vetorizada:",
        f"  {formatar_query_limpa(caso['query_texto_limpo'])}",
        "",
        f"Documento {status}",
        f"  doc_id: {caso['doc_id']}",
        f"  relevância real: {caso['relevancia_real']}",
        f"  posição no ranking: {caso['posicao_no_ranking']}",
        "",
        "Texto do documento:",
        doc_textos_originais[caso['doc_id']].strip(),
        "",
    ])


def main():
    doc_ids, corpus_strings, doc_textos_originais, qrels, gabarito, consultas = carregar_cranfield(
        DEL_STOPWORDS, USE_STEMMING
    )

    os.makedirs(os.path.dirname(ARQUIVO_SAIDA), exist_ok=True)
    with open(ARQUIVO_SAIDA, "w", encoding="utf-8") as arquivo:
        for nome_modelo in ["Vetorial", "BM25"]:
            casos = buscar_query_com_os_3_casos(
                nome_modelo, doc_ids, corpus_strings, qrels, gabarito, consultas
            )

            for caso in casos:
                bloco = formatar_caso(nome_modelo, caso, doc_textos_originais)
                print(bloco)
                arquivo.write(bloco + "\n")

    print(f"Resultado salvo em: {ARQUIVO_SAIDA}")


if __name__ == "__main__":
    main()