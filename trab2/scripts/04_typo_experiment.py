"""
04_typo_experiment.py
=====================
Experimento de tolerância a erros (requisito 4 do enunciado).

Executa:

    python scripts/04_typo_experiment.py

Procedimento:
  1. Constrói o corretor (k-gramas + Jaccard + distância de edição) usando o
     vocabulário da coleção ANTES do stemming (vocab salvo no passo 02).
  2. Seleciona 10 consultas do test com NDCG@10 (original) > 0.
  3. Para cada consulta, introduz um erro de um tipo (inserção, remoção,
     substituição ou troca de ordem), garantindo que o termo com erro NÃO
     exista no dicionário (exigência (ii)).
  4. Compara NDCG@10 em três condições: original, com erro e corrigida.
  5. Calcula a recuperação:
         (NDCG_corrigida - NDCG_erro) / (NDCG_original - NDCG_erro)
     quando o denominador for > 0; senão registra que o erro não reduziu o
     desempenho.

Arquivo gerado:
    outputs/typo_experiment.csv
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_loader import load_all, get_query_ids_from_qrels  # noqa: E402
from src.evaluation import evaluate_query  # noqa: E402
from src.inverted_index import InvertedIndex  # noqa: E402
from src.lexical_search import BM25  # noqa: E402
from src.preprocessing import ensure_nltk_data, preprocess  # noqa: E402
from src.spell_correction import SpellCorrector, introduce_typo  # noqa: E402


# Configuração usada para a tokenização de "superfície" (sem stemming/stopwords).
SURFACE_CFG = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_digits": False,
    "remove_stopwords": False,
    "stem": False,
    "lemmatize": False,
}

# Tipos de erro distribuídos pelas 10 consultas (cobrindo os 4 tipos).
TYPO_TYPES = ["insertion", "deletion", "substitution", "transposition",
              "insertion", "deletion", "substitution", "transposition",
              "insertion", "deletion"]


def surface_tokens(text: str) -> list:
    """Tokenização de superfície (minúsculas, sem pontuação, sem stemming)."""
    return preprocess(text, cfg=SURFACE_CFG)


def pick_typo_token(tokens: list, vocabulary: set, frequencies: dict,
                    stopwords: set) -> int | None:
    """
    Escolhe o índice de um token adequado para receber o erro.

    Estratégia: prefere termos LONGOS (>= 6) e mais RAROS (menor DF), pois são
    mais distintivos — o erro derruba mais o ranking — e menos ambíguos — a
    correção recupera com mais facilidade. Retorna o índice ou None.
    """
    candidates = [
        (frequencies.get(tok, 0), i)
        for i, tok in enumerate(tokens)
        if len(tok) >= 6 and tok in vocabulary and tok not in stopwords
    ]
    if candidates:
        candidates.sort()  # menor DF (mais raro) primeiro
        return candidates[0][1]

    # Fallback 1: qualquer token do vocabulário com tamanho >= 4.
    for i, tok in enumerate(tokens):
        if len(tok) >= 4 and tok in vocabulary and tok not in stopwords:
            return i

    # Fallback 2: qualquer token do vocabulário com tamanho >= 3 (siglas curtas).
    for i, tok in enumerate(tokens):
        if len(tok) >= 3 and tok in vocabulary:
            return i
    return None


def make_unknown_typo(token: str, typo_type: str, vocabulary: set,
                      rng: random.Random) -> str:
    """
    Gera um erro em `token` do tipo `typo_type`, garantindo que o resultado
    NÃO esteja no vocabulário (exigência (ii)). Tenta até 20 posições.
    """
    for _ in range(20):
        typo = introduce_typo(token, typo_type, rng)
        if typo != token and typo not in vocabulary:
            return typo
    # Último recurso: substituição simples forçando caractere diferente.
    for ch in "xqz":
        typo = token[:-1] + ch
        if typo not in vocabulary:
            return typo
    return token + "x"  # extremamente improvável estar no vocabulário


def main() -> None:
    ensure_nltk_data()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)  # reprodutibilidade

    print("[1/5] Carregando dados e índice ...")
    corpus, queries, qrels_dev, qrels_test = load_all(
        config.CORPUS_FILE, config.QUERIES_FILE,
        config.QRELS_DEV_FILE, config.QRELS_TEST_FILE,
    )
    test_ids = get_query_ids_from_qrels(qrels_test)
    index = InvertedIndex.load(config.INDEX_DIR / "inverted_index.pkl")
    bm25 = BM25(index)

    print("[2/5] Carregando vocabulário (antes do stemming) e construindo corretor ...")
    with open(config.INDEX_DIR / "spell_vocabulary.json", "r", encoding="utf-8") as f:
        vocab_freq = json.load(f)  # {termo: document_frequency}
    vocabulary = set(vocab_freq.keys())
    corrector = SpellCorrector(
        ngram_size=config.SPELL["kgram_size"],
        jaccard_threshold=config.SPELL["jaccard_threshold"],
        max_candidates=config.SPELL["max_candidates"],
        max_edit_distance=config.SPELL["max_edit_distance"],
    )
    corrector.build_dictionary(vocabulary, frequencies=vocab_freq)

    from nltk.corpus import stopwords
    stopwords_set = set(stopwords.words("english"))

    print("[3/5] Calculando NDCG@10 original das consultas do test ...")
    ndcg_original = {}
    for qid in test_ids:
        ranked = bm25.search(preprocess(queries[qid]))
        doc_ids = [d for d, _ in ranked]
        ndcg_original[qid] = evaluate_query(doc_ids, qrels_test[qid], 10)["NDCG"]

    # Seleciona 10 consultas com NDCG@10 > 0 (exigência (iii)), determinístico.
    candidates = sorted(qid for qid in test_ids if ndcg_original[qid] > 0)
    selected = candidates[:10]
    print(f"      {len(candidates)} consultas com NDCG>0; 10 selecionadas: {selected}")

    print("[4/5] Introduzindo erros e corrigindo ...")
    rows = []
    for idx, qid in enumerate(selected):
        typo_type = TYPO_TYPES[idx % len(TYPO_TYPES)]
        q_text = queries[qid]
        tokens = surface_tokens(q_text)
        ti = pick_typo_token(tokens, vocabulary, vocab_freq, stopwords_set)

        if ti is None:
            # Não há termo adequado; registra e pula (não deve ocorrer na prática).
            rows.append({"query_id": qid, "query_original": q_text,
                         "query_erro": "", "query_corrigida": "",
                         "tipo_erro": typo_type, "NDCG10_original": ndcg_original[qid],
                         "NDCG10_erro": None, "NDCG10_corrigida": None,
                         "recuperacao": None, "observacao": "sem termo p/ erro"})
            continue

        original_term = tokens[ti]
        typo_term = make_unknown_typo(original_term, typo_type, vocabulary, rng)

        # Monta as três versões (em superfície, unidas por espaço).
        typo_tokens = tokens[:]
        typo_tokens[ti] = typo_term

        # Corrige APENAS o termo com erro (os demais já são conhecidos e não
        # devem ser alterados). Isso evita, por exemplo, que uma stopword seja
        # "corrigida" indevidamente.
        corrected_tokens = typo_tokens[:]
        corrected = corrector.correct(typo_term)
        if corrected is not None:
            corrected_tokens[ti] = corrected

        query_erro = " ".join(typo_tokens)
        query_corrigida = " ".join(corrected_tokens)

        # NDCG@10 das três condições (aplicando o pré-processamento com stemming).
        def ndcg(text):
            ranked = bm25.search(preprocess(text))
            return evaluate_query([d for d, _ in ranked], qrels_test[qid], 10)["NDCG"]

        ndcg_orig = ndcg_original[qid]
        ndcg_err = ndcg(query_erro)
        ndcg_corr = ndcg(query_corrigida)

        # Recuperação (exigência (iv)).
        denom = ndcg_orig - ndcg_err
        if denom > 0:
            recuperacao = (ndcg_corr - ndcg_err) / denom
            obs = ""
        else:
            recuperacao = None
            obs = "erro não reduziu o desempenho"

        rows.append({
            "query_id": qid,
            "query_original": q_text,
            "query_erro": query_erro,
            "query_corrigida": query_corrigida,
            "tipo_erro": typo_type,
            "termo_alterado": original_term,
            "termo_com_erro": typo_term,
            "NDCG10_original": ndcg_orig,
            "NDCG10_erro": ndcg_err,
            "NDCG10_corrigida": ndcg_corr,
            "recuperacao": recuperacao,
            "observacao": obs,
        })

    print("[5/5] Salvando typo_experiment.csv ...")
    df = pd.DataFrame(rows)
    out = config.OUTPUT_DIR / "typo_experiment.csv"
    df.to_csv(out, index=False)
    print(f"      -> {out}")

    # Resumo rápido no console.
    def _fmt(x):
        return "n/a" if x is None else f"{x:.4f}"

    print("\n--- Resumo (NDCG@10) ---")
    for r in rows:
        rec = "n/a" if r["recuperacao"] is None else f"{r['recuperacao']:.3f}"
        print(f"  {r['query_id']:12s} {r['tipo_erro']:14s} "
              f"orig={_fmt(r['NDCG10_original'])} erro={_fmt(r['NDCG10_erro'])} "
              f"corr={_fmt(r['NDCG10_corrigida'])} recuperação={rec}")


if __name__ == "__main__":
    main()
