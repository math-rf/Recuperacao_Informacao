"""
04_typo_experiment.py
=====================
Experimento de tolerância a erros.

1) Seleção no DEV: varre as combinações de distância (Levenshtein x Damerau-OSA),
   dicionário (com/sem stopwords) e limiar de Jaccard. Critério: maior
   recuperação média (NDCG@10) no dev. Em caso de empate, vence a primeira
   configuração da grade (Damerau + dicionário completo, Jaccard 0,2).
2) Aplicação no TEST: 10 consultas com NDCG@10 original > 0 (as 10 primeiras
   por ID), um erro por consulta (inserção, remoção, substituição ou
   transposição), corrigidas com a configuração escolhida no dev.

Para cada caso são medidos a recuperação (fórmula do enunciado, NDCG@10) e o
acerto exato (o termo corrigido é igual ao termo original).

Saídas: outputs/typo_dev_selection.csv, outputs/typo_experiment.csv,
        outputs/typo_summary.json
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_all, get_query_ids_from_qrels
from src.evaluation import evaluate_query
from src.inverted_index import InvertedIndex
from src.lexical_search import BM25
from src.preprocessing import ensure_nltk_data, preprocess
from src.spell_correction import (
    SpellCorrector, damerau_levenshtein, introduce_typo, levenshtein,
)

SURFACE_CFG = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_digits": False,
    "remove_stopwords": False,
    "stem": False,
}

TYPO_TYPES = ["insertion", "deletion", "substitution", "transposition",
              "insertion", "deletion", "substitution", "transposition",
              "insertion", "deletion"]

# Primeiro valor = padrão do config (0,2). Em empate, vence o primeiro da grade.
JACCARD_GRID = [0.2, 0.1, 0.3, 0.4]


def surface_tokens(text: str) -> list:
    return preprocess(text, cfg=SURFACE_CFG)


def pick_typo_token(tokens, vocabulary, frequencies, stopwords):
    """Token para receber o erro: termo longo, de baixa DF, não stopword."""
    cands = [(frequencies.get(t, 0), i) for i, t in enumerate(tokens)
             if len(t) >= 6 and t in vocabulary and t not in stopwords]
    if cands:
        return sorted(cands)[0][1]
    for i, t in enumerate(tokens):
        if len(t) >= 4 and t in vocabulary and t not in stopwords:
            return i
    for i, t in enumerate(tokens):
        if len(t) >= 3 and t in vocabulary:
            return i
    return None


def make_unknown_typo(token, typo_type, vocabulary, rng):
    """Erro do tipo pedido que produz um termo inexistente no vocabulário."""
    for _ in range(20):
        typo = introduce_typo(token, typo_type, rng)
        if typo != token and typo not in vocabulary:
            return typo
    # Fallback: substitui o último caractere. Não garante o tipo pedido; a
    # coluna tipo_erro registra o tipo solicitado, não o efetivamente gerado.
    for ch in "xqz":
        typo = token[:-1] + ch
        if typo not in vocabulary:
            return typo
    return token + "x"


def ndcg_for(text, bm25, qrels_q):
    ranked = bm25.search(preprocess(text))
    return evaluate_query([d for d, _ in ranked], qrels_q, 10)["NDCG"]


def recovery(orig, err, corr):
    denom = orig - err
    if denom <= 0:
        return None
    return (corr - err) / denom


def run_condition(query_ids, queries, qrels, bm25, vocab, freq, stopwords,
                  distance, dict_terms, jaccard, seed, limit=10):
    """
    Executa o experimento para um conjunto de consultas e uma configuração.

    `vocab` é o vocabulário completo da coleção (garante que o erro gere termo
    inexistente). `dict_terms` é o dicionário do corretor, que depende da
    variante. Os erros não dependem da configuração (o corretor não usa o RNG),
    então são idênticos em todas as variantes.
    """
    rng = random.Random(seed)
    corrector = SpellCorrector(
        ngram_size=config.SPELL["kgram_size"],
        jaccard_threshold=jaccard,
        max_candidates=config.SPELL["max_candidates"],
        max_edit_distance=config.SPELL["max_edit_distance"],
        distance=distance,
    )
    corrector.build_dictionary(dict_terms, frequencies=freq)

    ndcg_orig = {q: ndcg_for(queries[q], bm25, qrels[q]) for q in query_ids}
    selected = sorted(q for q in query_ids if ndcg_orig[q] > 0)
    if limit is not None:
        selected = selected[:limit]

    rows = []
    for idx, qid in enumerate(selected):
        typo_type = TYPO_TYPES[idx % len(TYPO_TYPES)]
        q_text = queries[qid]
        tokens = surface_tokens(q_text)
        ti = pick_typo_token(tokens, vocab, freq, stopwords)
        if ti is None:
            continue
        original_term = tokens[ti]
        typo_term = make_unknown_typo(original_term, typo_type, vocab, rng)
        typo_tokens = tokens[:]
        typo_tokens[ti] = typo_term
        corrected_tokens = typo_tokens[:]
        corr = corrector.correct(typo_term)
        if corr is not None:
            corrected_tokens[ti] = corr

        q_err = " ".join(typo_tokens)
        q_cor = " ".join(corrected_tokens)
        n_o = ndcg_orig[qid]
        n_e = ndcg_for(q_err, bm25, qrels[qid])
        n_c = ndcg_for(q_cor, bm25, qrels[qid])
        rec = recovery(n_o, n_e, n_c)
        rows.append({
            "query_id": qid,
            "query_original": q_text,
            "query_erro": q_err,
            "query_corrigida": q_cor,
            "tipo_erro": typo_type,
            "termo_alterado": original_term,
            "termo_com_erro": typo_term,
            "termo_corrigido": corr if corr is not None else "",
            "correcao_exata": int(corr == original_term),
            "NDCG10_original": n_o,
            "NDCG10_erro": n_e,
            "NDCG10_corrigida": n_c,
            "recuperacao": rec,
            "observacao": "" if rec is not None else "erro não reduziu o desempenho",
        })
    return rows


def summarize(rows):
    """Médias de uma configuração. A recuperação média usa só os casos válidos."""
    vals = [r["recuperacao"] for r in rows if r["recuperacao"] is not None]
    exact = [r["correcao_exata"] for r in rows]
    return {
        "consultas": len(rows),
        "casos_validos_recuperacao": len(vals),
        "recuperacao_media": sum(vals) / len(vals) if vals else 0.0,
        "taxa_correcao_exata": sum(exact) / len(exact) if exact else 0.0,
    }


def main() -> None:
    ensure_nltk_data()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    corpus, queries, qrels_dev, qrels_test = load_all(
        config.CORPUS_FILE, config.QUERIES_FILE,
        config.QRELS_DEV_FILE, config.QRELS_TEST_FILE,
    )
    dev_ids = get_query_ids_from_qrels(qrels_dev)
    test_ids = get_query_ids_from_qrels(qrels_test)

    index = InvertedIndex.load(config.INDEX_DIR / "inverted_index.pkl")
    bm25 = BM25(index)

    with open(config.INDEX_DIR / "spell_vocabulary.json", encoding="utf-8") as f:
        freq_all = json.load(f)
    from nltk.corpus import stopwords as sw
    stop = set(sw.words("english"))
    freq_nostop = {t: f for t, f in freq_all.items() if t not in stop}
    vocab_all = set(freq_all)

    # Ordem da grade: a configuração padrão vem primeiro (desempate).
    variants = {
        "Damerau + dicionário completo": (damerau_levenshtein, freq_all),
        "Damerau + sem stopwords": (damerau_levenshtein, freq_nostop),
        "Levenshtein + dicionário completo": (levenshtein, freq_all),
        "Levenshtein + sem stopwords": (levenshtein, freq_nostop),
    }

    # 1) Seleção no DEV: variante x limiar de Jaccard.
    sel_rows, scores = [], {}
    for name, (dist, freq) in variants.items():
        for jac in JACCARD_GRID:
            rows = run_condition(dev_ids, queries, qrels_dev, bm25, vocab_all,
                                 freq, stop, dist, set(freq), jac, seed=42, limit=None)
            s = summarize(rows)
            scores[(name, jac)] = s["recuperacao_media"]
            sel_rows.append({"variante": name, "jaccard": jac, **s,
                             "recuperacao_media": round(s["recuperacao_media"], 4),
                             "taxa_correcao_exata": round(s["taxa_correcao_exata"], 4)})
            print(f"[dev] {name} | Jaccard={jac}: recuperação média = "
                  f"{s['recuperacao_media']:.4f}, acerto exato = {s['taxa_correcao_exata']:.2f}")

    pd.DataFrame(sel_rows).to_csv(config.OUTPUT_DIR / "typo_dev_selection.csv", index=False)
    best_name, best_jac = max(scores, key=scores.get)  # max devolve o primeiro empatado
    print(f"[dev] configuração escolhida: {best_name}, Jaccard={best_jac}")

    # 2) Aplicação no TEST
    dist, freq = variants[best_name]
    rows = run_condition(test_ids, queries, qrels_test, bm25, vocab_all,
                         freq, stop, dist, set(freq), best_jac, seed=42, limit=10)
    pd.DataFrame(rows).to_csv(config.OUTPUT_DIR / "typo_experiment.csv", index=False)

    summary = {
        "dev_configuracao_escolhida": {"variante": best_name, "jaccard": best_jac},
        "test": summarize(rows),
        "test_consultas_com_erro": [r["query_id"] for r in rows],
    }
    with open(config.OUTPUT_DIR / "typo_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"[test] {summary['test']}")
    print("-> outputs/typo_experiment.csv, typo_dev_selection.csv, typo_summary.json")


if __name__ == "__main__":
    main()
