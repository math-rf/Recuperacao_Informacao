"""
02_build_index.py
=================
Pré-processa o corpus, constrói o índice invertido (do zero) e o vocabulário
para a correção ortográfica.

Passo 2 do pipeline:

    python scripts/02_build_index.py

Arquivos gerados:
    index/inverted_index.pkl   -> índice invertido serializado (pickle)
    index/spell_vocabulary.json-> termos da coleção ANTES do stemming
    outputs/index_stats.json   -> estatísticas do índice (p/ relatório)
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from src.data_loader import load_corpus  # noqa: E402
from src.inverted_index import InvertedIndex  # noqa: E402
from src.preprocessing import ensure_nltk_data, preprocess  # noqa: E402


def build_spell_vocabulary(corpus: dict) -> dict:
    """
    Constrói o vocabulário para a correção ortográfica usando os termos da
    coleção ANTES do stemming/lematização (exigência (i) do enunciado).

    Como na busca as stopwords são removidas ANTES do stemming, o dicionário
    também as exclui: assim, termos de conteúdo não competem com stopwords de
    altíssima frequência ("of", "for", "or") no desempate por DF do corretor.

    Retorna {termo: document_frequency}, para permitir desempate por
    frequência no corretor ortográfico.
    """
    surface_cfg = {
        "lowercase": True,
        "remove_punctuation": True,
        "remove_digits": False,
        "remove_stopwords": True,   # exclui stopwords (sem stemming)
        "stem": False,
        "lemmatize": False,
    }
    df: dict = {}
    for text in corpus.values():
        # Termos únicos por documento (DF conta documentos, não ocorrências).
        for term in set(preprocess(text, cfg=surface_cfg)):
            df[term] = df.get(term, 0) + 1
    return df


def main() -> None:
    ensure_nltk_data()
    config.INDEX_DIR.mkdir(parents=True, exist_ok=True)
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/3] Carregando corpus ...")
    corpus = load_corpus(config.CORPUS_FILE)
    print(f"      {len(corpus)} documentos carregados.")

    print("[2/3] Pré-processando e construindo o índice invertido ...")
    tokenized = [(doc_id, preprocess(text)) for doc_id, text in corpus.items()]
    index = InvertedIndex().build(tokenized)
    index.save(config.INDEX_DIR / "inverted_index.pkl")

    print("[3/3] Construindo vocabulário p/ correção ortográfica ...")
    vocab = build_spell_vocabulary(corpus)
    with open(config.INDEX_DIR / "spell_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    # Estatísticas exigidas pelo enunciado (requisito 2).
    stats = index.stats()
    print("\n--- Estatísticas do índice invertido ---")
    for k, v in stats.items():
        print(f"  {k:18s}: {v}")

    with open(config.OUTPUT_DIR / "index_stats.json", "w", encoding="utf-8") as f:
        json.dump({**stats, "spell_vocabulary_size": len(vocab)}, f, indent=2)
    print(f"\nEstatísticas salvas em {config.OUTPUT_DIR / 'index_stats.json'}")


if __name__ == "__main__":
    main()
