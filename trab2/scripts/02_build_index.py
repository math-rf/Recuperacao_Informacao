"""
02_build_index.py
=================
Pré-processa o corpus, constrói o índice invertido (do zero) e o vocabulário
para a correção ortográfica.

    python scripts/02_build_index.py

Gera:
    index/inverted_index.pkl     -> índice invertido serializado
    index/spell_vocabulary.json  -> {termo: DF} de todos os termos da coleção,
                                    antes do stemming (stopwords incluídas; a
                                    exclusão é decidida no dev pelo 04)
    outputs/index_stats.json     -> estatísticas do índice
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config
from src.data_loader import load_corpus
from src.inverted_index import InvertedIndex
from src.preprocessing import ensure_nltk_data, preprocess

SURFACE_CFG = {
    "lowercase": True,
    "remove_punctuation": True,
    "remove_digits": False,
    "remove_stopwords": False,
    "stem": False,
}


def build_spell_vocabulary(corpus: dict) -> dict:
    """{termo: document frequency} sobre os termos antes do stemming."""
    df: dict = {}
    for text in corpus.values():
        for term in set(preprocess(text, cfg=SURFACE_CFG)):
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

    print("[3/3] Construindo vocabulário para a correção ortográfica ...")
    vocab = build_spell_vocabulary(corpus)
    with open(config.INDEX_DIR / "spell_vocabulary.json", "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False)

    stats = index.stats()
    print("\n--- Estatísticas do índice invertido ---")
    for k, v in stats.items():
        print(f"  {k:18s}: {v}")

    with open(config.OUTPUT_DIR / "index_stats.json", "w", encoding="utf-8") as f:
        json.dump({**stats, "spell_vocabulary_size": len(vocab)}, f, indent=2)
    print(f"\nEstatísticas salvas em {config.OUTPUT_DIR / 'index_stats.json'}")


if __name__ == "__main__":
    main()
