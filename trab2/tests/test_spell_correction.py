"""
Testes do corretor ortográfico (k-gramas, Jaccard, distância de edição, erros).

Rodar a partir da raiz do projeto:
    python -X utf8 -m unittest discover -s tests -v
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.spell_correction import (  # noqa: E402
    SpellCorrector, char_ngrams, damerau_levenshtein, introduce_typo, jaccard,
    levenshtein,
)

# Vocabulário de brinquedo com DF (frequência documental) para os testes.
TOY_VOCAB = {"food": 300, "fenugreek": 20, "chemo": 40, "ecmo": 5, "medicine": 900}


def make_corrector(jaccard_threshold=0.2):
    c = SpellCorrector(ngram_size=3, jaccard_threshold=jaccard_threshold,
                       max_candidates=50, max_edit_distance=3,
                       distance=damerau_levenshtein)
    c.build_dictionary(set(TOY_VOCAB), frequencies=TOY_VOCAB)
    return c


class NGramTests(unittest.TestCase):
    def test_char_ngrams_with_padding(self):
        self.assertEqual(
            char_ngrams("teste", 3),
            {"$$t", "$te", "tes", "est", "ste", "te$", "e$$"},
        )

    def test_jaccard_identical_and_disjoint(self):
        a = {"abc", "bcd"}
        self.assertEqual(jaccard(a, a), 1.0)
        self.assertEqual(jaccard({"abc"}, {"xyz"}), 0.0)

    def test_jaccard_partial_overlap(self):
        self.assertAlmostEqual(jaccard({"a", "b"}, {"b", "c"}), 1 / 3)


class DistanceTests(unittest.TestCase):
    def test_levenshtein_substitution_costs_one(self):
        self.assertEqual(levenshtein("kitten", "sitting"), 3)

    def test_levenshtein_transposition_costs_two(self):
        self.assertEqual(levenshtein("cemo", "ecmo"), 2)

    def test_osa_transposition_costs_one(self):
        self.assertEqual(damerau_levenshtein("cemo", "ecmo"), 1)
        self.assertEqual(damerau_levenshtein("mediicne", "medicine"), 1)

    def test_osa_insertion_and_deletion(self):
        self.assertEqual(damerau_levenshtein("food", "fod"), 1)
        self.assertEqual(damerau_levenshtein("food", "foood"), 1)

    def test_distance_is_symmetric(self):
        pairs = [("food", "fod"), ("ecmo", "cemo"), ("medicine", "mediicne")]
        for a, b in pairs:
            self.assertEqual(damerau_levenshtein(a, b), damerau_levenshtein(b, a))


class CorrectorTests(unittest.TestCase):
    def test_known_word_is_returned_unchanged(self):
        self.assertEqual(make_corrector().correct("food"), "food")

    def test_corrects_deletion_insertion_transposition(self):
        c = make_corrector()
        self.assertEqual(c.correct("fod"), "food")
        self.assertEqual(c.correct("enugreek"), "fenugreek")
        self.assertEqual(c.correct("mediicne"), "medicine")

    def test_returns_none_for_unrelated_term(self):
        self.assertIsNone(make_corrector().correct("xyzzyq"))

    def test_jaccard_threshold_filters_candidates(self):
        # "fod" x "food": Jaccard = 4/7 ~ 0,57. Com limiar 0,9 não há candidato.
        self.assertIsNone(make_corrector(jaccard_threshold=0.9).correct("fod"))
        self.assertEqual(make_corrector(jaccard_threshold=0.2).correct("fod"), "food")

    def test_tie_is_broken_by_document_frequency(self):
        # "ecmo" e "chemo" estão ambos à distância 1 de "cemo". Com empate,
        # vence o termo de maior DF ("chemo", DF 40 > 5). Comportamento documentado:
        # é a causa do erro no caso ecmo -> cemo -> chemo.
        self.assertEqual(make_corrector().correct("cemo"), "chemo")


class TypoGenerationTests(unittest.TestCase):
    def test_typo_is_new_word_with_expected_length_change(self):
        expected = {"insertion": 1, "deletion": -1,
                    "substitution": 0, "transposition": 0}
        rng = random.Random(0)
        for typo_type, delta in expected.items():
            for _ in range(50):
                typo = introduce_typo("medicine", typo_type, rng)
                self.assertNotEqual(typo, "medicine")
                self.assertEqual(len(typo) - len("medicine"), delta, typo_type)


if __name__ == "__main__":
    unittest.main()
