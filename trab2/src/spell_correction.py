"""
spell_correction.py
===================
Correção ortográfica de termos desconhecidos em três passos:
  1. Geração de candidatos : k-gramas de caracteres (índice invertido de
                             k-gramas -> termos do vocabulário)
  2. Filtragem             : coeficiente de Jaccard entre os conjuntos de
                             k-gramas do termo e do candidato
  3. Seleção               : distância de edição, com desempate pelo termo de
                             maior frequência documental (DF)

O dicionário é formado pelos termos da coleção antes do stemming. Um termo é
"desconhecido" quando não está no dicionário; só esses termos são corrigidos.

A função `introduce_typo` gera erros de inserção, remoção, substituição e
troca de ordem de caracteres, usados no experimento de tolerância a erros.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Set


def char_ngrams(word: str, n: int = 3) -> Set[str]:
    """
    Devolve o conjunto de k-gramas de caracteres de `word`, com padding nas
    bordas (caracteres especiais de início/fim) para capturar prefixos e
    sufixos. Ex.: "teste" com n=3 -> {"$$t", "$te", "tes", "est", "ste", "te$", "e$$"}.
    """
    padded = "$" * (n - 1) + word.lower() + "$" * (n - 1)
    if len(padded) < n:
        return {padded}
    return {padded[i : i + n] for i in range(len(padded) - n + 1)}


def jaccard(a: Set[str], b: Set[str]) -> float:
    """Coeficiente de Jaccard entre dois conjuntos."""
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def levenshtein(a: str, b: str) -> int:
    """
    Distância de edição de Levenshtein (inserção, remoção e substituição, custo 1).
    Transposição de caracteres custa 2 (duas substituições).
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    if len(a) > len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        curr = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def damerau_levenshtein(a: str, b: str) -> int:
    """
    Distância de edição com suporte a TRANSPOSIÇÃO de caracteres adjacentes
    (variante Optimal String Alignment — OSA). Diferente do Levenshtein puro,
    "cemo" -> "ecmo" custa 1 (troca de "c" e "e"), e não 2.

    """
    a, b = a.lower(), b.lower()
    if a == b:
        return 0
    len_a, len_b = len(a), len(b)
    if len_a == 0:
        return len_b
    if len_b == 0:
        return len_a

    INF = len_a + len_b
    # Matriz com uma linha/coluna extras (convenção do algoritmo OSA).
    h = [[0] * (len_b + 2) for _ in range(len_a + 2)]
    h[0][0] = INF
    for i in range(len_a + 1):
        h[i + 1][1] = i
        h[i + 1][0] = INF
    for j in range(len_b + 1):
        h[1][j + 1] = j
        h[0][j + 1] = INF

    da: Dict[str, int] = {}
    for i in range(1, len_a + 1):
        db = 0
        for j in range(1, len_b + 1):
            i1 = da.get(b[j - 1], 0)
            j1 = db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j
            h[i + 1][j + 1] = min(
                h[i][j] + cost,                      # substituição
                h[i + 1][j] + 1,                     # inserção
                h[i][j + 1] + 1,                     # remoção
                h[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1),  # transposição
            )
        da[a[i - 1]] = i
    return h[len_a + 1][len_b + 1]


class SpellCorrector:
    """
    Corretor ortográfico baseado em k-gramas + Jaccard + distância de edição.
    """

    def __init__(self, ngram_size: int = 3, jaccard_threshold: float = 0.2,
                 max_candidates: int = 50, max_edit_distance: int = 3,
                 distance=None) -> None:
        self.distance = distance or damerau_levenshtein
        self.n = ngram_size
        self.jaccard_threshold = jaccard_threshold
        self.max_candidates = max_candidates
        self.max_edit_distance = max_edit_distance

        self.vocabulary: Set[str] = set()
        # Frequência (document frequency) de cada termo, para desempate.
        self.frequencies: Dict[str, int] = {}
        # Índice invertido de k-gramas: k-grama -> conjunto de termos.
        self.ngram_index: Dict[str, Set[str]] = {}
        # Cache termo -> correção (para não recalcular).
        self._cache: Dict[str, Optional[str]] = {}

    def build_dictionary(self, terms: Set[str], frequencies: Dict[str, int] = None) -> None:
        """
        Constrói o dicionário (vocabulário) e o índice de k-gramas a partir de
        um conjunto de termos, que devem estar SEM stemming.

        `frequencies` (opcional) mapeia termo -> document frequency (DF) e é
        usado para desempatar candidatos com a mesma distância de edição,
        preferindo o termo mais frequente da coleção.
        """
        self.vocabulary = set(terms)
        self.frequencies = dict(frequencies) if frequencies else {}
        self.ngram_index.clear()
        self._cache.clear()

        for term in self.vocabulary:
            for gram in char_ngrams(term, self.n):
                self.ngram_index.setdefault(gram, set()).add(term)

    def _candidates(self, term: str) -> List[str]:
        """
        Gera candidatos: termos do vocabulário que compartilham ao menos um
        k-grama com o termo, filtrados pelo coeficiente de Jaccard.

        Se restarem mais candidatos que `max_candidates` após o filtro de
        Jaccard, mantêm-se os de maior Jaccard.
        """
        grams = char_ngrams(term, self.n)
        if not grams:
            return []

        # Reúne todos os termos que compartilham ao menos um k-grama.
        candidate_pool: Set[str] = set()
        for gram in grams:
            candidate_pool |= self.ngram_index.get(gram, set())

        # Filtra pelo coeficiente de Jaccard.
        filtered: List[str] = []
        for cand in candidate_pool:
            if cand == term:
                continue
            if jaccard(grams, char_ngrams(cand, self.n)) >= self.jaccard_threshold:
                filtered.append(cand)

        # Se houver muitos, mantém os de maior Jaccard (limite determinístico).
        if len(filtered) > self.max_candidates:
            filtered.sort(
                key=lambda c: jaccard(grams, char_ngrams(c, self.n)), reverse=True
            )
            filtered = filtered[: self.max_candidates]

        return filtered

    def correct(self, term: str) -> Optional[str]:
        """
        Devolve a correção sugerida para `term`, ou None se nenhum candidato
        estiver dentro da distância máxima aceitável.

        Usa distância de edição com suporte a transposição (Damerau-Levenshtein)
        e, em caso de empate, prefere o termo mais frequente da coleção (DF).
        """
        # Palavras conhecidas não precisam de correção.
        if term in self.vocabulary:
            return term

        if term in self._cache:
            return self._cache[term]

        candidates = self._candidates(term)
        best: Optional[str] = None
        best_dist = self.max_edit_distance + 1
        for cand in candidates:
            d = self.distance(term, cand)
            if d < best_dist:
                best_dist = d
                best = cand
            elif d == best_dist and best is not None:
                # Empate: termo mais frequente (DF); se empatar de novo, o
                # lexicograficamente menor (determinístico).
                f_cand = self.frequencies.get(cand, 0)
                f_best = self.frequencies.get(best, 0)
                if f_cand > f_best or (f_cand == f_best and cand < best):
                    best = cand

        # Se a menor distância for maior que o permitido, descarta.
        if best_dist > self.max_edit_distance:
            best = None

        self._cache[term] = best
        return best


# Geração de erros ortográficos (para o experimento de tolerância a erros)
def introduce_typo(word: str, typo_type: str, rng: random.Random) -> str:
    """
    Insere um erro ortográfico em `word` segundo `typo_type`:

      * "insertion"    : duplica/injeta um caractere
      * "deletion"     : remove um caractere
      * "substitution" : troca um caractere por outro
      * "transposition": troca a ordem de dois caracteres adjacentes

    Retorna a palavra com erro. A função garante que o resultado seja
    DIFERENTE da palavra original.
    """
    if len(word) < 2:
        # Palavras muito curtas: apenas substitui ou insere.
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        if typo_type == "deletion":
            return word + "x"
        pos = rng.randrange(len(word))
        ch = rng.choice(alphabet)
        return word[:pos] + ch + word[pos:]

    pos = rng.randrange(len(word))

    if typo_type == "insertion":
        ch = word[pos]  # duplica o caractere na posição sorteada
        typo = word[:pos] + ch + word[pos:]
    elif typo_type == "deletion":
        typo = word[:pos] + word[pos + 1 :]
    elif typo_type == "transposition":
        i = rng.randrange(len(word) - 1)
        typo = word[:i] + word[i + 1] + word[i] + word[i + 2 :]
    else:  # substitution
        alphabet = "abcdefghijklmnopqrstuvwxyz"
        ch = rng.choice(alphabet.replace(word[pos], ""))
        typo = word[:pos] + ch + word[pos + 1 :]

    # Garante que o erro produziu uma palavra diferente.
    if typo == word:
        return introduce_typo(word, typo_type, rng)
    return typo
