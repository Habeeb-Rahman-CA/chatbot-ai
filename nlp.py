"""
Core Natural Language Processing (NLP) primitives built from scratch.

Covers:
1. Text Normalization & Tokenization
2. Stopword Filtering
3. Stemming (Morphological reduction)
4. Vectorization (Term Frequency - Inverse Document Frequency / TF-IDF)
5. Similarity Metrics (Cosine Similarity, Levenshtein Distance)
"""

from __future__ import annotations
import math
import re
from collections import Counter


# Common English stopwords (grammatical filler words with low conversational specificity)
# Note: In conversational NLP, we intentionally keep pronouns like 'i', 'you', 'my', 'your'
# so the system can distinguish between 'what is my name' and 'what is your name'.
DEFAULT_STOPWORDS: set[str] = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "had", "have", "in", "is", "it", "its", "of", "on", "that",
    "the", "to", "was", "were", "will", "with", "would", "this", "there"
}


def tokenize(text: str) -> list[str]:
    """
    Split text into words/tokens, strip punctuation, and convert to lowercase.

    Example:
        >>> tokenize("Hello, world! How's it going?")
        ['hello', 'world', 'how', 's', 'it', 'going']
    """
    cleaned = text.lower()
    # Extract alphanumeric word tokens (punctuation is stripped)
    tokens = re.findall(r"\b\w+\b", cleaned)
    return tokens


def remove_stopwords(tokens: list[str], stopwords: set[str] | None = None) -> list[str]:
    """
    Filter out common words that don't carry specific intent meaning.
    """
    stop_set = stopwords if stopwords is not None else DEFAULT_STOPWORDS
    return [t for t in tokens if t not in stop_set]


def simple_stem(word: str) -> str:
    """
    Rule-based suffix stemmer (reduces words to their root forms).

    Example:
        >>> simple_stem("running")
        'run'
        >>> simple_stem("cats")
        'cat'
    """
    word = word.lower()
    if len(word) <= 3:
        return word

    # Common English suffixes
    suffixes = [
        ("sses", "ss"),
        ("ies", "i"),
        ("ing", ""),
        ("edly", ""),
        ("ed", ""),
        ("ly", ""),
        ("es", ""),
        ("s", ""),
    ]

    for suffix, replacement in suffixes:
        if word.endswith(suffix):
            stemmed = word[:-len(suffix)] + replacement
            # Keep minimum root length of 2
            if len(stemmed) >= 2:
                # Handle double consonants before -ing (e.g., 'running' -> 'runn' -> 'run')
                if len(stemmed) >= 3 and stemmed[-1] == stemmed[-2] and stemmed[-1] not in "lsz":
                    stemmed = stemmed[:-1]
                return stemmed
            break

    return word


def preprocess(text: str, filter_stops: bool = True, stem_words: bool = True) -> list[str]:
    """
    Full text preprocessing pipeline:
    1. Tokenize & lowercase
    2. Optional stopword removal
    3. Optional stemming
    """
    tokens = tokenize(text)
    if filter_stops:
        tokens = remove_stopwords(tokens)
    if stem_words:
        tokens = [simple_stem(t) for t in tokens]
    return tokens


def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Compute minimum edit distance (insertions, deletions, substitutions)
    between two strings. Teaches how typo correction works.
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row

    return previous_row[-1]


def string_similarity(s1: str, s2: str) -> float:
    """
    Return a normalized similarity score in range [0.0, 1.0] using Levenshtein distance.
    1.0 means exact match, 0.0 means completely different.
    """
    s1, s2 = s1.lower().strip(), s2.lower().strip()
    if s1 == s2:
        return 1.0
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 1.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))


class SimpleTfidfVectorizer:
    """
    Lightweight TF-IDF (Term Frequency - Inverse Document Frequency) Vectorizer.

    Converts raw text documents into numeric vectors representing word significance.
    - TF(t, d): Frequency of term `t` in document `d`.
    - IDF(t, D): Log(Total documents / Documents containing `t`).
    - TF-IDF = TF * IDF.
    """

    def __init__(self, filter_stops: bool = True, stem_words: bool = True):
        self.filter_stops = filter_stops
        self.stem_words = stem_words
        self.vocabulary: list[str] = []
        self.word_to_idx: dict[str, int] = {}
        self.idf: dict[str, float] = {}

    def fit(self, documents: list[str]) -> SimpleTfidfVectorizer:
        """
        Learn vocabulary and IDF weights across the training document collection.
        """
        doc_tokens = [
            preprocess(doc, self.filter_stops, self.stem_words)
            for doc in documents
        ]
        num_docs = len(doc_tokens)
        if num_docs == 0:
            return self

        # Document frequency (DF): how many docs contain word
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens:
            unique_words = set(tokens)
            for word in unique_words:
                doc_freq[word] += 1

        # Sorted vocabulary for deterministic indexing
        self.vocabulary = sorted(list(doc_freq.keys()))
        self.word_to_idx = {w: i for i, w in enumerate(self.vocabulary)}

        # Smoothed IDF: log((1 + N) / (1 + df)) + 1
        self.idf = {
            w: math.log((1 + num_docs) / (1 + doc_freq[w])) + 1.0
            for w in self.vocabulary
        }
        return self

    def transform(self, text: str) -> list[float]:
        """
        Convert text string into a normalized TF-IDF vector.
        """
        tokens = preprocess(text, self.filter_stops, self.stem_words)
        if not self.vocabulary:
            return []

        counts = Counter(tokens)
        total_tokens = len(tokens) or 1

        vector = [0.0] * len(self.vocabulary)
        for word, count in counts.items():
            if word in self.word_to_idx:
                idx = self.word_to_idx[word]
                tf = count / total_tokens
                idf = self.idf[word]
                vector[idx] = tf * idf

        # L2 normalize vector
        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]

        return vector


def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Calculate the cosine of the angle between two vectors.
    Measures directional similarity regardless of vector magnitude.

    Formula: (A · B) / (||A|| * ||B||)
    """
    if len(vec_a) != len(vec_b) or len(vec_a) == 0:
        return 0.0

    dot_product = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))

    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return max(0.0, min(1.0, dot_product / (norm_a * norm_b)))