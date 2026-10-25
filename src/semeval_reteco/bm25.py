import re

import bm25s
import numpy as np
import Stemmer
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity

from semeval_reteco.data import load_corpus
from semeval_reteco.queries import build_query

# Same pipeline as the official RETECO baseline (Lucene analyzer + gensim LuceneBM25Model).
# Stemmer, one-letter tokens and "3.5mm" handling each moved the score noticeably,
# so don't change them without re-checking against the official numbers.
K1, B = 0.9, 0.4
TOKEN_PATTERN = r"(?u)\b\w+(?:['’.]\w+)*\b"
STEMMER = Stemmer.Stemmer("porter")


def analyze(texts):
    """Tokenize, lowercase, drop English stopwords, stem. Used for both documents and queries."""
    texts = [re.sub(r"['’]s\b", "", t) for t in texts]  # possessive 's, like Lucene
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, token_pattern=TOKEN_PATTERN,
        return_ids=False, show_progress=False)


class Bm25:
    def __init__(self, domain):
        self.doc_ids, texts = load_corpus(domain)
        # Tie-break for equal scores (mostly zeros). Corpus order and doc id both leak relevance
        # (gold docs are named ..._doc_N and come first), so use a fixed random order instead.
        self.tie_order = np.random.default_rng(0).permutation(len(self.doc_ids))
        doc_tokens = analyze(texts)
        self.dictionary = Dictionary(doc_tokens)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=K1, b=B)
        self.index = SparseMatrixSimilarity(
            self.model[[self.dictionary.doc2bow(d) for d in doc_tokens]],
            num_docs=len(doc_tokens), num_terms=len(self.dictionary),
            normalize_queries=False, normalize_documents=False)

    def search(self, queries, k=100):
        """Same return shape as embeddings.search: (doc ids per query, scores per query)."""
        result_ids, result_scores = [], []
        for tokens in analyze(queries):
            sims = np.asarray(self.index[self.model[self.dictionary.doc2bow(tokens)]])
            top = np.lexsort((self.tie_order, -sims))[:k]
            result_ids.append([self.doc_ids[i] for i in top])
            result_scores.append(sims[top])
        return result_ids, result_scores


def run_bm25(bm25, turns, variant, recent_first=False, k=100):
    """-> {qid: {doc_id: score}}, in rank order."""
    queries = [build_query(t, variant, recent_first) for t in turns]
    result_ids, result_scores = bm25.search(queries, k)
    return {t["qid"]: dict(zip(ids, map(float, scores)))
        for t, ids, scores in zip(turns, result_ids, result_scores)}
