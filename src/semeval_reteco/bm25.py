import re

import bm25s
import numpy as np
import Stemmer
from gensim.corpora import Dictionary
from gensim.models import LuceneBM25Model
from gensim.similarities import SparseMatrixSimilarity

from semeval_reteco.data import load_corpus
from semeval_reteco.queries import build_query

# Same settings as the official baseline. If you change any of them, check the scores against the official ones again.
K1, B = 0.9, 0.4
TOKEN_PATTERN = r"(?u)\b\w+(?:['’.]\w+)*\b"
STEMMER = Stemmer.Stemmer("porter")


def analyze(texts):
    texts = [re.sub(r"['’]s\b", "", t) for t in texts]
    return bm25s.tokenize(texts, stopwords="en", stemmer=STEMMER, token_pattern=TOKEN_PATTERN,
        return_ids=False, show_progress=False)


def analyze_corpus(domain):
    doc_ids, texts = load_corpus(domain)
    return doc_ids, analyze(texts)


class Bm25:
    def __init__(self, domain, k1=K1, b=B, analyzed=None):
        self.doc_ids, doc_tokens = analyzed or analyze_corpus(domain)
        # equal scores get a fixed random order, because doc ids give away which docs are relevant
        self.tie_order = np.random.default_rng(0).permutation(len(self.doc_ids))
        self.dictionary = Dictionary(doc_tokens)
        self.model = LuceneBM25Model(dictionary=self.dictionary, k1=k1, b=b)
        self.index = SparseMatrixSimilarity(
            self.model[[self.dictionary.doc2bow(d) for d in doc_tokens]],
            num_docs=len(doc_tokens), num_terms=len(self.dictionary),
            normalize_queries=False, normalize_documents=False)

    def search(self, queries, k=100):
        result_ids, result_scores = [], []
        for tokens in analyze(queries):
            sims = np.asarray(self.index[self.model[self.dictionary.doc2bow(tokens)]])
            top = np.lexsort((self.tie_order, -sims))[:k]
            result_ids.append([self.doc_ids[i] for i in top])
            result_scores.append(sims[top])
        return result_ids, result_scores


def run_bm25(bm25, turns, variant, recent_first=False, k=100):
    queries = [build_query(t, variant, recent_first) for t in turns]
    result_ids, result_scores = bm25.search(queries, k)
    return {t["qid"]: dict(zip(ids, map(float, scores)))
        for t, ids, scores in zip(turns, result_ids, result_scores)}
