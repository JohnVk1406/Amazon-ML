import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import combined_text


def make_texts(df: pd.DataFrame):
    return [combined_text(name, addr) for name, addr in zip(
        df["business_name"].fillna("").astype(str),
        df["business_address"].fillna("").astype(str),
    )]


def fit_vectorizer(sample_texts, max_features=40000):
    vec = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(3, 5),
        max_features=max_features,
        min_df=2,
        sublinear_tf=True,
        norm="l2",
    )
    vec.fit(sample_texts)
    return vec


def topk_sparse_row(row, k):
    if row.nnz == 0:
        return [], []
    idx = row.indices
    dat = row.data
    if len(dat) <= k:
        order = np.argsort(-dat)
        return idx[order].tolist(), dat[order].tolist()
    part = np.argpartition(-dat, k - 1)[:k]
    order = part[np.argsort(-dat[part])]
    return idx[order].tolist(), dat[order].tolist()


def retrieve_topk(queries_texts, queries_ids, corpus_texts, corpus_ids, vec, k=30, q_batch=5000):
    import scipy.sparse as sp

    C = vec.transform(corpus_texts)
    out = {qid: {} for qid in queries_ids}
    n = len(queries_texts)
    for s in range(0, n, q_batch):
        e = min(n, s + q_batch)
        Q = vec.transform(queries_texts[s:e])
        S = Q @ C.T
        S = S.tocsr()
        for i in range(e - s):
            qid = queries_ids[s + i]
            r = S.getrow(i)
            idxs, scores = topk_sparse_row(r, k)
            for ci, sc in zip(idxs, scores):
                cid = corpus_ids[ci]
                prev = out[qid].get(cid, 0)
                if sc > prev:
                    out[qid][cid] = float(sc)
    res = {}
    for qid, d in out.items():
        if not d:
            res[qid] = []
        else:
            items = sorted(d.items(), key=lambda x: -x[1])[:k]
            res[qid] = [cid for cid, _ in items]
    return res


def retrieve_sharded(queries_texts, queries_ids, corpus_texts, corpus_ids, vec, k=30, q_batch=5000, c_shard=200000):
    best = {qid: {} for qid in queries_ids}
    n_c = len(corpus_texts)
    for cs in range(0, n_c, c_shard):
        ce = min(n_c, cs + c_shard)
        shard_texts = corpus_texts[cs:ce]
        shard_ids = corpus_ids[cs:ce]
        C = vec.transform(shard_texts)
        n_q = len(queries_texts)
        for qs in range(0, n_q, q_batch):
            qe = min(n_q, qs + q_batch)
            Q = vec.transform(queries_texts[qs:qe])
            S = (Q @ C.T).tocsr()
            for i in range(qe - qs):
                qid = queries_ids[qs + i]
                r = S.getrow(i)
                if r.nnz == 0:
                    continue
                idxs, scores = topk_sparse_row(r, k)
                d = best[qid]
                for ci, sc in zip(idxs, scores):
                    cid = shard_ids[ci]
                    if cid not in d or sc > d[cid]:
                        d[cid] = float(sc)
                if len(d) > k * 2:
                    top = sorted(d.items(), key=lambda x: -x[1])[:k]
                    best[qid] = dict(top)
    res = {}
    for qid, d in best.items():
        items = sorted(d.items(), key=lambda x: -x[1])[:k]
        res[qid] = [cid for cid, _ in items]
    return res
