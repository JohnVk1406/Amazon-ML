import os
import time

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from .normalize import combined_text


def make_texts(df):
    return [combined_text(name, addr) for name, addr in zip(
        df['business_name'].fillna('').astype(str),
        df['business_address'].fillna('').astype(str),
    )]


def fit_vec(texts_sample, max_features=50000, max_df=0.02):
    vec = TfidfVectorizer(
        analyzer='word',
        ngram_range=(1, 1),
        max_features=max_features,
        min_df=2,
        max_df=max_df,
        sublinear_tf=True,
        norm='l2',
    )
    vec.fit(texts_sample)
    return vec


def retrieve_with_scores(qt, qi, ct, ci, vec, k=25, q_batch=5000, c_shard=400000):
    shards = []
    for s in range(0, len(ct), c_shard):
        e = min(len(ct), s + c_shard)
        Cm = vec.transform(ct[s:e])
        shards.append((Cm, ci[s:e]))

    out = {}
    for qs in range(0, len(qt), q_batch):
        qe = min(len(qt), qs + q_batch)
        Q = vec.transform(qt[qs:qe])
        batch = [dict() for _ in range(qe - qs)]
        for C, cids in shards:
            S = (Q @ C.T).tocsr()
            for i in range(qe - qs):
                r = S.getrow(i)
                if r.nnz == 0:
                    continue
                if r.nnz > 2000:
                    idx = r.indices
                    dat = r.data
                    part = np.argpartition(-dat, 2000)[:2000]
                    for col, sc in zip(idx[part], dat[part]):
                        cid = cids[col]
                        d = batch[i]
                        if cid not in d or sc > d[cid]:
                            d[cid] = float(sc)
                else:
                    for col, sc in zip(r.indices, r.data):
                        cid = cids[col]
                        d = batch[i]
                        if cid not in d or sc > d[cid]:
                            d[cid] = float(sc)
        for i, qid in enumerate(qi[qs:qe]):
            d = batch[i]
            top = sorted(d.items(), key=lambda x: -x[1])[:k]
            out[qid] = top
    return out


def run_country(s1_df, s23_df, k=25, max_features=50000):
    q_texts = make_texts(s1_df)
    q_ids = s1_df['entity_id'].tolist()
    c_texts = make_texts(s23_df)
    c_ids = s23_df['entity_id'].tolist()

    vec_p = fit_vec(c_texts[:200000] if len(c_texts) > 200000 else c_texts, max_features, max_df=0.02)
    t0 = time.time()
    res_p = retrieve_with_scores(q_texts, q_ids, c_texts, c_ids, vec_p, k=k)
    low = [qid for qid, lst in res_p.items() if len(lst) < 5 or (lst and lst[0][1] < 0.35)]
    if low:
        vec_f = fit_vec(c_texts[:200000] if len(c_texts) > 200000 else c_texts, max_features, max_df=1.0)
        low_idx = {qid: i for i, qid in enumerate(q_ids)}
        low_qt = [q_texts[low_idx[q]] for q in low]
        res_f = retrieve_with_scores(low_qt, low, c_texts, c_ids, vec_f, k=k)
        for qid in low:
            merged = dict(res_p[qid])
            for cid, sc in res_f[qid]:
                if cid not in merged or sc > merged[cid]:
                    merged[cid] = sc
            res_p[qid] = sorted(merged.items(), key=lambda x: -x[1])[:k]
    return res_p
