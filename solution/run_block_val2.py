import sys
sys.path.insert(0, "D:/projects/Amazon ML/solution")
from src.normalize import combined_text
import pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts, retrieve_sharded, topk_sparse_row
from src.evaluate import blocking_recall

base = "D:/projects/Amazon ML/student_resource/student_resource/dataset"

# Build proper validation: sample 3000 S1 across countries, find their GT by scanning full GT file
import random
# get random S1 sample from full file via reservoir? simpler: read first 500k, sample
s1pool = pd.read_csv(f"{base}/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=400000)
s1sample = s1pool.sample(3000, random_state=42)
want = set(s1sample["entity_id"])
print("sampled", len(want), s1sample["country"].value_counts().to_dict())

# scan GT for these
true_dict = {}
for chunk in pd.read_csv(f"{base}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
    hit = chunk[chunk["source1_entity_id"].isin(want)]
    for _,r in hit.iterrows():
        v = r["matched_entity_ids"]
        true_dict[r["source1_entity_id"]] = [] if v=="" else v.split(",")
    if len(true_dict) >= len(want)*0.95:
        pass
for eid in want:
    if eid not in true_dict:
        true_dict[eid] = []  # treat missing as singleton (shouldn't happen)
print("found gt for", len(true_dict))
need_ids = set()
for v in true_dict.values():
    need_ids.update(v)
print("need true corpus ids", len(need_ids))

# corpus: per country, 150k random + all needed
from collections import defaultdict
corpus_parts = []
for country in ["US","India"]:
    print("== corpus", country)
    c2 = pd.read_csv(f"{base}/train/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=600000)
    c2 = c2[c2["country"]==country].head(75000)
    c3 = pd.read_csv(f"{base}/train/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=600000)
    c3 = c3[c3["country"]==country].head(75000)
    part = pd.concat([c2,c3], ignore_index=True)
    corpus_parts.append(part)
corpus = pd.concat(corpus_parts, ignore_index=True)
print("base corpus", len(corpus))
# add missing true ids
missing = need_ids - set(corpus["entity_id"].values)
print("missing", len(missing))
for src in ["train/train_source2.tsv","train/train_source3.tsv"]:
    if not missing: break
    for chunk in pd.read_csv(f"{base}/{src}", sep="\t", dtype=str, keep_default_na=False, chunksize=800000):
        hit = chunk[chunk["entity_id"].isin(missing)]
        if len(hit):
            corpus = pd.concat([corpus, hit], ignore_index=True)
            missing -= set(hit["entity_id"].values)
            print(src, "added", len(hit), "left", len(missing))
            if not missing: break
print("final corpus", len(corpus), corpus["country"].value_counts().to_dict())

# test word-level vs char-level
q_texts_all = make_texts(s1sample)
q_ids_all = s1sample["entity_id"].tolist()
q_countries = s1sample["country"].tolist()

for analyzer, ngram, mf in [( "word",(1,2),100000), ("char_wb",(3,5),30000)]:
    print(f"\n=== analyzer={analyzer} {ngram} mf={mf} ===")
    # fit per-country? fit globally on corpus sample for speed
    c_texts_all = make_texts(corpus)
    vec = TfidfVectorizer(analyzer=analyzer, ngram_range=ngram, max_features=mf, min_df=2, sublinear_tf=True)
    t0=time.time()
    vec.fit(c_texts_all[:150000])
    print("fit", time.time()-t0, "vocab", len(vec.vocabulary_))
    # per-country retrieval (realistic)
    for country in ["US","India"]:
        idx_q = [i for i,c in enumerate(q_countries) if c==country]
        qt = [q_texts_all[i] for i in idx_q]
        qi = [q_ids_all[i] for i in idx_q]
        cmask = corpus["country"].values==country
        ct = [t for t,m in zip(c_texts_all, cmask) if m]
        ci = corpus["entity_id"].values[cmask].tolist()
        td = {k:true_dict[k] for k in qi}
        for k in [10,30]:
            t0=time.time()
            res = retrieve_sharded(qt, qi, ct, ci, vec, k=k, q_batch=1000, c_shard=100000)
            dt=time.time()-t0
            m = blocking_recall(res, td)
            print(f" {country} K={k} time={dt:.1f}s micro={m['micro_recall']:.4f} macro={m['macro_recall']:.4f}")
