"""Quick blocking validation on a sample."""
import pandas as pd, random, time, sys
sys.path.insert(0, "D:/projects/Amazon ML/solution/src")
sys.path.insert(0, "D:/projects/Amazon ML/solution")
from src.normalize import combined_text
from src.blocking import fit_vectorizer, retrieve_sharded, make_texts
from src.evaluate import blocking_recall

base = "D:/projects/Amazon ML/student_resource/student_resource/dataset"
# sample: 2000 S1 US + their true matches must be in corpus sample
# To guarantee recall measurable, take corpus = 100k random US from S2+S3 + all true matches for sampled S1
print("loading s1 sample...")
s1 = pd.read_csv(f"{base}/train/train_source1.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=60000)
s1 = s1[s1["country"]=="US"].head(2000)
print("s1", len(s1))
gt_full = pd.read_csv(f"{base}/train/train_ground_truth.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=60000)
# gt order != s1 order; filter to sampled s1 ids
gt = gt_full[gt_full["source1_entity_id"].isin(set(s1["entity_id"]))]
print("gt matched rows", len(gt))
true_dict = {}
need_ids = set()
for _,r in gt.iterrows():
    v = r["matched_entity_ids"]
    ids = [] if v=="" else v.split(",")
    true_dict[r["source1_entity_id"]] = ids
    need_ids.update(ids)
# ensure singletons included
for eid in s1["entity_id"].values:
    if eid not in true_dict:
        # find in full gt? need scan; for now treat as singleton if not in gt chunk
        true_dict[eid] = []
print("need true ids", len(need_ids))

# corpus sample
print("loading corpus...")
c2 = pd.read_csv(f"{base}/train/train_source2.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=300000)
c2 = c2[c2["country"]=="US"]
c3 = pd.read_csv(f"{base}/train/train_source3.tsv", sep="\t", dtype=str, keep_default_na=False, nrows=300000)
c3 = c3[c3["country"]=="US"]
corpus = pd.concat([c2.head(50000), c3.head(50000)], ignore_index=True)
# add all needed true ids by scanning full files for them
import pandas as pd
for src in ["train/train_source2.tsv","train/train_source3.tsv"]:
    missing = need_ids - set(corpus["entity_id"].values)
    if not missing:
        break
    print(f"scanning {src} for {len(missing)} missing...")
    for chunk in pd.read_csv(f"{base}/{src}", sep="\t", dtype=str, keep_default_na=False, chunksize=500000):
        hit = chunk[chunk["entity_id"].isin(missing)]
        if len(hit):
            corpus = pd.concat([corpus, hit], ignore_index=True)
            missing = missing - set(hit["entity_id"].values)
            print("  found", len(hit), "remaining", len(missing))
            if not missing:
                break
print("corpus size", len(corpus))

q_texts = make_texts(s1)
q_ids = s1["entity_id"].tolist()
c_texts = make_texts(corpus)
c_ids = corpus["entity_id"].tolist()

print("fitting vectorizer on corpus sample...")
t0=time.time()
vec = fit_vectorizer(c_texts[:80000] if len(c_texts)>80000 else c_texts, max_features=30000)
print("fit done", time.time()-t0, "vocab", len(vec.vocabulary_))

for k in [10, 30, 50]:
    t0=time.time()
    res = retrieve_sharded(q_texts, q_ids, c_texts, c_ids, vec, k=k, q_batch=1000, c_shard=50000)
    dt=time.time()-t0
    m = blocking_recall(res, true_dict)
    print(f"K={k} time={dt:.1f}s recall_micro={m['micro_recall']:.4f} macro={m['macro_recall']:.4f} avg_cands={m['avg_cands']:.1f}")
