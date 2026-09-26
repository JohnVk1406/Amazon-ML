"""Dry-run: 5k test S1 sample -> outputs + validator."""
import os, sys, time, gc
sys.path.insert(0, 'D:/projects/Amazon ML/solution')
import pandas as pd, numpy as np
from src.blocking_prod import fit_vec, retrieve_with_scores
from src.blocking import make_texts
from src.features import feats_for_pair
import lightgbm as lgb
BASE='D:/projects/Amazon ML/student_resource/student_resource/dataset'
OUT='D:/projects/Amazon ML/solution/output'
os.makedirs(OUT, exist_ok=True)
MODEL='D:/projects/Amazon ML/solution/lgb_global.txt'
THR=0.70; K=25
model=lgb.Booster(model_file=MODEL)
# sample S1: 2000 US, 2000 India, 1000 France (from first 500k rows; ensure France included - France may be later in file, scan)
print('sampling S1...', flush=True)
s1parts={}; need={'US':2000,'India':2000,'France':1000}
for chunk in pd.read_csv(f'{BASE}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=200000, usecols=['entity_id','business_name','business_address','country']):
    for co in list(need.keys()):
        if need[co]<=0: continue
        hit=chunk[chunk['country']==co].head(need[co])
        if len(hit):
            s1parts.setdefault(co,[]).append(hit)
            need[co]-=len(hit)
    if all(v<=0 for v in need.values()): break
print('remaining need',need)
s1sample=pd.concat([pd.concat(v) for v in s1parts.values()], ignore_index=True)
print(s1sample['country'].value_counts().to_dict(), flush=True)
# for validator we need full test-dir? validator checks IDs exist in test set - our sample outputs will fail "missing entities" (expected, since sample). We'll still run validator to check format, plus manual checks.
# load corpus for those countries (full S2+S3 per country? for dry-run use 300k per country to be fast)
from collections import defaultdict
corpus_parts=[]
for co in s1sample['country'].unique():
    for src in [f'{BASE}/test/test_source2.tsv', f'{BASE}/test/test_source3.tsv']:
        cnt=0
        for chunk in pd.read_csv(src, sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_name','business_address','country']):
            hit=chunk[chunk['country']==co]
            if len(hit):
                # take up to 150k per file per country
                take=hit.head(max(0,150000-cnt))
                corpus_parts.append(take); cnt+=len(take)
                if cnt>=150000: break
        print(co, src, cnt, flush=True)
corpus=pd.concat(corpus_parts, ignore_index=True)
print('corpus',len(corpus), corpus['country'].value_counts().to_dict(), flush=True)
# blocking per country
all_res={}
for co in s1sample['country'].unique():
    print('blocking',co, flush=True)
    s1c=s1sample[s1sample['country']==co]
    coc=corpus[corpus['country']==co]
    qt=make_texts(s1c); qi=s1c['entity_id'].tolist()
    ct=make_texts(coc); ci=coc['entity_id'].tolist()
    vec=fit_vec(ct[:100000],50000,0.02)
    res=retrieve_with_scores(qt,qi,ct,ci,vec,k=K,q_batch=1000,c_shard=100000)
    all_res.update(res)
print('avg cands', np.mean([len(v) for v in all_res.values()]), flush=True)
# matching
s1map={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in s1sample.iterrows()}
cmap={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in corpus.iterrows()}
cand_lines=[]; match_lines=[]
for qid, lst in all_res.items():
    cands=[c for c,_ in lst]
    sm={c:s for c,s in lst}
    cand_lines.append((qid,cands))
    n1,a1,co1=s1map[qid]
    X=[]; cl=[]
    for cid in cands:
        if cid not in cmap: continue
        n2,a2,co2=cmap[cid]
        X.append(feats_for_pair(n1,a1,co1,n2,a2,co2,sm.get(cid,0))); cl.append(cid)
    if not X:
        match_lines.append((qid,[])); continue
    probs=model.predict(np.array(X,float))
    matched=[c for c,p in zip(cl,probs) if p>=THR]
    match_lines.append((qid,matched))
# write dry outputs
with open(f'{OUT}/candidate_pairs_dry.tsv','w',encoding='utf-8') as f:
    f.write('source1_entity_id\tcandidate_entity_ids\n')
    for qid,cands in cand_lines:
        f.write(f'{qid}\t{",".join(cands)}\n')
with open(f'{OUT}/matching_results_dry.tsv','w',encoding='utf-8') as f:
    f.write('source1_entity_id\tmatched_entity_ids\n')
    for qid,m in match_lines:
        f.write(f'{qid}\t{",".join(m)}\n')
print('wrote dry outputs', flush=True)
# stats
import collections
mc=collections.Counter(len(m) for _,m in match_lines)
print('match count histogram (num_matches: num_entities):', dict(sorted(mc.items())[:10]), flush=True)
print('empty preds:', sum(1 for _,m in match_lines if len(m)==0), '/', len(match_lines), flush=True)
print('avg matches per non-empty:', np.mean([len(m) for _,m in match_lines if m]) if any(m for _,m in match_lines) else 0, flush=True)
