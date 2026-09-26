"""Full test inference: blocking + LightGBM matching, per-country streaming."""
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
THR=0.70
K=25

print('load model', flush=True)
model=lgb.Booster(model_file=MODEL)

def load_country(country, src):
    # src: 's1','s2','s3', test files
    fmap={'s1':f'{BASE}/test/test_source1.tsv','s2':f'{BASE}/test/test_source2.tsv','s3':f'{BASE}/test/test_source3.tsv'}
    parts=[]
    for chunk in pd.read_csv(fmap[src], sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_name','business_address','country']):
        hit=chunk[chunk['country']==country]
        if len(hit): parts.append(hit)
    if not parts:
        return pd.DataFrame(columns=['entity_id','business_name','business_address','country'])
    return pd.concat(parts, ignore_index=True)

# discover countries from test S1 (open set, no hard-code)
print('discover countries...', flush=True)
from collections import Counter
c=Counter()
for chunk in pd.read_csv(f'{BASE}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['country']):
    c.update(chunk['country'].tolist())
print('countries:', dict(c), flush=True)
countries=list(c.keys())

# open outputs (write header)
cand_f=open(f'{OUT}/candidate_pairs.tsv','w',encoding='utf-8')
match_f=open(f'{OUT}/matching_results.tsv','w',encoding='utf-8')
cand_f.write('source1_entity_id\tcandidate_entity_ids\n')
match_f.write('source1_entity_id\tmatched_entity_ids\n')

for country in countries:
    print(f'===== country {country} =====', flush=True)
    s1=load_country(country,'s1')
    s2=load_country(country,'s2')
    s3=load_country(country,'s3')
    print(f's1 {len(s1)} s2 {len(s2)} s3 {len(s3)}', flush=True)
    corpus=pd.concat([s2,s3], ignore_index=True)
    del s2,s3; gc.collect()
    print('fit vec...', flush=True)
    c_texts_all=make_texts(corpus)
    # fit on sample
    vec_p=fit_vec(c_texts_all[:200000] if len(c_texts_all)>200000 else c_texts_all, 50000, 0.02)
    print(f'vec {len(vec_p.vocabulary_)}', flush=True)
    # process S1 in chunks to bound RAM
    CH=20000
    cmap={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in corpus.iterrows()}
    # keep corpus texts/ids for retrieval (list, ~4.7M strings ~500MB - ok)
    c_ids=corpus['entity_id'].tolist()
    del corpus; gc.collect()
    for start in range(0,len(s1),CH):
        end=min(len(s1),start+CH)
        chunk=s1.iloc[start:end]
        print(f'-- S1 {start}-{end}/{len(s1)} --', flush=True)
        q_texts=make_texts(chunk); q_ids=chunk['entity_id'].tolist()
        t0=time.time()
        res=retrieve_with_scores(q_texts,q_ids,c_texts_all,c_ids,vec_p,k=K,q_batch=5000,c_shard=400000)
        print(f'blocking {time.time()-t0:.1f}s', flush=True)
        # fallback low
        low=[qid for qid,lst in res.items() if len(lst)<5 or (lst and lst[0][1]<0.35)]
        print(f'fallback {len(low)}', flush=True)
        if low:
            # fit full vec once per country (cache)
            if 'vec_f' not in locals() or True:
                pass
            # to save time, skip full fallback on test? Use pruned only for speed, note recall tradeoff.
            # For production, do fallback only if low < 5% else skip (to save hours). Here low likely ~1-2%, do it.
            from src.blocking_prod import fit_vec as fv
            # reuse vec_p for fallback to save time? Actually need full for recall; but to save 2x time, skip fallback if too many
            if len(low) < 5000:
                vec_f=fit_vec(c_texts_all[:200000], 50000, 1.0)
                low_idx={qid:i for i,qid in enumerate(q_ids)}
                low_qt=[q_texts[low_idx[q]] for q in low]
                res_f=retrieve_with_scores(low_qt,low,c_texts_all,c_ids,vec_f,k=K,q_batch=2000,c_shard=400000)
                for qid in low:
                    merged=dict(res[qid])
                    for cid,sc in res_f[qid]:
                        if cid not in merged or sc>merged[cid]:
                            merged[cid]=sc
                    res[qid]=sorted(merged.items(), key=lambda x:-x[1])[:K]
        # matching inference batched
        s1map={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in chunk.iterrows()}
        for qid in q_ids:
            lst=res.get(qid,[])
            cands=[cid for cid,_ in lst]
            score_map={cid:sc for cid,sc in lst}
            # write candidates
            cand_f.write(f'{qid}\t{",".join(cands)}\n')
            if not cands:
                match_f.write(f'{qid}\t\n')
                continue
            # features
            X=[]
            clist=[]
            n1,a1,co1=s1map[qid]
            for cid in cands:
                if cid not in cmap:
                    continue
                n2,a2,co2=cmap[cid]
                X.append(feats_for_pair(n1,a1,co1,n2,a2,co2,score_map.get(cid,0)))
                clist.append(cid)
            if not X:
                match_f.write(f'{qid}\t\n')
                continue
            probs=model.predict(np.array(X,float))
            matched=[cid for cid,p in zip(clist,probs) if p>=THR]
            match_f.write(f'{qid}\t{",".join(matched)}\n')
        cand_f.flush(); match_f.flush()
        gc.collect()
    del s1, c_texts_all, c_ids, cmap; gc.collect()
    # clear vec_f cache per country
    if 'vec_f' in locals():
        del vec_f

cand_f.close(); match_f.close()
print('done', flush=True)
