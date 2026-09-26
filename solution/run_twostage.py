import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts, retrieve_sharded, topk_sparse_row
from src.evaluate import blocking_recall
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
# small test: 500 India queries, 300k corpus
print('loading...')
c2=pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c2=c2[c2['country']=='India']
c3=pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c3=c3[c3['country']=='India']
corpus=pd.concat([c2.head(150000),c3.head(150000)],ignore_index=True)
s1pool=pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=200000, usecols=['entity_id','business_name','business_address','country'])
s1q=s1pool[s1pool['country']=='India'].sample(500, random_state=21)
want=set(s1q['entity_id'])
true_dict={}
for chunk in pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000):
    hit=chunk[chunk['source1_entity_id'].isin(want)]
    for _,r in hit.iterrows():
        v=r['matched_entity_ids']
        true_dict[r['source1_entity_id']]=[] if v=='' else v.split(',')
    if len(true_dict)>=500:
        break
for eid in want:
    if eid not in true_dict:
        true_dict[eid]=[]
need=set()
for v in true_dict.values():
    need.update(v)
missing=need-set(corpus['entity_id'].values)
for src in ['train/train_source2.tsv','train/train_source3.tsv']:
    if not missing: break
    for chunk in pd.read_csv(f'{base}/{src}', sep='\t', dtype=str, keep_default_na=False, chunksize=800000, usecols=['entity_id','business_name','business_address','country']):
        hit=chunk[chunk['entity_id'].isin(missing)]
        if len(hit):
            corpus=pd.concat([corpus,hit],ignore_index=True)
            missing-=set(hit['entity_id'].values)
            if not missing: break
print('corpus',len(corpus))
q_texts=make_texts(s1q); q_ids=s1q['entity_id'].tolist()
c_texts=make_texts(corpus); c_ids=corpus['entity_id'].tolist()
# fit full vec, then create pruned vec by dropping high-df terms? Simpler: fit two vecs
# full
full_vec=TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=50000, min_df=2, sublinear_tf=True)
full_vec.fit(c_texts[:200000])
print('full vocab',len(full_vec.vocabulary_))
# pruned: max_df absolute? sklearn max_df as absolute count if int
pruned_vec=TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=50000, min_df=2, max_df=3000, sublinear_tf=True)
# NOTE: max_df=3000 on 200k fit sample ~ 1.5% ; for full 4.7M this would be 0.06% - very strict. For test, use 3000 on 300k ~1%
pruned_vec.fit(c_texts[:200000])
print('pruned vocab',len(pruned_vec.vocabulary_))
# two-stage: pruned first, fallback to full if <10 cands or max score <0.3
# need scores: modify retrieve to return scores
import scipy.sparse as sp
def retrieve_with_scores(qt, qi, ct, ci, vec, k, qb=1000, cs=100000):
    C_mats=[]
    # pre-transform corpus shards
    shards=[]
    for s in range(0,len(ct),cs):
        e=min(len(ct),s+cs)
        shards.append((vec.transform(ct[s:e]), ci[s:e]))
    out={qid: {} for qid in qi}
    for qs in range(0,len(qt),qb):
        qe=min(len(qt),qs+qb)
        Q=vec.transform(qt[qs:qe])
        # accumulate across shards
        batch_best=[dict() for _ in range(qe-qs)]
        for C, cids in shards:
            S=(Q @ C.T).tocsr()
            for i in range(qe-qs):
                r=S.getrow(i)
                if r.nnz==0: continue
                for col, score in zip(r.indices, r.data):
                    cid=cids[col]
                    d=batch_best[i]
                    if cid not in d or score>d[cid]:
                        d[cid]=float(score)
        for i,qid in enumerate(qi[qs:qe]):
            d=batch_best[i]
            top=sorted(d.items(), key=lambda x:-x[1])[:k]
            out[qid]=top  # list of (cid,score)
    return out

t0=time.time()
pruned_res=retrieve_with_scores(q_texts,q_ids,c_texts,c_ids,pruned_vec,k=30)
dt1=time.time()-t0
print('pruned time %.1f' % dt1)
# stats: how many have <10 or low score?
low=[qid for qid, lst in pruned_res.items() if len(lst)<10 or (lst and lst[0][1]<0.4)]
print('low queries',len(low), '/', len(pruned_res))
# fallback full for low
t0=time.time()
if low:
    low_idx=[q_ids.index(q) for q in low]
    low_qt=[q_texts[i] for i in low_idx]
    full_low=retrieve_with_scores(low_qt,low,c_texts,c_ids,full_vec,k=30)
    for qid in low:
        # merge
        merged=dict(pruned_res[qid])
        for cid,sc in full_low[qid]:
            if cid not in merged or sc>merged[cid]:
                merged[cid]=sc
        pruned_res[qid]=sorted(merged.items(), key=lambda x:-x[1])[:30]
dt2=time.time()-t0
print('fallback time %.1f total %.1f' % (dt2, dt1+dt2))
# eval
final={qid:[cid for cid,_ in lst] for qid,lst in pruned_res.items()}
m=blocking_recall(final,true_dict)
print('two-stage recall %.4f' % m['micro_recall'])
# compare full only
t0=time.time()
full_res=retrieve_with_scores(q_texts,q_ids,c_texts,c_ids,full_vec,k=30)
dtf=time.time()-t0
finalf={qid:[cid for cid,_ in lst] for qid,lst in full_res.items()}
mf=blocking_recall(finalf,true_dict)
print('full time %.1f recall %.4f' % (dtf, mf['micro_recall']))
