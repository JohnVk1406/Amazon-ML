import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, time, numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts
from src.features import feats_for_pair
from src.evaluate import f05_macro, blocking_recall
import lightgbm as lgb
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
print('loading train sample India 3000 S1...')
s1pool=pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=400000, usecols=['entity_id','business_name','business_address','country'])
s1q=s1pool[s1pool['country']=='India'].sample(3000, random_state=99)
want=set(s1q['entity_id'])
true_dict={}
for chunk in pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000):
    hit=chunk[chunk['source1_entity_id'].isin(want)]
    for _,r in hit.iterrows():
        v=r['matched_entity_ids']
        true_dict[r['source1_entity_id']]=[] if v=='' else v.split(',')
    if len(true_dict)>=3000: break
for eid in want:
    if eid not in true_dict: true_dict[eid]=[]
need=set()
for v in true_dict.values(): need.update(v)
print('need',len(need))
# corpus 200k India + needed
c2=pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c2=c2[c2['country']=='India'].head(100000)
c3=pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c3=c3[c3['country']=='India'].head(100000)
corpus=pd.concat([c2,c3],ignore_index=True)
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
# blocking
from src.blocking import retrieve_sharded
q_texts=make_texts(s1q); q_ids=s1q['entity_id'].tolist()
c_texts=make_texts(corpus); c_ids=corpus['entity_id'].tolist()
vec=TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=50000, min_df=2, sublinear_tf=True)
vec.fit(c_texts[:150000])
print('vec',len(vec.vocabulary_))
# retrieve with scores
import scipy.sparse as sp
from src.blocking import topk_sparse_row
def retrieve_scores(qt,qi,ct,ci,vec,k=30,qb=1000,cs=100000):
    shards=[]
    for s in range(0,len(ct),cs):
        e=min(len(ct),s+cs)
        shards.append((vec.transform(ct[s:e]), ci[s:e]))
    out={}
    for qs in range(0,len(qt),qb):
        qe=min(len(qt),qs+qb)
        Q=vec.transform(qt[qs:qe])
        batch=[dict() for _ in range(qe-qs)]
        for C,cids in shards:
            S=(Q@C.T).tocsr()
            for i in range(qe-qs):
                r=S.getrow(i)
                if r.nnz==0: continue
                for col,sc in zip(r.indices,r.data):
                    cid=cids[col]
                    d=batch[i]
                    if cid not in d or sc>d[cid]:
                        d[cid]=float(sc)
        for i,qid in enumerate(qi[qs:qe]):
            top=sorted(batch[i].items(), key=lambda x:-x[1])[:k]
            out[qid]=top
    return out
t0=time.time()
cand_scores=retrieve_scores(q_texts,q_ids,c_texts,c_ids,vec,k=30)
print('blocking %.1fs' % (time.time()-t0))
cand_dict={k:[c for c,_ in v] for k,v in cand_scores.items()}
m=blocking_recall(cand_dict,true_dict)
print('blocking recall',m)
# build maps
s1map={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in s1q.iterrows()}
cmap={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in corpus.iterrows()}
# split train/val by S1
import random
random.seed(1)
qids=list(q_ids)
random.shuffle(qids)
tr=qids[:2000]; va=qids[2000:]
def build_rows(qlist):
    X=[]; y=[]
    for qid in qlist:
        tset=set(true_dict.get(qid,[]))
        for cid,sc in cand_scores.get(qid,[]):
            if cid not in cmap: continue
            n1,a1,co1=s1map[qid]
            n2,a2,co2=cmap[cid]
            f=feats_for_pair(n1,a1,co1,n2,a2,co2,sc)
            X.append(f); y.append(1 if cid in tset else 0)
    return np.array(X,float), np.array(y,int)
Xtr,ytr=build_rows(tr); Xva,yva=build_rows(va)
print('train pairs',len(ytr),'pos rate',ytr.mean(),'val pairs',len(yva),'pos',yva.mean())
trn=lgb.Dataset(Xtr,label=ytr)
val=lgb.Dataset(Xva,label=yva,reference=trn)
params={'objective':'binary','metric':'binary_logloss','boosting_type':'gbdt','num_leaves':63,'learning_rate':0.05,'feature_fraction':0.9,'bagging_fraction':0.8,'bagging_freq':5,'verbose':-1,'num_threads':8}
model=lgb.train(params,trn,num_boost_round=500,valid_sets=[val],callbacks=[lgb.early_stopping(50),lgb.log_evaluation(50)])
model.save_model('D:/projects/Amazon ML/solution/lgb_india.txt')
# threshold tune for F0.5 on val S1s
from src.evaluate import f05_macro
# score val candidates
probs=model.predict(Xva)
# need per-pair qid/cid order
pairs=[]
idx=0
for qid in va:
    for cid,sc in cand_scores.get(qid,[]):
        if cid not in cmap: continue
        pairs.append((qid,cid,probs[idx]))
        idx+=1
for thr in [0.3,0.5,0.6,0.7,0.8,0.85,0.9]:
    pred={}
    for qid in va:
        pred[qid]=[]
    for qid,cid,p in pairs:
        if p>=thr:
            pred[qid].append(cid)
    true_va={k:true_dict[k] for k in va}
    s=f05_macro(pred,true_va)
    print('thr %.2f F05 %.4f P %.4f R %.4f' % (thr,s['F05'],s['P'],s['R']))
