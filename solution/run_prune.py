import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts, retrieve_sharded
from src.evaluate import blocking_recall
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
print('loading...')
c2=pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c2=c2[c2['country']=='India']
c3=pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=800000, usecols=['entity_id','business_name','business_address','country'])
c3=c3[c3['country']=='India']
corpus=pd.concat([c2.head(150000),c3.head(150000)],ignore_index=True)
print('corpus',len(corpus))
# queries 500 with gt
s1pool=pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=200000, usecols=['entity_id','business_name','business_address','country'])
s1q=s1pool[s1pool['country']=='India'].sample(500, random_state=11)
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
print('missing',len(missing))
for src in ['train/train_source2.tsv','train/train_source3.tsv']:
    if not missing: break
    for chunk in pd.read_csv(f'{base}/{src}', sep='\t', dtype=str, keep_default_na=False, chunksize=800000, usecols=['entity_id','business_name','business_address','country']):
        hit=chunk[chunk['entity_id'].isin(missing)]
        if len(hit):
            corpus=pd.concat([corpus,hit],ignore_index=True)
            missing-=set(hit['entity_id'].values)
            if not missing: break
print('final corpus',len(corpus))
q_texts=make_texts(s1q); q_ids=s1q['entity_id'].tolist()
c_texts=make_texts(corpus); c_ids=corpus['entity_id'].tolist()
for maxf, mf in [(1.0,50000),(0.02,50000),(0.005,50000),(0.002,30000)]:
    vec=TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=mf, min_df=2, max_df=maxf, sublinear_tf=True)
    t0=time.time(); vec.fit(c_texts[:200000]); ft=time.time()-t0
    t0=time.time()
    res=retrieve_sharded(q_texts,q_ids,c_texts,c_ids,vec,k=30,q_batch=1000,c_shard=100000)
    dt=time.time()-t0
    m=blocking_recall(res,true_dict)
    print('max_df=%.4f mf=%d fit=%.1f time=%.1f msq=%.1f recall=%.4f vocab=%d' % (maxf,mf,ft,dt,dt/500*1000,m['micro_recall'],len(vec.vocabulary_)))
