import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
from src.normalize import combined_text
import pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts, retrieve_sharded
from src.evaluate import blocking_recall
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
s1pool=pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=150000)
s1sample=s1pool[s1pool['country']=='US'].sample(500, random_state=8)
want=set(s1sample['entity_id'])
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
c2=pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=400000)
c2=c2[c2['country']=='US'].head(40000)
c3=pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=400000)
c3=c3[c3['country']=='US'].head(40000)
corpus=pd.concat([c2,c3],ignore_index=True)
missing=need-set(corpus['entity_id'].values)
for src in ['train/train_source2.tsv','train/train_source3.tsv']:
    if not missing:
        break
    for chunk in pd.read_csv(f'{base}/{src}', sep='\t', dtype=str, keep_default_na=False, chunksize=800000):
        hit=chunk[chunk['entity_id'].isin(missing)]
        if len(hit):
            corpus=pd.concat([corpus,hit],ignore_index=True)
            missing-=set(hit['entity_id'].values)
            if not missing:
                break
print('corpus',len(corpus),'need',len(need),'missing left',len(missing))
q_texts=make_texts(s1sample)
q_ids=s1sample['entity_id'].tolist()
c_texts=make_texts(corpus)
c_ids=corpus['entity_id'].tolist()
for ana,ng in [('word',(1,1)),('word',(1,2))]:
    vec=TfidfVectorizer(analyzer=ana, ngram_range=ng, max_features=80000, min_df=2, sublinear_tf=True)
    vec.fit(c_texts[:60000])
    print(ana,ng,'vocab',len(vec.vocabulary_))
    t0=time.time()
    res=retrieve_sharded(q_texts,q_ids,c_texts,c_ids,vec,k=30,q_batch=500,c_shard=50000)
    dt=time.time()-t0
    m=blocking_recall(res,true_dict)
    print('K30 time %.1f micro %.4f' % (dt, m['micro_recall']))
