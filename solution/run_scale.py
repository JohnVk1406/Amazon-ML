import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, time
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts, retrieve_sharded
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
# corpus 500k India (400k+100k?) load 500k rows, filter India ~200k? To get 400k need more rows. Let's load 1.2M rows to get ~480k India
print('loading corpus...')
c2=pd.read_csv(f'{base}/train/train_source2.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=1200000, usecols=['entity_id','business_name','business_address','country'])
c2=c2[c2['country']=='India']
print('c2 india',len(c2))
c3=pd.read_csv(f'{base}/train/train_source3.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=1200000, usecols=['entity_id','business_name','business_address','country'])
c3=c3[c3['country']=='India']
print('c3 india',len(c3))
corpus=pd.concat([c2,c3],ignore_index=True)
print('corpus',len(corpus))
c_texts=make_texts(corpus)
c_ids=corpus['entity_id'].tolist()
vec=TfidfVectorizer(analyzer='word', ngram_range=(1,1), max_features=50000, min_df=2, sublinear_tf=True)
t0=time.time()
vec.fit(c_texts[:200000])
print('fit %.1f vocab %d' % (time.time()-t0, len(vec.vocabulary_)))
# queries 2000
s1pool=pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=300000, usecols=['entity_id','business_name','business_address','country'])
s1q=s1pool[s1pool['country']=='India'].head(2000)
q_texts=make_texts(s1q); q_ids=s1q['entity_id'].tolist()
for qb, cs in [(1000,100000),(2000,200000),(5000,500000)]:
    t0=time.time()
    res=retrieve_sharded(q_texts,q_ids,c_texts,c_ids,vec,k=20,q_batch=qb,c_shard=cs)
    dt=time.time()-t0
    print('qb=%d cs=%d time=%.1f ms_per_q=%.2f' % (qb,cs,dt,dt/len(q_ids)*1000))
