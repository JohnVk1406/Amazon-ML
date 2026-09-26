"""Train global LightGBM matcher on mixed US+India."""
import sys, os
sys.path.insert(0, 'D:/projects/Amazon ML/solution')
import pandas as pd, numpy as np, time, random
from sklearn.feature_extraction.text import TfidfVectorizer
from src.blocking import make_texts
from src.blocking_prod import retrieve_with_scores, fit_vec
from src.features import feats_for_pair, FEAT_NAMES
from src.evaluate import f05_macro, blocking_recall
import lightgbm as lgb
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'
random.seed(42); np.random.seed(42)

def load_sample(n_s1_per_country=8000, k=30):
    # S1 sample
    s1_us=[]; s1_in=[]
    for chunk in pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_name','business_address','country']):
        u=chunk[chunk['country']=='US']
        ii=chunk[chunk['country']=='India']
        s1_us.append(u); s1_in.append(ii)
        if sum(len(x) for x in s1_us)>60000 and sum(len(x) for x in s1_in)>60000:
            break
    s1_us=pd.concat(s1_us).sample(n_s1_per_country, random_state=1)
    s1_in=pd.concat(s1_in).sample(n_s1_per_country, random_state=2)
    s1=pd.concat([s1_us,s1_in], ignore_index=True)
    want=set(s1['entity_id'])
    true_dict={}
    for chunk in pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000):
        hit=chunk[chunk['source1_entity_id'].isin(want)]
        for _,r in hit.iterrows():
            v=r['matched_entity_ids']
            true_dict[r['source1_entity_id']]=[] if v=='' else v.split(',')
        if len(true_dict)>=len(want)*0.99:
            # don't break early, need all? break when found most
            if len(true_dict)>=len(want)-100:
                pass
    for eid in want:
        if eid not in true_dict:
            # scan remaining? assume singleton (rare)
            true_dict[eid]=[]
    need=set()
    for v in true_dict.values():
        need.update(v)
    print('s1',len(s1),'gt found',len(true_dict),'need ids',len(need))
    # corpus: 150k per country per source + needed
    corp_parts=[]
    for country in ['US','India']:
        for src, nrows in [('train/train_source2.tsv',900000),('train/train_source3.tsv',900000)]:
            df=pd.read_csv(f'{base}/{src}', sep='\t', dtype=str, keep_default_na=False, nrows=nrows, usecols=['entity_id','business_name','business_address','country'])
            df=df[df['country']==country].head(75000)
            corp_parts.append(df)
    corpus=pd.concat(corp_parts, ignore_index=True)
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
    print('corpus',len(corpus))
    return s1, corpus, true_dict

if __name__=='__main__':
    s1, corpus, true_dict = load_sample(8000, 30)
    # blocking per country
    all_scores={}
    for country in ['US','India']:
        print('=== blocking',country)
        s1c=s1[s1['country']==country]
        coc=corpus[corpus['country']==country]
        q_texts=make_texts(s1c); q_ids=s1c['entity_id'].tolist()
        c_texts=make_texts(coc); c_ids=coc['entity_id'].tolist()
        vec=fit_vec(c_texts[:150000], 50000, 0.02)
        res=retrieve_with_scores(q_texts,q_ids,c_texts,c_ids,vec,k=30,q_batch=2000,c_shard=150000)
        all_scores.update(res)
    cand={k:[c for c,_ in v] for k,v in all_scores.items()}
    print('blocking', blocking_recall(cand,true_dict))
    # maps
    s1map={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in s1.iterrows()}
    cmap={r['entity_id']:(r['business_name'],r['business_address'],r['country']) for _,r in corpus.iterrows()}
    # split
    qids=list(s1['entity_id'].values)
    random.shuffle(qids)
    n=len(qids); tr=qids[:int(n*0.8)]; va=qids[int(n*0.8):]
    def build(qlist):
        X=[]; y=[]
        for qid in qlist:
            tset=set(true_dict.get(qid,[]))
            for cid,sc in all_scores.get(qid,[]):
                if cid not in cmap: continue
                n1,a1,c1=s1map[qid]; n2,a2,c2=cmap[cid]
                X.append(feats_for_pair(n1,a1,c1,n2,a2,c2,sc)); y.append(1 if cid in tset else 0)
        return np.array(X,float), np.array(y,int)
    Xtr,ytr=build(tr); Xva,yva=build(va)
    print('pairs',len(ytr),ytr.mean(),len(yva),yva.mean())
    trn=lgb.Dataset(Xtr,label=ytr); val=lgb.Dataset(Xva,label=yva,reference=trn)
    params={'objective':'binary','metric':'binary_logloss','num_leaves':127,'learning_rate':0.05,'feature_fraction':0.8,'bagging_fraction':0.8,'bagging_freq':5,'min_data_in_leaf':100,'verbose':-1,'num_threads':8}
    model=lgb.train(params,trn,800,valid_sets=[val],callbacks=[lgb.early_stopping(80),lgb.log_evaluation(50)])
    model.save_model('D:/projects/Amazon ML/solution/lgb_global.txt')
    # tune thr
    probs=model.predict(Xva)
    pairs=[]; idx=0
    for qid in va:
        for cid,sc in all_scores.get(qid,[]):
            if cid not in cmap: continue
            pairs.append((qid,cid,probs[idx])); idx+=1
    for thr in [0.5,0.6,0.65,0.7,0.75,0.8,0.85]:
        pred={q:[] for q in va}
        for qid,cid,p in pairs:
            if p>=thr: pred[qid].append(cid)
        s=f05_macro(pred,{k:true_dict[k] for k in va})
        print('thr %.2f F05 %.4f P %.4f R %.4f' % (thr,s['F05'],s['P'],s['R']))
