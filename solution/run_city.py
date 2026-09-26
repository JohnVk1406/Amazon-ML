import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, re
from src.normalize import basic_clean
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'

def extract_city(addr):
    if not addr or addr=='':
        return ''
    s = basic_clean(addr)
    if not s:
        return ''
    parts = [p.strip() for p in s.split(',') if p.strip()]
    # heuristic: city is second-last or last meaningful token group?
    # Simpler: take last 2 parts, first word of each? Let's just take full last-2 parts as key
    # For blocking, we want coarse: state/city token
    # Return last part + second-last part tokens
    if len(parts)>=2:
        # e.g. ['6 29 c i t colony 2nd main road mylapore chennai', 'tamil nadu'] -> city=chennai?
        # take last 2 tokens of first? Hmm.
        # Let's return last 3 tokens of whole address as key? No.
        # Better: return last part (state) + first token of second-last (city)
        return (parts[-1] + ' ' + parts[-2]).strip()
    return parts[-1] if parts else ''

# test on earlier examples
tests = [
 '6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu',
 '630 45th Terrace, Kansas City, MO',
 '85 Wayne Avenue, Ticonderoga, NY',
 '105 ELM ST, MORGANTON, NC',
 'KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi',
 '',
 'Near SBI ATM, Bhopal',
]
for t in tests:
    print(repr(t), '->', repr(extract_city(t)))

# check agreement on 200 pairs
s1map={}
for chunk in pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_address']):
    for eid,a in zip(chunk['entity_id'], chunk['business_address']):
        s1map[eid]=a
    if len(s1map)>300000:
        break
s23map={}
for f in ['train/train_source2.tsv','train/train_source3.tsv']:
    for chunk in pd.read_csv(f'{base}/{f}', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_address']):
        for eid,a in zip(chunk['entity_id'], chunk['business_address']):
            s23map[eid]=a
    print(f, len(s23map))
    if len(s23map)>600000:
        break
agree=0; total=0; empty_s1=0; empty_m=0
for chunk in pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=200000):
    for _,r in chunk.iterrows():
        s1=r['source1_entity_id']
        if s1 not in s1map:
            continue
        c1=extract_city(s1map[s1])
        if c1=='':
            empty_s1+=1
        v=r['matched_entity_ids']
        if v=='':
            continue
        for mid in v.split(','):
            if mid not in s23map:
                continue
            total+=1
            c2=extract_city(s23map[mid])
            if c2=='':
                empty_m+=1
            if c1!='' and c1==c2:
                agree+=1
    break
print('total pairs',total,'agree',agree,'rate',agree/max(1,total),'empty_s1',empty_s1,'empty_m',empty_m)
