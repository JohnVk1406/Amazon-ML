import sys; sys.path.insert(0,'D:/projects/Amazon ML/solution')
import pandas as pd, re
from src.normalize import basic_clean
base='D:/projects/Amazon ML/student_resource/student_resource/dataset'

def extract_city_state(addr):
    if not addr or addr=='' or str(addr).lower()=='nan':
        return '', ''
    raw = str(addr)
    if 'null' in raw.lower():
        raw = re.sub(r'\bnull\b', '', raw, flags=re.I)
    parts = [p.strip() for p in raw.split(',') if p.strip()]
    if not parts:
        return '', ''
    # last part = state (+pin), second-last = city
    state_raw = parts[-1]
    city_raw = parts[-2] if len(parts)>=2 else ''
    # clean
    state = basic_clean(state_raw)
    city = basic_clean(city_raw)
    # city may contain street: take last 1-2 tokens as city? e.g. '2nd main road mylapore chennai' -> 'mylapore chennai' or 'chennai'?
    # take last 2 tokens
    ct = city.split()
    if len(ct) > 2:
        city = ' '.join(ct[-2:])
    st = state.split()
    # state may include pin: take first 1-2 tokens, drop digits
    st = [t for t in st if not t.isdigit()]
    state = ' '.join(st[:2])
    return city, state

tests = [
 '6(29), C.I.T. Colony, 2Nd Main Road Mylapore, Chennai, Tamil Nadu',
 '630 45th Terrace, Kansas City, MO',
 '85 Wayne Avenue, Ticonderoga, NY',
 '105 ELM ST, MORGANTON, NC',
 'KH NO. -570/13, NEW DELHI, WEST DELHI, Delhi',
 '45ND TERRACE, null, KANSAS CITY, MO',
 'KANSAS CITY, MO, 630 45ND TERRACE, null',
 '',
 'Near SBI ATM, Bhopal',
 '6(29), C.i.t. Colony, 2Nd Main Road Mylapore, Chennai, TN',
]
for t in tests:
    print(repr(t)[:70], '->', extract_city_state(t))

s1map={}
for chunk in pd.read_csv(f'{base}/train/train_source1.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_address']):
    for eid,a in zip(chunk['entity_id'], chunk['business_address']):
        s1map[eid]=a
    if len(s1map)>300000:
        break
s23map={}
for f in ['train/train_source2.tsv']:
    for chunk in pd.read_csv(f'{base}/{f}', sep='\t', dtype=str, keep_default_na=False, chunksize=500000, usecols=['entity_id','business_address']):
        for eid,a in zip(chunk['entity_id'], chunk['business_address']):
            s23map[eid]=a
    print('s23', len(s23map))
    break
agree_c=0; agree_s=0; agree_both=0; total=0
for chunk in pd.read_csv(f'{base}/train/train_ground_truth.tsv', sep='\t', dtype=str, keep_default_na=False, chunksize=200000):
    for _,r in chunk.iterrows():
        s1=r['source1_entity_id']
        if s1 not in s1map:
            continue
        c1,s1t=extract_city_state(s1map[s1])
        v=r['matched_entity_ids']
        if v=='':
            continue
        for mid in v.split(','):
            if not mid.startswith('S2-'):
                continue
            if mid not in s23map:
                continue
            total+=1
            c2,s2t=extract_city_state(s23map[mid])
            if c1!='' and c1==c2:
                agree_c+=1
            if s1t!='' and s1t==s2t:
                agree_s+=1
            if c1!='' and c1==c2 and s1t!='' and s1t==s2t:
                agree_both+=1
            if total<5:
                print(s1map[s1], '|', s23map[mid], '|', (c1,s1t),(c2,s2t))
    break
print('total',total,'city agree',agree_c/total,'state agree',agree_s/total,'both',agree_both/total)
