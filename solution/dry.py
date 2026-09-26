"""Dry run on 1000 test S1 per country."""
import os, sys
sys.path.insert(0, 'D:/projects/Amazon ML/solution')
import pandas as pd
BASE='D:/projects/Amazon ML/student_resource/student_resource/dataset'
# sample 500 S1 per country
s1=pd.read_csv(f'{BASE}/test/test_source1.tsv', sep='\t', dtype=str, keep_default_na=False, nrows=200000, usecols=['entity_id','business_name','business_address','country'])
print(s1['country'].value_counts().to_dict())
for co in s1['country'].unique():
    sub=s1[s1['country']==co].head(300)
    print(co, len(sub))
