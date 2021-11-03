import pandas as pd
import json
import glob
import os
import numpy as np

def run(file):
    lis = []
    for i in range(len(file)):
        f = file[i]
        read = json.load(open(f))['cluster']
        read['file'] = f
        lis.append(read)
    return lis

'''
"chem_struct":"N/A OR STRUCT", "compound": "NAME", "PubChem ID": "ID", "npAtlas ID": "ID", "ncbi_tax_id": "ID",  "organism_name": "NAME", "Which Entry": "Name of Json file"
'''
f_name = glob.glob('mibig_json_2.0/*.json')  # list
data = run(f_name)  # list of dicts
# print(len(f_name))
# print(len(data))
'''
items = []
for x in data:
    for k in list(x.keys()):
        if k not in items:
            items.append(k)
#       temp_list.append(k)
for x in data:
    for y in items:
        if not y in x:
            x[y] = 'N/A'

cls = pd.DataFrame(data)
cls.to_csv('output_by_panda_dir.csv')
'''