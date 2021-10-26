import numpy as np
import pandas as pd
hts_df = pd.read_csv('htsdata.csv')


hts_df['HTS Number'] = hts_df['HTS Number'].replace({'NaN':np.nan})

hts_subset = hts_df[(hts_df['HTS Number'].astype(np.str).apply(len)<=4)&~hts_df['HTS Number'].isnull()].copy()

hts_subset.reset_index(drop=True,inplace=True)

def split_hts_codes(x):
    x_split = x.split('.')[:3]
    
    first_four = x_split[0]
    if len(first_four)<4:
        first_four = '0'+first_four
        
    second_two = '00'
    if len(x_split)>2:
        third_two = split[1]
        
    third_two = '00'
    if len(x_split)>2:
        third_two = x_split
        
    return first_four, second_two, third_two

hts_subset['first_four'],hts_subset['second_two'],hts_subset['third_two'] = zip(*hts_subset['HTS Number'].map(split_hts_codes))

hts_subset['HTS Number'] = hts_subset['HTS Number'].astype(np.str).apply(lambda x: '0'+x if len(str(x))<4 else x)

hts_subset['HTS Category'] = hts_subset['HTS Number'].apply(lambda x: str(x)[:2])


data = """01-05 Animal Products
06-14 Vegetable Products
15-15 Animal and Vegetable Fats and Oils
16-24 Foodstuffs, Beverages and Tobacco
25-27 Mineral Products
28-38 Chemicals &amp; Allied Industries
39-40 Plastics/Rubbers
41-43 Raw Hides, Skins, Leather, &amp; Furs
44-46 Wood &amp; Wood Products
47-49 Pulp of Wood and Fibrous Material
50-63 Textiles
64-67 Footwear/Headgear
68-70 Stone/Glass
71-71 Precious Stone, Metal, Pearls and Coins
72-83 Base Metals
84-85 Machinery/Electrical
86-89 Transportation
90-92 Precision Instruments
93-93 Arms and Ammunition
94-96 Miscellaneous Manufactured Articles
97-97 Works of Art
98-99 Unique US National HS Codes"""

cat_codes = {}
for line in data.split('\n'):
    start = int(line[:2])
    end = int(line[3:5])
    
    text = line[6:]
    
    if start==end:
        cat_codes[line[:2]]=text
        continue
        
    for i in range(start,end+1):
        i = str(i)
        if len(i)<2:
            i = '0'+i
        cat_codes[i] = i+' '+text
        
hts_subset['HTS Category Name'] = hts_subset['HTS Category'].replace(cat_codes)

hts_assc_words = {}

for unq in hts_subset['HTS Category Name'].unique():
    hts_assc_words[unq] = {}
    
    for i in hts_subset[hts_subset['HTS Category Name']==unq].index:
        desc = hts_subset.loc[i,'Description']
        cat = hts_subset.loc[i,'first_four']
        hts_assc_words[unq][cat+' '+desc] = []
    
import json
json.dump(hts_assc_words,open('hts_assc_cats.json','w'),indent=2)