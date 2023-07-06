import os
import glob
import pandas as pd
from csv import QUOTE_NONE
os.chdir("./")

extension = 'tsv'
all_filenames = ['train.tsv', 'dev.tsv']

#combine all files in the list
combined_tsv = pd.concat([pd.read_csv(f, sep='\t', quoting=QUOTE_NONE) for f in all_filenames], ignore_index=True)
combined_tsv['index'] = range(combined_tsv.shape[0])
#export to csv
combined_tsv.to_csv( "all.tsv",  sep='\t', index=False, index_label=False, encoding='utf-8-sig', quoting=QUOTE_NONE) 

data1=pd.read_csv("all.tsv", sep='\t', quoting=QUOTE_NONE )
data2=pd.read_csv("all.tsv", sep='\t', index_col=0, quoting=QUOTE_NONE )
print(data1)
