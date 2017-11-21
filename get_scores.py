# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:12:01 2017

@author: SzMike
"""
import matplotlib.pyplot as plt
import skimage.io as io
import file_helper
import os
import json
import pandas as pd
import aes_Picturio
import aes_AADB
import numpy as np

base_dir=r'c:\Users\szmike\OneDrive\AES\Photo_DB\Annotator_Processed'
image_dir=os.path.join(base_dir,'images')
#image_dir=r'C:\Users\szmike\Documents\Projects\AES\images'

score_file=os.path.join(base_dir,'ann_scores20160822_processed.csv')
score_file_merged=os.path.join(base_dir,'scores_merged.csv')

"""
Scoring model - manual scores
"""
df_manual=pd.read_csv(score_file,delimiter=';')

image_files = [os.path.join(image_dir,f) for f in df_manual['imID']]
#image_files=file_helper.imagelist_in_depth(image_dir,level=0)


df_scores=pd.DataFrame(data=image_files,columns=['image_files'])

"""
Scoring model - AADB
"""
scoring=aes_AADB.scoring()
image_all_scores=scoring.get_scores(image_files[0:2])

#scores = [image_all_scores[i]['AestheticScore'] for i in range(len(image_all_scores)) if image_all_scores[i]['AestheticScore']!= 'None']
scores = [image_all_scores[i]['AestheticScore'] for i in range(len(image_all_scores))]
for i,s in enumerate(scores):
    if s=='None':
        scores[i]=np.NaN
    else:
        scores[i]=float(s)
        
df_scores['AADB']=scores

"""
Scoring model - Picturio
"""
scoring=aes_Picturio.scoring()
image_all_scores=scoring.get_scores(image_files)

scores=[np.NaN]*len(image_files)
for i,sc in enumerate(image_all_scores):
    fn=os.path.join(image_dir,sc['Photo'])
    ind=df_scores.index[df_scores['image_files']==fn].tolist()
    scores[ind[0]]=sc['AestheticScore']
        
df_scores['Picturio']=scores
df_scores.to_csv(r'scores.csv',index=None)

df_scores=pd.read_csv(r'scores.csv',delimiter=',')

"""
Scoring model - add manual scores
"""

df_scores['AES']=df_manual['Aesthetic'].values

df_scores.to_csv(score_file_merged,index=None)
df_scores=pd.read_csv(score_file_merged,delimiter=',')


"""
Statsistics
"""

df_scores=pd.read_csv(score_file_merged,delimiter=',')

df_scores.corr(method='pearson')

from pandas.tools.plotting import scatter_matrix
scatter_matrix(df_scores, alpha=0.2, figsize=(6, 6), diagonal='kde')

pd.options.display.mpl_style = 'default'
df_scores.boxplot()

df_scores.hist()

df_top=df_scores[(df_scores['Picturio']==df_scores['Picturio'].max()) & (df_scores['AES']>0)]

for i in range(len(df_top)):
    print(df_top['image_files'].values[i])
    img=io.imread(df_top['image_files'].values[i])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    fig.suptitle('AES: '+str(df_top['AES'].values[i])+'     AADB:'+str(df_top['AADB'].values[i])+'     Picturio:'+str(df_top['Picturio'].values[i]))
    ax1.imshow(img)