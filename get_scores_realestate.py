# -*- coding: utf-8 -*-
"""
Created on Fri Mar 17 20:12:01 2017

@author: SzMike
"""
import matplotlib.pyplot as plt
import skimage.io as io
import file_helper as fh
import os
import numpy as np
import json
import pandas as pd

from sklearn import cluster

import aes_Picturio
import aes_AADB

base_dir=os.path.join('e:','OneDrive','AES','Photo_DB')
image_dir=os.path.join(base_dir,'RealEstate')

save_dir=os.path.join(r'D:\DATA\RealEstate\AES')

score_file_merged=os.path.join(image_dir,'scores_merged.csv')

"""
Scoring model - manual scores
"""
image_list=fh.imagelist_in_depth(image_dir,level=1)

"""
Class names from folder names
"""
class_names=[os.path.dirname(f).split('\\')[-1] for f in image_list]
df_db = pd.DataFrame(data={'Filename':image_list,'Class name':class_names})

df_scores=df_db.copy()
"""
Scoring model - AADB
"""
scoring=aes_AADB.scoring()
image_all_scores=scoring.get_scores(df_db['Filename'].values)

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
image_all_scores=scoring.get_scores(df_db['Filename'].values)

scores=[np.NaN]*len(df_db['Filename'].values)
for i,sc in enumerate(image_all_scores):
    scores[i]=sc['AestheticScore']
        
df_scores['Picturio']=scores
df_scores.to_csv(r'scores.csv',index=None)

df_scores=pd.read_csv(r'scores.csv',delimiter=',')

"""
Save scores
"""


df_scores.to_csv(score_file_merged,index=None)


"""
Statsistics
"""

df_scores=pd.read_csv(score_file_merged,delimiter=',')

h=df_scores.groupby('Class name').hist()


scores=df_scores['AADB'].values
sep_value=0.5


#
#df_scores.corr(method='pearson')
#
#from pandas.tools.plotting import scatter_matrix
#scatter_matrix(df_scores, alpha=0.2, figsize=(6, 6), diagonal='kde')
#
#pd.options.display.mpl_style = 'default'
#df_scores.boxplot()

df_scores.hist()

df_top=df_scores[(df_scores['AADB']>0.8*df_scores['AADB'].max())]
df_top=df_scores[df_scores['AADB']<=0.4]

df_top.groupby('Class name').count()
for i, row in df_top.iterrows():    
    print(row['Filename'])
    img=io.imread(row['Filename'])
    fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(8, 3))
    fig.suptitle('AADB:'+str(row['AADB'])+'     Picturio:'+str(row['Picturio'])+'      CAT:'+row['Class name'])
    ax1.imshow(img)
    ax1.grid(False)

    fig.savefig(os.path.join(save_dir,'bottom',row['Class name']+'_'+os.path.basename(row['Filename'])))
    plt.close('all')
    
    
