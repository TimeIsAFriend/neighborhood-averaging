#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.neighbors import NearestNeighbors
def NA(score,X,k):
    # neighborhood average to produce the soomthed outlier scores
    # reference: Jiawei Yang, Susanto Rahardja, Pasi Fränti. Smoothing Outlier Scores Is All You Need to Improve Outlier Detectors. IEEE TKDE, 2023.
    nbrs = NearestNeighbors(n_neighbors=k, algorithm='ball_tree',metric='euclidean').fit(X) # by defualt, n_neighbors=100,algorithm='ball_tree',metric='euclidean'
    _, neighbor = nbrs.kneighbors(X)
    return np.array([np.mean(score[np.array(neighbor[[i],:k+1][0]).astype(int)]) for i in range(score.shape[0])])

# An example
import numpy as np
from sklearn.neighbors import LocalOutlierFactor as LOF

X=np.array([[1,1],[2,2],[1.5,1.5],[2,1],[1,2],[10,10]])
k=3
score=-LOF(n_neighbors=k).fit(X).negative_outlier_factor_
score_na=NA(score,X,k)

print('original outlier score by LOF:',np.round(score,2))
print('revised outlier score by NA:',np.round(score_na,2))

