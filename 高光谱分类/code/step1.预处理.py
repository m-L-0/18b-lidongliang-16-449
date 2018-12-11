# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 16:41:42 2018

@author: 李栋良
"""
import scipy.io as scio
import numpy as np
import pandas as pd
data = scio.loadmat('data2_train.mat')
test = scio.loadmat('data_test_final.mat')
x=data['data2_train']
X=test['data_test_final']
y=np.zeros([x.shape[0],1],dtype=np.uint16)
y[y==0]=2
Data=np.concatenate([x,y],axis=1)

for i in [3,5,6,8,10,11,12,14]:
    data = scio.loadmat('data'+str(i)+'_train.mat')
    x=data['data'+str(i)+'_train']
    y=np.zeros([x.shape[0],1],dtype=np.uint16)
    y[y==0]=i
    data=np.concatenate([x,y],axis=1)
    Data=np.concatenate([Data,data],axis=0)


D=pd.DataFrame(Data)
X=pd.DataFrame(X)
data=D.iloc[:,200]
D=D.drop(columns=200)

df_norm = (D - D.min()) / (D.max() - D.min())
tf_norm = (X - X.min()) / (X.max() - X.min())