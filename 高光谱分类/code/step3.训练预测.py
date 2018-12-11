# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 19:04:24 2018

@author: 李栋良
"""

from sklearn.svm import SVC
import pandas as pd
from sklearn.model_selection import train_test_split


x=pd.read_csv('pca-x.csv',index_col=0)
y=pd.read_csv('labels.csv',index_col=0)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

clf = SVC(gamma='auto',C=120)
clf.fit(X_train, y_train)
y_pred=clf.predict(X_test)
rightrate=sum(y_pred==y_test['200'])/1385


test=pd.read_csv('pca-test-x.csv',index_col=0)
Y_pred=clf.predict(test)