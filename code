import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing,cross_validation
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
x=pd.read_csv("/Users/xor/Downloads/data.csv",header=0)
x2=pd.get_dummies(x,columns=["diagnosis"])
x2['diag']=x2['diagnosis_M']
x2.drop('diagnosis_B',axis=1,inplace=True)
x2.drop('diagnosis_M',axis=1,inplace=True)
x2.drop('Unnamed: 32',axis=1,inplace=True)
x2.drop('id',axis=1,inplace=True)
y=x2['diag']
x2.drop('diag',axis=1,inplace=True)
trx2,tsx2,ytr,yts=cross_validation.train_test_split(x2,y,test_size = 0.2)
lr=LogisticRegression()
lr.fit(trx2,ytr)
print(lr.score(tsx2,yts))
n= KNeighborsClassifier(n_neighbors=3)
n.fit(trx2,ytr) 
print(n.score(tsx2,yts))
