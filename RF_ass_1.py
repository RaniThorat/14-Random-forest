# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:23:33 2024

@author: Admin
"""



import pandas as pd
company=pd.read_csv('Company_Data.csv')
dir(company)

df=pd.DataFrame(company)
df.head()

df['Sales']=company.Sales
df[0:12]

X=df.drop('Sales',axis='columns')
y=df.Sales

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)
#n_estimator:number of trees in the forest
model.fit(X_train,y_train)

model.score(X_test,y_test)
y_predicted=model.predict(X_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_predicted)
cm

#matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(10,7))
sn.heatmap(cm,annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')