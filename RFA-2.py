# -*- coding: utf-8 -*-
"""
Created on Mon Feb  5 14:45:06 2024

@author: Admin
"""

import pandas as pd
from sklearn.datasets import load_iris
digits=load_iris()
dir(digits)

df=pd.DataFrame(digits.data)
df.head()

df['target']=digits.target
df[0:12]

x=df.drop('target',axis='columns')
y=df.target

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x, y,test_size=0.2)

from sklearn.ensemble import RandomForestClassifier
model=RandomForestClassifier(n_estimators=20)

#n_estimors : numbers of trees in the forest
model.fit(x_train,y_train)

model.score(x_test, y_test)
y_pre=model.predict(x_test)
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, y_pre)
cm

#Matplotlib inline
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Truth")
