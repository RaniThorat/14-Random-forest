# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:28:27 2024

@author: Admin
"""

'''
Divide the diabetes data into train and test datasets
and build a Random Forest and Decision Tree model with 
Outcome as the output variable. 

Business Contraints:
Interpretability: The models need to be interpretable so 
that the company can understand which factors contribute 
most to high sales.

Scalability: The models should be scalable and able to 
handle a large amount of data as the company expands its
 operations.

Accuracy: While interpretability is important, the models 
should also be accurate in predicting sales to a reasonable
 degree.

Resource Constraints: The company may have limitations on 
computational resources, so we should consider model 
complexity and computational efficiency.
Maximize:
The primary objective is to maximize sales volume. 
This means aiming for higher revenues through increased 
product sales.

Minimize:
Minimize the cost associated with misclassification.
 Misclassification can lead to inefficient resource
 allocation and suboptimal decision-making. By minimizing
 misclassification, the company can reduce potential losses
 and improve overall operational efficiency.
'''


import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv("Diabetes.csv")
data.head()

df=pd.DataFrame(data)
df.head()

# Features and target variable
features = df.drop(df.columns[-1], axis=1)
target = df[df.columns[-1]]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Create and train the Random Forest model
from sklearn.ensemble import RandomForestClassifier
model= RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test, predictions)
cm

import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['NO', 'YES'], yticklabels=['NO', 'YES'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()