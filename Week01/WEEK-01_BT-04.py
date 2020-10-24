import numpy as np 
import pandas as pd 
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib.pyplot as  plt 
import seaborn as sns 
import os 

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import cross_val_score

df = pd.read_csv("xAPI-Edu-Data.csv")
df.head(10)

df.info()

df.describe()

#Check each colunms
df.columns

df.rename(index = str, columns={'gender':'Gender', 
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                                inplace = True)

#Check once again the name of colunms
df.columns

#Splitting Data
X = df.drop('Class', axis = 1)
y = df['Class']

#Check the type of each of colunms
df.dtypes

#Using LabelEncoder
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
Oj_Colunms = X.dtypes.pipe(lambda X: X[ X== 'object']).index
for colunms in Oj_Colunms:
  X[colunms] = labelencoder.fit_transform(X[colunms])

#Check once again dataset
X.dtypes

X.head(10)

#Splitting Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.33, random_state= 42)

"""#Using Model"""

from sklearn.svm import SVC

svm=SVC()
svm.fit(X_train,y_train)

y_pred=svm.predict(X_test)

print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_pred))

from sklearn.linear_model import LogisticRegression

lr=LogisticRegression(solver='liblinear')
lr.fit(X_train,y_train)

y_pred=lr.predict(X_test)

print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_pred))