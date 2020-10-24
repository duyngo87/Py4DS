import numpy as np 
import pandas as pd 
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from sklearn.model_selection import cross_val_score

df = pd.read_csv("mushrooms.csv")

df.head(10)

y=df['class']
X=df.drop('class',axis=1)
y.head()

"""#Using LabelEncoder"""

#Check type of data
df.dtypes

labelencoder = LabelEncoder()
for cl in df.columns:
  df[cl] = labelencoder.fit_transform(df[cl])

#Check once again colunms dataset 
df.dtypes

df.head(10)

#slipt the data
X=df.drop(['class'], axis=1)
Y=df['class']
x_train, x_test,y_train,y_test = train_test_split(X,Y, test_size = 0.1)

"""# Using Model"""

clf = DecisionTreeClassifier(criterion= 'entropy')

#Fit decision tree
clf = clf.fit(x_train,y_train)

y_pred = clf.predict(x_test)

print("CRAT Accuracy: ",metrics.accuracy_score(y_test,y_pred))

from sklearn.svm import SVC
clf = SVC()
clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print("SVM Accuracy: ", metrics.accuracy_score(y_test, y_pred) )

from sklearn.linear_model import  LogisticRegression
lr = LogisticRegression(solver='liblinear')
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

print("LR Accuracy: ", metrics.accuracy_score(y_test, y_pred))