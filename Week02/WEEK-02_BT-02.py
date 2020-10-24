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

df = pd.read_csv('xAPI-Edu-Data.csv')

df.head(10)

df.info()

df.describe()

df.columns

df.rename(index=str, columns={'gender':'Gender', 
                              'NationalITy':'Nationality',
                              'raisedhands':'RaisedHands',
                              'VisITedResources':'VisitedResources'},
                               inplace=True)
df.columns

df.nunique()

print("Class Unique Values : ", df["Class"].unique())
print("Topic Unique Values : ", df["Topic"].unique())
print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())
print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())
print("Relation Unique Values : ", df["Relation"].unique())
print("SectionID Unique Values : ", df["SectionID"].unique())
print("Gender Unique Values : ", df["Gender"].unique())

#Check the type of each of colunms
df.dtypes

#Relationshio annalysis
corelation = df.corr()

sns.heatmap(corelation, xticklabels= corelation.columns, yticklabels= corelation.columns, annot = True)

sns.pairplot(df, hue = 'StageID')

sns.relplot(x = 'RaisedHands', y = 'StageID', hue= 'Gender', data = df )

sns.distplot(df['RaisedHands'], bins= 5)

plt.figure(figsize=(10,10))
Raise_Hands= sns.boxplot(x = "StageID", y = "RaisedHands", data = df)
plt.show()

Facegrid = sns.FacetGrid(df, hue = "StageID")
Facegrid.map(sns.kdeplot, "RaisedHands", shade = True)
Facegrid.set(xlim = (0, df.RaisedHands.max()))

df.groupby(['Nationality'])['StageID'].value_counts()

pd.crosstab(df['StageID'], df['Nationality'])

sns.countplot(x = "StageID", data = df, hue = "Gender")