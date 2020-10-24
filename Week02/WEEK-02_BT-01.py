import numpy as np 
import pandas as pd 

import matplotlib.pyplot as plt 
# %matplotlib inline
import seaborn as sns 
import os 
from sklearn.preprocessing import LabelEncoder

df = pd.read_csv("xAPI-Edu-Data.csv")

df.head(10)

df.info()

df.columns

df.rename(index = str, columns={'gender':'Gender',
                                'NationalITy':'Nationality',
                                'raisedhands':'RaisedHands',
                                'VisITedResources':'VisitedResources'},
                                inplace = True )
df.columns

print("Class Unique Values: ", df["Class"].unique())
print("Topic Unique Values : ", df["Topic"].unique())
print("StudentAbsenceDays Unique Values : ", df["StudentAbsenceDays"].unique())
print("ParentschoolSatisfaction Unique Values : ", df["ParentschoolSatisfaction"].unique())
print("Relation Unique Values : ", df["Relation"].unique())
print("SectionID Unique Values : ", df["SectionID"].unique())
print("Gender Unique Values : ", df["Gender"].unique())

for i in range (1,17):
  print(df.iloc[:,i].value_counts())
  print("*"*20)

sns.pairplot(df, hue = 'Class')

#Heat map
plt.figure(figsize= (14,14))
sns.heatmap(df.corr(), linewidths = 0.1, cmap = "YlGnBu", annot=True)

plt.figure(figsize=(20,14))
df.RaisedHands.value_counts().sort_index().plot.bar()

plt.figure(figsize=(10,10))
Raise_Hands= sns.boxplot(x = "Class", y = "RaisedHands", data = df)
plt.show()

Facegrid = sns.FacetGrid(df, hue = "Class")
Facegrid.map(sns.kdeplot, "RaisedHands", shade = True)
Facegrid.set(xlim = (0, df.RaisedHands.max()))

# data.groupby
df.groupby(['ParentschoolSatisfaction'])['Class'].value_counts()

pd.crosstab(df['Class'], df['ParentschoolSatisfaction'])

sns.countplot(x = "ParentschoolSatisfaction", data = df, hue = "Class")

# pie chart
labels = df.ParentschoolSatisfaction.value_counts()
colors = ["blue", "green"]
explode = [0,0]
sizes = df.ParentschoolSatisfaction.value_counts().values

plt.pie(sizes, explode=explode, labels = labels, colors = colors)
plt.show