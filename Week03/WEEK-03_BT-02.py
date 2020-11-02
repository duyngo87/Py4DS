#!/usr/bin/env python
# coding: utf-8

# # Import required libraries 
# 
# 1.   Élément de liste
# 2.   Élément de liste
# 
# 

# In[301]:


import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import itertools
from numpy import mean,std 
import missingno as msno 
import seaborn as sns 
import plotly.express as px 
# preprocessing 
from sklearn.preprocessing import LabelEncoder, RobustScaler 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
## building model to dectect extreme values or well values out of normal range 
### dectect extreme 
from sklearn.cluster import KMeans
import keras 
import tensorflow as tf 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
import sklearn.metrics as sm
from sklearn.metrics import mean_absolute_error
import math
# train and test split 

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
# normalization 
from sklearn.preprocessing import RobustScaler, StandardScaler

# model 
import xgboost as xg 
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
# evaluate model 
from sklearn.metrics import auc, accuracy_score, confusion_matrix, mean_squared_error 
from sklearn.model_selection import cross_val_score


# In[202]:


AB = pd.read_csv("AB_NYC_2019_1.csv")
AB.tail()


# In[203]:


AB.neighbourhood_group.unique()


# In[204]:


categorical = ['Bronx','Queens','Staten Island','Brooklyn','Manhattan']


# In[205]:


encod = OrdinalEncoder(categories= [categorical])


# In[206]:


encod.fit(AB[['neighbourhood_group']])


# In[207]:


AB['neighbourhood_group'] = pd.DataFrame(encod.transform(AB[['neighbourhood_group']]))


# In[208]:


AB['neighbourhood_group'].iloc[:30605]


# # Data Understanding 

# In[209]:


AB.neighbourhood_group.head()


# In[210]:


def count_missing_values() :  
      null = AB.isnull().sum() 
      info  = AB.info()
      return null ,info
print(count_missing_values() )
print("The data has 7 categorical  features and the remaing are numerical values")
print(" From the table below, we can see that there are missing values in features Host_id, neighbourhood_group,reviews_per_month,last_review. We should consider on them")


# In[211]:


def visualize_mssing_values() : 
  visualize = msno.bar(AB,color = 'blue') 
  heat = msno.heatmap(AB)
  return visualize , heat

print(visualize_mssing_values())
print("Visualizing missing values with bar char")


# In[212]:


def distribution() : 
  hist = AB.hist(figsize = (10,10)) 
  return hist 
print(distribution())


# ## Visualizing the mean price and std groupby neighbourhood 

# In[213]:


def mean_price() : 
  figure = plt.figure()
  ax = figure.add_subplot()
  ax2 = ax.twinx()
  width = 0.4
  price = AB.groupby('room_type').mean()['price'].plot(kind = 'bar', color = 'blue', ax = ax, position = 1, width = width)
  std = AB.groupby('room_type').std()['price'].plot(kind = 'bar', color = 'red', ax = ax2, position = 0, width = 0.3)
  ax.set_ylabel('mean')
  ax2.set_ylabel('std')
  plt.show() 
print(mean_price())
print("The blue column represents for central tendency distribution")
print("The red column represents for the spread of scores")


# ## Visualing the mean and standard deviation by group by two columns 

# In[214]:


def visualization() : 
  figure = plt.figure()
  ax = figure.add_subplot() 
  ax2 = ax.twinx
  width = 0.4
  mean = AB.groupby(['neighbourhood_group','room_type']).mean()['price'].plot(kind = 'bar',position = 1, ax = ax, width = width, figsize = (15,7), color = 'green', title = 'Mean Distribution')
  #std =  AB.groupby(['neighbourhood_group','room_type']).std()['price'].plot(kind = 'bar', position = 0, ax = ax2, width = width) 
  ax.set_ylabel('mean')
  #ax2.set_ylabel('std')
  plt.show()
  
print(visualization())


# In[214]:





# ### Average price by room type 

# In[215]:


correlation = AB.corr(method = 'pearson')
sns.heatmap(correlation, cmap = 'YlGnBu')
correlation


# In[215]:





# ## Detect outliers 

# In[215]:





# In[216]:


def availability() : 
  availability = px.box(AB, y = "availability_365", title = "Availability Outliers") 
  availability.show()

print(availability())


# In[217]:


def price() : 
  price = px.box(AB, y = "price", title = "price") 
  price.show()

print(price())


# In[218]:


def price_1() : 
  box = sns.boxplot(x = AB['price'])
  return box
print(price_1())


# In[219]:


def minimum_nights() : 
  fig = px.box(AB, y = "minimum_nights", title = "Outliers") 
  fig.show() 

print(minimum_nights())


# In[220]:


def calculated_host_listings () : 
  listings = px.box(AB, y ='calculated_host_listings_count', title = 'Listings Outliers')
  return listings.show()
print(calculated_host_listings())


# ### lets figure our extreme values out with statistical table 

# In[221]:


def describe() : 
  des = AB.groupby('neighbourhood_group')[['price']].describe()
  pd.set_option('display.max_columns', None)
  return des 

print(describe())
print("From the statistical table below. It can be clearly seen that there are a lot extreme values in the price and minimum_nights")
print("The ")


# In[221]:





# In[222]:


def describe_1() : 
  des_1 = AB[['availability_365']].describe()
  pd.set_option('display.max_columns',None)
  return des_1
print(describe_1())


# In[223]:


def describe_1() : 
  des_1 = AB[['calculated_host_listings_count']].describe()
  pd.set_option('display.max_columns',None)
  return des_1
print(describe_1())


# ### I would go for conclusion that our data has extreme values. My job is now remove them out of data before doing further analysis

# ### lets say I would do with give latitude and longitude 

# In[224]:


import urllib
#initializing the figure size
plt.figure(figsize=(10,8))
#loading the png NYC image found on Google and saving to my local folder along with the project
i=urllib.request.urlopen('https://upload.wikimedia.org/wikipedia/commons/e/ec/Neighbourhoods_New_York_City_Map.PNG')
nyc_img=plt.imread(i)
#scaling the image based on the latitude and longitude max and mins for proper output
plt.imshow(nyc_img,zorder=0,extent=[-74.258, -73.7, 40.49,40.92])
ax=plt.gca()
#using scatterplot again
AB[AB.price < 500].plot(kind='scatter', x='longitude', y='latitude', label='availability_365', c='price', ax=ax, 
           cmap=plt.get_cmap('jet'), colorbar=True, alpha=0.4, zorder=5)

plt.legend()
plt.show()


# ### Categorical encoding ordinal 

# In[224]:





# # Building model to dectect outliers before going futher analysis 

# ### My objective : In this project, I am going to build a neuron network to detect outliers. This is an unspervised learning in machine learning algorithms. Autoencoders is an unspervised learning in machine learning with non-label data 

# ### Data preprocessing 

# ### Create a subdataframe 

# In[225]:


Sub_dataframe = AB[['price','calculated_host_listings_count','availability_365','minimum_nights']]
Sub_dataframe.sort_values(by = ['price'], ascending = (True))
pd.set_option('display.max_columns',None)
Sub_dataframe.shape


# In[226]:


X = Sub_dataframe.iloc[:,[0,1,2,3]].values


# In[227]:


print(X)


# ### Using the eblow method to find the optimal number of clusters 

# In[228]:


WCSS = []
for i in range(1,101) : 
  kmeans = KMeans(n_clusters = i,init='k-means++' , random_state = 42) 
  kmeans.fit(X)
  WCSS.append(kmeans.inertia_)

plt.plot(range(1,101),WCSS)
plt.title("ABC")
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()


# In[229]:


kmeans = KMeans(n_clusters = 10,init='k-means++' , random_state = 42) 
y_kmeans = kmeans.fit_predict(X) 
y_kmeans


# In[230]:


print(y_kmeans)
pd.set_option('display.max_rows', None)
data = ({"cluster":y_kmeans})
df = pd.DataFrame(data)
df.to_csv('file.csv')


# In[231]:


plt.scatter(X[y_kmeans == 0,0],X[y_kmeans == 0,1],X[y_kmeans == 0,2],X[y_kmeans == 0,3], label ='Cluster')
plt.scatter(X[y_kmeans == 1,0],X[y_kmeans == 1,1],X[y_kmeans == 1,2],X[y_kmeans == 1,3], label ='Cluster2')
plt.scatter(X[y_kmeans == 2,0],X[y_kmeans == 2,1],X[y_kmeans == 2,2],X[y_kmeans == 2,3], label ='Cluster3')
plt.scatter(X[y_kmeans == 3,0],X[y_kmeans == 3,1],X[y_kmeans == 3,2],X[y_kmeans == 3,3], label ='Cluster4')
plt.scatter(X[y_kmeans == 4,0],X[y_kmeans == 4,1],X[y_kmeans == 4,2],X[y_kmeans == 4,3], label ='Cluster5')
plt.scatter(X[y_kmeans == 5,0],X[y_kmeans == 5,1],X[y_kmeans == 5,2],X[y_kmeans == 5,3], label ='Cluster6')
plt.scatter(X[y_kmeans == 7,0],X[y_kmeans == 7,1],X[y_kmeans == 7,2],X[y_kmeans == 7,3], label ='Cluster7')
plt.scatter(X[y_kmeans == 8,0],X[y_kmeans == 8,1],X[y_kmeans == 8,2],X[y_kmeans == 8,3], label ='Cluster9')
plt.scatter(X[y_kmeans == 6,0],X[y_kmeans == 6,1],X[y_kmeans == 6,2],X[y_kmeans == 6,3], label ='Cluster9')
plt.scatter(X[y_kmeans == 9,0],X[y_kmeans == 9,1],X[y_kmeans == 9,2],X[y_kmeans == 9,3], label ='Cluster10')
plt.scatter(kmeans.cluster_centers_ [:,0],kmeans.cluster_centers_[:,1],kmeans.cluster_centers_[:,2],kmeans.cluster_centers_[:,3],label = 'center')
plt.legend()


# In[232]:


X_1 = X[y_kmeans == 0]
X_2 = X[y_kmeans == 1 ]
X_3 = X[y_kmeans == 2]
X_4 = X[y_kmeans == 3 ] 
X_5 = X[y_kmeans == 4 ]
X_6 = X[y_kmeans == 5 ]
X_7 = X[y_kmeans == 6]
X_8 = X[y_kmeans == 7 ]


# In[253]:


df_1 = pd.DataFrame(X_1)
df_2 = pd.DataFrame(X_2)
df_3  = pd.DataFrame(X_3)
df_4 = pd.DataFrame(X_4)
df_5 = AB['neighbourhood_group'].iloc[:19121]
df_6 = pd.DataFrame(X_5)
df_7 = pd.DataFrame(X_6)
df_8 = pd.DataFrame(X_7)
df_9 = pd.DataFrame(X_8)


# In[322]:


df = pd.concat([df_1,df_2,df_3,df_4,df_6,df_7,df_8,df_9])
df.rename(columns = {0:"price",1:"calculated_host_listings_count",2:"availability_365",3:"minimum_nights",4:"neighbourhood_group"} )
df.shape
df.insert(4,"df_5",df_5)
df.head()
df.shape
df.head()


# In[283]:


X = df[[1,2,3,"df_5"]]
y = df.iloc[:,0]


# In[284]:


X.head()


# In[285]:


y.head()


# ## Split into training and testing 

# In[286]:


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33, random_state = 43, shuffle = True) 


# In[321]:


X_train.head()


# In[323]:


y_train.head()


# In[289]:


y_test.shape


# In[290]:


std = StandardScaler() 
std.fit_transform(X_train,y_train)


# Starting import model and predict the result 

# In[291]:


xgb_r = xg.XGBRegressor(objective ='reg:linear', 
                  n_estimators = 10, seed = 123)


# In[292]:


xgb_r.fit(X_train,y_train)


# In[293]:


pred = xgb_r.predict(X_test)
pd.DataFrame(pred).head()


# In[294]:


rmse = (mean_squared_error(y_test, pred)) 
print((rmse)) 


# In[295]:


print(math.sqrt(mean_squared_error(y_test, pred)))


# In[296]:


print(mean_absolute_error(y_test, pred))


# In[297]:


print("R2 score =", round(sm.r2_score(y_test,pred), 2))


# In[298]:


print("Explain variance score =", round(sm.explained_variance_score(y_test,pred), 2)) 


# In[299]:


a = pd.DataFrame({"pred":pred, "test":y_test})
a.to_csv("file_2.csv")


# ## Decision Tree

# In[303]:


decision = DecisionTreeRegressor() 

decision.fit(X_train,y_train)


# In[309]:


pred_2 = decision.predict(X_test)
pd.DataFrame(pred_2).head()


# In[316]:


print(mean_absolute_error(y_test, pred_2))


# In[320]:


print("Explain variance score =", round(sm.explained_variance_score(y_test,pred_2),2))


# In[311]:


print("R2 score =", round(sm.r2_score(y_test,pred_2), 2))


# In[ ]:





# ### Randomforest 

# In[313]:


Random = RandomForestRegressor() 
Random.fit(X_train,y_train)


# In[314]:


pred_3 = Random.predict(X_test)


# In[315]:


print("R2 score =", round(sm.r2_score(y_test,pred_3), 2))


# In[ ]:




