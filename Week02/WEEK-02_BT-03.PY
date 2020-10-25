import pandas as pd 
import numpy as np 
from numpy import where
import matplotlib.pyplot as plt 
import seaborn as sns  
import plotly.express as px 
# process scaling libraries 
from numpy import mean
from sklearn.preprocessing import StandardScaler, RobustScaler

# train and test 

from sklearn.model_selection import train_test_split


# split data into n test and train set 

from sklearn.model_selection import StratifiedShuffleSplit
from imblearn.pipeline import Pipeline

# Oversample with SMOTE and random undersample for imbalanced dataset
from collections import Counter 
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

# model and evaluation 
from  sklearn.model_selection import cross_val_score  # evaluation model 
from  sklearn.model_selection import RepeatedStratifiedKFold # divide data into n sample 

from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix, classification_report

credit_card = pd.read_csv('creditcard.csv')
credit_card

"""# Read data"""

credit_card

"""# Data understanding with pandas functions"""

def first_rows() : 
   rows = credit_card.head(5)
   return rows 
print(first_rows())
print("Present the first few rows of the data")

"""##Present the structure of dataframe"""

def data_shape() : 
  shape = credit_card.shape
  return shape 
print(data_shape())
print("The data has 284087 rows and 31 columns")

"""## Describe the data with some statistical elements"""

def data_describe() : 
  describe = credit_card.describe(include = np.number)
  return describe
print(data_describe())
print("Please take a look at the table statistics and having clear observation about those numbers")
print("Please noticed due to the length of features, the table is not able to show up all statistical of features")
print("In order to be easily understandable the table. You must require have a little domain knowledge about the bank industry")
print("In order to understand the how PCA works","https://builtin.com/data-science/step-step-explanation-principal-component-analysis")

"""## Check data type"""

def data_type()  : 
  type = credit_card.dtypes
  return type
print(data_type())
print("The data has only hold numerical variables")

"""## Checking for missing values"""

def data_infor() :
  information = credit_card.info()
  return information 
print(data_infor())
print("The data has not missing values or NAN values")

"""## Class Distribution with histogram"""

def class_count() : 
   count = credit_card.Class.value_counts()
   positive_percent = (((credit_card['Class'] == 1).sum()/(len(credit_card.Class)))*100.0)
   plot = credit_card.hist(color = 'blue', figsize = (20,5), column = ['Class'],xlabelsize= 100, ylabelsize= 10)
   return count,positive_percent,plot
print(class_count())
print("The class feature illustrates highly inbalanced values. The positive class 1 (frauds) accounted for 0.172% of all transactions ")
print("It can see clearly seen that the distribution of the class extremely imbalanced. It can be hardly to see the distribution of non-frauds from the plot")
print("The histogram is the best choice to show the frequency of variables immediately apparent, but it is not a good choice to compare two variables")

"""## Distribution of time and amount"""

def visualize_time_amount() : 
  # create two subplots 
  fig,ax = plt.subplots(nrows = 1,ncols = 2,figsize = (18,4))
  # declare variables 
  Time  = credit_card["Time"].values 
  Amount  = credit_card["Amount"].values
  sns.distplot(Time,bins = 10, ax = ax[0], color = 'r')
  ax[1].set_title("Distribution of time")
  ax[1].set_xlim([min(Time),max(Time)])
  # create for amount 
  sns.distplot(Amount,bins = 10, ax =ax[1], color = 'b')
  ax[1].set_title("Distribution of amount")
  # used to set the x-axis view limits 
  ax[1].set_xlim([min(Amount),max(Amount)])
print(visualize_time_amount())
print("It can be clearly see that the red figure shows up the distribution on Time transactions")
print("The second figure does not represent clear view for the distribution. It can hardly to extract insights from the blue figure ")
print("Having aim to comprehensive understand both figures")

"""## Data Preprocessing :

### Data scaling
"""

rob = RobustScaler() 
credit_card['Time']= rob.fit_transform(credit_card["Time"].values.reshape(-1,1))
credit_card['Amount'] = rob.fit_transform(credit_card["Amount"].values.reshape(-1,1))
credit_card.iloc[500:600,:]

Copy = credit_card.copy()
Copy

"""### Drawling the correlation between time and amout features"""

scat = px.scatter(Copy, x = 'Time', y = 'Amount', marginal_x = 'box', marginal_y = 'rug')
scat.show()

"""# X_train, y_train"""

X = credit_card.iloc[:,:-1]
y = credit_card.iloc[:,-1]

print(X.head())

print(y.head())
