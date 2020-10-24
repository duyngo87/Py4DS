import numpy as np #tinh toan, dang mang
import pandas as pd #tai du lieu len 
import scipy as sp 
import matplotlib as mpl #hien thi file 
import seaborn as sns # 
import matplotlib.pyplot as plt 

#Path of dataset in your gg drive 
path = 'spam.csv'

#Read in the data into a pandas dataframe
dataset_pd = pd.read_csv(path)

#Read in the data into a numpy array
dataset_np = np.genfromtxt(path, delimiter= ',')

#Separete between feature (X) and (Y)
X = dataset_np[:, :-1]
Y = dataset_np[:, -1]

#Print
print(X.shape)
print(Y.shape)

print(X[0:5,:])
print(Y[0: 5]) #5 frist value of y 
print(Y[-5:]) #5 last value of y 

#Split the train and test sets
from sklearn.model_selection import train_test_split

#Split dataset
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size= 0.33, random_state= 42)
print(X_train.shape, "\n\n", X_test.shape, "\n\n", Y_train.shape, "\n\n", Y_test.shape, "\n\n")

#Import ML models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC 
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

#Import metrics to evaluate the perfomance at each model 
from sklearn import metrics

#Import libraries for crous validation 
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

clf = DecisionTreeClassifier(criterion= 'entropy')

#Fit Decision Tree Classifler #dark-box dữ liêu đầu vào không cần xử lý 
clf = clf.fit(X_train, Y_train)

#Predict testset
Y_pred = clf.predict(X_test) #dự đoán kết quả những thông số biến đầu vào 

#Evaluate performance of the model #đánh giá
print("CART (Tree Prediction) Accuracy: {}", format(sum(Y_pred == Y_test) / len(Y_pred))) #eqlvalent
print("CART (Tree Prediction) Accuracy by calling metrics : ", metrics.accuracy_score(Y_test, Y_pred))

clf = SVC() #Support vecter mechine  thuộc nhóm Supervised Learning(học có giám sát)dùng để phân chia dữ liệu thành các nhòm rieng biệt (tách xanh và đỏ)

#Fit SVM Classifier 
clf.fit(X_train, Y_train)

#Predict testset
Y_pred = clf.predict(X_test)

#Evaluate performance of the model 
print("SVM Accuracy : ", metrics.accuracy_score(Y_test, Y_pred))
print("\n")

#Evaluate a score by cross-validation 
scores = cross_val_score(clf, X, Y, cv= 5) #cross_val_score tra ve diem cua lan kt day #cross_val_perdict trả về gt y dự đoán cho lần kiểm tra 
print("scores = {} \n final score= {} \n".format(scores,scores.mean()))
print("\n")

#Fit logistic 
lr = LogisticRegression(max_iter=10000)
lr.fit(X_train, Y_train)

#Predict testset
Y_pred = lr.predict(X_test)

#Evaluate performance of the model
print("LR: ", metrics.accuracy_score(Y_test, Y_pred))

#Evaluate a scores by cross validation 
scores = cross_val_score(lr, X, Y, cv = 5)
print("scores = {} \n final scores = {} \n ".format(scores, scores.mean()))
print("\n")

# Fit Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, Y_train)

# Predict testset
Y_pred=rf.predict(X_test)

# Evaluate performance of the model
print("RF:  ", metrics.accuracy_score(Y_test, Y_pred))


# Evaluate a score by cross-validation
scores = cross_val_score(rf, X, Y, cv=5)
print("scores = {} \n final score = {} \n".format(scores, scores.mean()))