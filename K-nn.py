# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 09:58:40 2019

@author: MINAL MOON
"""

####  KNN  ##########
#knn works using distance formula so it is necessory to having numeric data
#and is work for non parametric model

# Supervised learning
# KNN-classification
# Datasets:-wheat,cars,diab

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn import neighbors
from sklearn .metrics import accuracy_score
from pandas_ml import ConfusionMatrix
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report as cr
from sklearn.model_selection import train_test_split

# REad the file
path="C:/Users/MINAL MOON/Desktop/sonal R/python code/datasets/wheat.csv"
wheat=pd.read_csv(path)

wheat.head(10)
wheat.dtypes
wheat.shape
# Y var-type
# Null check
wheat.isnull().sum()
wheat[wheat==0].count()
wheat.columns

# Standardisation  of dataset

wheat_std=wheat.copy(deep=True) #copy by value

minmax=preprocessing.MinMaxScaler()

totalcols=len(wheat.columns)
totalcols

scaledvals=minmax.fit_transform(wheat_std.iloc[:
    ,0:totalcols-1])

# copy the scaled values to the X-features
wheat_std.iloc[:,0:totalcols-1]=scaledvals
wheat_std.columns


#split the original data
###
train,test=train_test_split(wheat,test_size=0.3)
print('train={},test={}'.format(train.shape,test.shape))

trainX=train.drop(['type'],axis=1)
trainY=train['type']
print('trainX={},trainY={}'.format(trainX.shape,trainY.shape))

testX=test.drop(['type'],axis=1)
testY=test['type']
print('testX={},testY={}'.format(testX.shape,testY.shape))
####

##anotehr method to split the data
X=wheat_std.iloc[:,0:totalcols-1]
Y=wheat_std.iloc[:,totalcols-1]

trainX,testX,trainY,testY=train_test_split(X,Y,test_size=0.3)

#cross-validation to select the best neighbor
nn=list(range(3,12,2))
print(nn)

cv_scores=[]

#run CV for each value of K and determine the best K
for k in nn:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,trainX,trainY,cv=5,scoring="accuracy")
    cv_scores.append(np.mean(scores))
    
print(cv_scores)    

nn[cv_scores.index(max(cv_scores))]

optimalK=nn[cv_scores.index(max(cv_scores))]
print('optimal vlaue of k={}'.format(optimalK))

#build the KNN model using the optimal K and predict
m1=neighbors.KNeighborsClassifier(n_neighbors=optimalK).fit(trainX,trainY)
#predict
p1=m1.predict(testX)

#Confusion Matrix
ConfusionMatrix(list(testY),list(p1))

#classification report
print(cr(testY,p1))

testY.value_counts()

#####################################################

###build model on real data(wheat ) instead od std. data
path="C:/Users/MINAL MOON/Desktop/sonal R/python code/datasets/wheat.csv"

wheat=pd.read_csv(path)

#split the data
train,test=train_test_split(wheat,test_size=0.3)
print('train={},test={}'.format(train.shape,test.shape))

trainX=train.drop(['type'],axis=1)
trainY=train['type']
print('trainX={},trainY={}'.format(trainX.shape,trainY.shape))

testX=test.drop(['type'],axis=1)
testY=test['type']
print('testX={},testY={}'.format(testX.shape,testY.shape))

nn=list(range(3,12,2))
print(nn)

cv_scores=[]

#run CV for each value of K and determine the best K
for k in nn:
    knn=neighbors.KNeighborsClassifier(n_neighbors=k)
    scores=cross_val_score(knn,trainX,trainY,cv=5,scoring="accuracy")
    cv_scores.append(np.mean(scores))
    
print(cv_scores)    

nn[cv_scores.index(max(cv_scores))]

optimalK=nn[cv_scores.index(max(cv_scores))]
print('optimal vlaue of k={}'.format(optimalK))

#build the KNN model using the optimal K and predict
m1=neighbors.KNeighborsClassifier(n_neighbors=optimalK).fit(trainX,trainY)

#predict
p1=m1.predict(testX)

#Confusion Matrix
ConfusionMatrix(list(testY),list(p1))

#classification report
print(cr(testY,p1))

############################################################

#write a function 
#split tha dta
def splitdata(mydata):
    totalrows=len(mydata)
    X=mydata.iloc[:,0:totalcols-1]
    Y=mydata.iloc[:,totalcols-1]
    trainX,testX,trainY,testY=train_test_split(X,Y,test_size=0.3)
    print('trainX={},trainY={},testX={},testY={}'.format(trainX.shape,trainY.shape,testX.shape,testY.shape))
    return(trainX,trainY,testX,testY)
    
  path="C:/Users/MINAL MOON/Desktop/sonal R/python code/datasets/wheat.csv"
mydata=pd.read_csv(path)
df=splitdata(mydata)
df[0]
df[1]
df[2]
df[3]

#build model
def buildmodel(trainX,trainY,testX,testY):
    NN=list(range(5,18,2))
    print(NN)

    cv_scores=[]
    for k in NN:
        knn=neighbors.KNeighborsClassifier(n_neighbors=k)
        scores=cross_val_score(knn,trainX,trainY,cv=5,scoring="accuracy")
    cv_scores.append(np.mean(scores))
    print(cv_scores)    
   #build model
    model=neighbors.KNeighborsClassifier(n_neighbors=optimalK).fit(trainX,trainY)

   #predict
    prdct=model.predict(testX)

   #Confusion Matrix
    CM=ConfusionMatrix(list(testY),list(prdct))

    return(model,prdct,CM)
    
    
model2,predct2,cm2=buildmodel(trainX,trainY,testX,testY)
cm2
predct2
model2

