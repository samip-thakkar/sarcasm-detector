# -*- coding: utf-8 -*-
"""
Created on Thu Dec 19 18:11:31 2019

@author: Samip
"""

#Import libraries
import pandas as pd
import re
from nltk.stem.porter import PorterStemmer

#import model libraries
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

#Loading a dataset
data = pd.read_json('Sarcasm_Headlines_Dataset.json', lines = True)

#Check for null values
print(data.isnull().any(axis = 0))

#Data Cleaning (Eliminating the special symbols from headline)
data['headline'] = data['headline'].apply(lambda s : re.sub('[^a-zA-Z]', ' ', s))

#Feature and Label extraction. Here, feature is headline column and label is is_sarcastic column
features = data['headline']
labels = data['is_sarcastic']

#Feature stemming. Stemming is the proceess of reducing to its stem word. eg. reading and reader to read.
#Stemming data
ps = PorterStemmer()
features = features.apply(lambda x: x.split())
features = features.apply(lambda x : ' '.join([ps.stem(word) for word in x]))

#Vectorization of features using TF-IDF(Term Frequecy Inverse Document Frequency) vectors. It transforms text into meaningful representation of numbers
#Vectoriztion data with max no. of features = 5000
from sklearn.feature_extraction.text import TfidfVectorizer
tv = TfidfVectorizer(max_features = 5000)
features = list(features)
features = tv.fit_transform(features).toarray()

#Splitting the data into training and testing data
features_train, features_test, label_train, label_test = train_test_split(features, labels, train_size = 0.75, random_state = 0)

#TRAINING MODELS

#Model 1: Logistic Regession
lr = LogisticRegression()
lr.fit(features_train, label_train)
print("Logistic Regression")
print("Training Score:", lr.score(features_train, label_train))
print("Testing Score:", lr.score(features_test, label_test))

#Model 2: Support Vector Classifier
svc = LinearSVC()
svc.fit(features_train, label_train)
print("Support Vector Classifier")
print("Training Score:", svc.score(features_train, label_train))
print("Testing Score:", svc.score(features_test, label_test))

#Model 3: Naive Bayes
nb = GaussianNB()
nb.fit(features_train, label_train)
print("Naive Bayes")
print("Training Score:", nb.score(features_train, label_train))
print("Testing Score:", nb.score(features_test, label_test))

#Model 4: Random Forest Classifier
rf = RandomForestClassifier(n_estimators = 10, random_state = 0)
rf.fit(features_train, label_train)
print("Random Forest Classifier")
print("Training Score:", rf.score(features_train, label_train))
print("Testing Score:", rf.score(features_test, label_test))

