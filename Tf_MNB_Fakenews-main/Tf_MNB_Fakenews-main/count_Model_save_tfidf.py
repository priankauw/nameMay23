#Importing the Libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import numpy as np
from flask import Flask, request,render_template
from flask_cors import CORS
import os
#from sklearn.externals import joblib
import pickle
import flask
import os
import newspaper
from newspaper import Article
import urllib
from sklearn.metrics import classification_report, confusion_matrix
    # Load the training dataset
dataset = pd.read_csv('train.csv')
dataset=dataset.dropna()
dataset.reset_index(inplace=True)

# Prepare the input data for the classifier
X = dataset['text']
print(X.head(8))
y = dataset['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
# Define the pipeline
pipeline = Pipeline([
    ('vectorizer', CountVectorizer(stop_words='english', max_features=5000, ngram_range=(1,3))),
    ('classifier', MultinomialNB())
])

# Train the classifier


#Training our data
pipeline.fit(X_train, y_train)

#Predicting the label for the test data
pred = pipeline.predict(X_test)
print(pred)
#probabilities = pipeline.predict_proba(X_test)
#C=np.argmax(probabilities,axis=1)

#print("class", C)
#print(probabilities, C)

#Checking the performance of our model
print(classification_report(y_test, pred))
print(confusion_matrix(y_test, pred))


#Serialising the file
with open('count_small_model.pkl', 'wb') as handle:
    pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)

