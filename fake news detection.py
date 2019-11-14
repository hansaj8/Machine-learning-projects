#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd

#Read the data

df=pd.read_csv(r'C:\Users\dell\Downloads\news\news.csv') # I have uploaded news.csv on my  github account 

# Get shape and head
df.shape  # shape is -(6335, 4)
df.head()

# Get the labels
labels=df["label"]
labels.head()

# Split the dataset

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(df['text'], df["label"], test_size=0.2, random_state=7)

# Initialize a TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

'''
Stop words are the most common words in a language that are to be filtered out before processing the natural language data
and a TfidfVectorizer turns a collection of raw documents into a matrix of TF-IDF features.
'''



#Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train) 
tfidf_test=tfidf_vectorizer.transform(x_test)

# Initialize a PassiveAggressiveClassifier
from sklearn.linear_model import PassiveAggressiveClassifier
pac=PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train,y_train)

#Predict on the test set and calculate accuracy
from sklearn.metrics import accuracy_score, confusion_matrix

y_pred=pac.predict(tfidf_test) 
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')

# Build confusion matrix
from sklearn.metrics import accuracy_score, confusion_matrix
confusion_matrix(y_test,y_pred, labels=['FAKE','REAL'])


# In[ ]:





# In[ ]:




