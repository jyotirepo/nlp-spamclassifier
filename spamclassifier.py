# -*- coding: utf-8 -*-
"""
Created on Wed May 26 00:22:34 2021

@author: jysethy
"""
## SPAM Classifier using NLP

import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

message = pd.read_csv("SMSSpamCollection", sep='\t', names=['label', 'messages'])

## pre-processing of data
lema = WordNetLemmatizer()
corpus = []

for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['messages'][i])
    review = review.lower()
    review = review.split()
    review = [lema.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
#creating TF-IDF model

from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X = tfidf.fit_transform(corpus).toarray()

y = pd.get_dummies(message['label'])

y = y.iloc[:,1].values

## Train test split for X and y
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


## Implementaion of Naive base classifier for classificaiton

from sklearn.naive_bayes import MultinomialNB

mnb = MultinomialNB()

spam_detection_model = mnb.fit(X_train, y_train)

y_pred = spam_detection_model.predict(X_test)


from sklearn.metrics import accuracy_score, confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print('model Confusion Matrix :\n', cm)

accuracy = accuracy_score(y_test, y_pred)

print('Model Accuracy:', accuracy)


