#!/usr/bin/env python
# coding: utf-8

# In[12]:


import os
from time import time

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from string import punctuation

import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Importação dos dicionários NLTK necessários

# In[3]:


# necessario em toda inicializacao
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# In[5]:


import pandas as pd
dataset = pd.read_csv(
    os.path.join(project_path, dataset_filename), header=None, sep='\t'
)

documents = dataset[0]
categories = dataset[1]


# In[18]:


vec_uni = CountVectorizer(binary=True, stop_words=None, ngram_range=(1,1), max_features=1000)
bow_uni = vec_uni.fit_transform(documents)
bow_uni = pd.DataFrame(bow_uni.toarray(), columns=vec_uni.get_feature_names_out())


# In[19]:


selector = SelectKBest(mutual_info_classif, k=300)
selector.fit(bow_uni, categories)


# In[20]:


mask = selector.get_support()
selected_features = bow_uni.columns[mask]

selected_bow_uni = bow_uni[bow_uni.columns & list(selected_features)]


# In[21]:


mnb = MultinomialNB()
scores = cross_val_score(mnb, bow_uni, categories, cv=10)
print("Unigrams: "+str(scores.mean()))

scores = cross_val_score(mnb, selected_bow_uni, categories, cv=10)
print("Selected uigrams: "+str(scores.mean()))


# In[ ]:




