#!/usr/bin/env python
# coding: utf-8

# Imports

# In[2]:


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


# Carregando o conjunto de dados de depressão e separando classes do corpus

# In[5]:


import pandas as pd
dataset = pd.read_csv(
    os.path.join(project_path, dataset_filename), header=None, sep='\t'
)

dataset = dataset[dataset[1]!= 1]
dataset = dataset[dataset[1]!= 2]

documents = dataset[0]
categories = dataset[1]


# In[6]:


vec_tri = CountVectorizer(binary=True, stop_words=None, ngram_range=(3,3), max_features=1000)
bow_tri = vec_tri.fit_transform(documents)
bow_tri = pd.DataFrame(bow_tri.toarray(), columns=vec_tri.get_feature_names_out())


# In[7]:


selector = SelectKBest(mutual_info_classif, k=300)
selector.fit(bow_tri, categories)


# In[8]:


mask = selector.get_support()
selected_features = bow_tri.columns[mask]

selected_bow_tri = bow_tri[bow_tri.columns & list(selected_features)]


# In[9]:


mnb = MultinomialNB()
scores = cross_val_score(mnb, bow_tri, categories, cv=10)
print("Trigrams: "+str(scores.mean()))

scores = cross_val_score(mnb, selected_bow_tri, categories, cv=10)
print("Selected trigrams: "+str(scores.mean()))


# In[ ]:




