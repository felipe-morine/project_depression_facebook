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

documents = dataset[0]
categories = dataset[1]


# Função customizada para pré-processamento. São retirados as stopwords a partir da lista de stopwords do NLTK em português (outros podem ser utilizados), as palavras ficam todas em minúsculo, e são retirados pontuações e dígitos (etapas que provavelmente já foram feitas previamente, mas por prevenção também são executadas aqui), e é feita a tokenização. Stemming a ser explorado.
# O conjunto de dados obtido não é vetorizado (e.g. BOW); apenas são separados os tokens.

# In[43]:


from sklearn.feature_extraction.text import TfidfVectorizer 
tfidf_vectorizer=TfidfVectorizer()
stemmed_tfidf_vectors = tfidf_vectorizer.fit_transform(stemmed_dataset)


# In[44]:


from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score

mnb = MultinomialNB()
scores = cross_val_score(mnb, stemmed_tfidf_vectors , categories, cv=10)
print("Stemmed BoW: "+str(scores.mean()))

