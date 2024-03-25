#!/usr/bin/env python
# coding: utf-8

# Imports

# In[ ]:


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

from sklearn.metrics import confusion_matrix
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV

get_ipython().system('pip install liwc')
import liwc


# Importação dos dicionários NLTK necessários

# In[ ]:


# necessario em toda inicializacao
import nltk
nltk.download('stopwords')
nltk.download('punkt')


# Carregando o conjunto de dados de depressão e separando classes do corpus

# In[ ]:


import pandas as pd
train_dataset = pd.read_csv(
    os.path.join(project_path, train_dataset_filename)
)

test_dataset = pd.read_csv(
    os.path.join(project_path, test_dataset_filename)
)

train_corpus = train_dataset['document']
test_corpus = test_dataset['document']
train_categories = train_dataset['class']
test_categories = test_dataset['class']


# Função customizada para pré-processamento. São retirados as stopwords a partir da lista de stopwords do NLTK em português (outros podem ser utilizados), as palavras ficam todas em minúsculo, e são retirados pontuações e dígitos (etapas que provavelmente já foram feitas previamente, mas por prevenção também são executadas aqui), e é feita a tokenização. Stemming a ser explorado.
# O conjunto de dados obtido não é vetorizado (e.g. BOW); apenas são separados os tokens.

# In[ ]:


def embedding_preprocess_corpus(document):
    
    mystopwords = set(stopwords.words("portuguese"))

    document = word_tokenize(document)
    return [token.lower() for token in document if token not in mystopwords
               and not token.isdigit() and token not in punctuation]


# In[ ]:


embedding_train_corpus = train_corpus.apply(embedding_preprocess_corpus)
embedding_test_corpus = test_corpus.apply(embedding_preprocess_corpus)


# Carregando o modelo de Word2Vec pré-treinado. Modelo disponível no repositório do NILC. CBOW de 929606 tokens e 300 dimensões. Importante notar que ele não foi treinado em nenhum corpus de redes sociais.

# In[ ]:


get_ipython().run_line_magic('time', 'w2v_model = KeyedVectors.load_word2vec_format(os.path.join(project_path, w2v_model_filename), binary=False) # txt format')


# Extração da matriz de vetores de características obtidas a partir da combinação dos tokens dos documentos e dos embeddings no modelo Word2Vec.
# 
# O vetor de características para cada documento inicialmente é um vetor de 300 dimensões, preenchido com 0s. Para cada token no documento, se existe um vetor de embedding no modelo W2V correspondente a esse token (isto é, se o vetor está no vocabulário do modelo de embedding), os valores são somados e a soma é dividida pela quantidade de tokens válidos encontrados. Em outras palavras, o vetor de caracteríticas para o documento é a média dos vetores de embedding de cada token válido.

# In[ ]:


def embedding_features(corpus):
    DIMENSIONS = 300
    features = []
    for document in corpus:
        document_features = np.zeros(DIMENSIONS)
        valid_tokens_counter = 0
        for token in document:
            if token in w2v_model:
                document_features+=w2v_model[token]
                valid_tokens_counter+=1
        if(valid_tokens_counter == 0):
            count_for_this=1
        features.append(document_features/valid_tokens_counter)
    return features


# In[ ]:


embedding_training_features_dataset = embedding_features(embedding_train_corpus)
embedding_test_features_dataset = embedding_features(embedding_test_corpus)


# In[ ]:


from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import mutual_info_classif

feature_selector = SelectPercentile(mutual_info_classif, percentile=0.8)
embedding_training_features_dataset = feature_selector.fit_transform(embedding_training_features_dataset, train_categories)
embedding_test_features_dataset = feature_selector.transform(embedding_test_features_dataset)


# Resultados para um SVM linear.

# In[ ]:


param_grid = [
  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'kernel': ['linear']},
  {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['rbf']},
   {'C': [0.01, 0.1, 1, 10, 100, 1000], 'gamma': [0.01, 0.1, 1, 10], 'kernel': ['poly'], 'degree': [3, 5, 7]},
 ]


svm_classifier = SVC()
grid_search = GridSearchCV(svm_classifier, param_grid, cv=10)
grid_search.fit(embedding_training_features_dataset, train_categories)


# In[ ]:


best_svm = grid_search.best_estimator_

predictions = best_svm.predict(embedding_test_features_dataset)
print("Accuracy: ", accuracy_score(test_categories, predictions))
print(confusion_matrix(test_categories, predictions))

