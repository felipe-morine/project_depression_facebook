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


from sklearn.feature_selection import SelectKBest, mutual_info_classif, VarianceThreshold
from sklearn.feature_extraction.text import CountVectorizer

from sklearn.preprocessing import Normalizer

import pandas as pd

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


# Definindo os caminhos para a base de dados de depressivos e o embedding escolhido pré-treinado

# In[5]:


import pandas as pd
dataset = pd.read_csv(
    os.path.join(project_path, dataset_filename), header=None, sep='\t'
)

documents = dataset[0]
categories = dataset[1]


# Função customizada para pré-processamento. São retirados as stopwords a partir da lista de stopwords do NLTK em português (outros podem ser utilizados), as palavras ficam todas em minúsculo, e são retirados pontuações e dígitos (etapas que provavelmente já foram feitas previamente, mas por prevenção também são executadas aqui), e é feita a tokenização. Stemming a ser explorado.
# O conjunto de dados obtido não é vetorizado (e.g. BOW); apenas são separados os tokens.

# In[6]:


def preprocess_corpus(document):
    mystopwords = set(stopwords.words("portuguese"))
    
    document = word_tokenize(document)
    return [token.lower() for token in document if token not in mystopwords
               and not token.isdigit() and token not in punctuation]


# Execução do pré-processamento feito anteriormente no conjunto de dados.

# In[ ]:


documents = documents.apply(preprocess_corpus)


# Carregando o modelo de Word2Vec pré-treinado. Modelo disponível no repositório do NILC. CBOW de 929606 tokens e 300 dimensões. Importante notar que ele não foi treinado em nenhum corpus de redes sociais.

# In[8]:


get_ipython().run_line_magic('time', 'w2v_model = KeyedVectors.load_word2vec_format(os.path.join(project_path, w2v_model_filename), binary=False) # txt format')


# Extração da matriz de vetores de características obtidas a partir da combinação dos tokens dos documentos e dos embeddings no modelo Word2Vec.
# 
# O vetor de características para cada documento inicialmente é um vetor de 300 dimensões, preenchido com 0s. Para cada token no documento, se existe um vetor de embedding no modelo W2V correspondente a esse token (isto é, se o vetor está no vocabulário do modelo de embedding), os valores são somados e a soma é dividida pela quantidade de tokens válidos encontrados. Em outras palavras, o vetor de caracteríticas para o documento é a média dos vetores de embedding de cada token válido.

# In[9]:


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


# In[10]:


embedding_features_dataset = embedding_features(documents)


# In[11]:


documents = dataset[0]

vec = CountVectorizer()
bow = vec.fit_transform(documents)
bow = pd.DataFrame(bow.toarray(), columns=vec.get_feature_names())


selector = VarianceThreshold(threshold=(.8 * (1 - .8)))
selector.fit(bow, categories)

mask = selector.get_support()
selected_features = bow.columns[mask]

selected_bow = bow[bow.columns & list(selected_features)]


# In[12]:


liwc_dataset_filename = "liwc_dataset.csv"

liwc_dataset = pd.read_csv(
    os.path.join(project_path, liwc_dataset_filename)
)

liwc_data = liwc_dataset.drop('class', axis=1)

liwc_normalized = Normalizer().fit_transform(liwc_data)


# In[13]:


print(type(selected_bow))
print(type(embedding_features_dataset))
print(type(liwc_normalized))


# In[14]:


list_test = pd.DataFrame(embedding_features_dataset)
print(list_test)


# In[15]:


liwc_dataframe = pd.DataFrame(liwc_normalized)


# In[16]:


combined_features = pd.concat([list_test, selected_bow, liwc_dataframe], axis=1)


# São separados os conjuntos de treinamento e de teste. Conjunto de teste representa 0.25 do conjunto completo (688 = 516 + 172)

# In[17]:


train_data, test_data, train_categories, test_categories = train_test_split(combined_features, categories)


# Resultados para um Regressor Logístico.

# In[18]:


lr_classifier = LogisticRegression(random_state=1234)
lr_classifier.fit(train_data, train_categories)
predictions = lr_classifier.predict(test_data)
print("Accuracy: ", accuracy_score(test_categories, predictions))
print(confusion_matrix(test_categories, predictions))


# Resultados para um SVM linear.

# In[19]:


linear_svm_classifier = LinearSVC()
linear_svm_classifier.fit(train_data, train_categories)
predictions = linear_svm_classifier.predict(test_data)
print("Accuracy: ", accuracy_score(test_categories, predictions))
print(confusion_matrix(test_categories, predictions))


# Resultados para um SVM de kernel RBF com C=1.

# In[20]:


svm_classifier = SVC() #c=1, rbf
svm_classifier.fit(train_data, train_categories)
predictions = svm_classifier.predict(test_data)
print("Accuracy: ", accuracy_score(test_categories, predictions))
print(confusion_matrix(test_categories, predictions))


# Resultado para o mesmo SVM anterior, mas com as classes balanceadas pela frequência de cada classe.

# In[21]:


balanced_svm = SVC(class_weight='balanced') #c=1, rbf
balanced_svm.fit(train_data, train_categories)
predictions = balanced_svm.predict(test_data)
print("Accuracy: ", accuracy_score(test_categories, predictions))
print(confusion_matrix(test_categories, predictions))

