#!/usr/bin/env python
# coding: utf-8

# Imports e constantes

# In[26]:


import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dense, Input, GlobalMaxPooling1D
from tensorflow.keras.layers import Conv1D, MaxPool1D, Embedding, LSTM
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.initializers import Constant


from tensorflow.keras import layers
from tensorflow.keras import losses
from tensorflow.keras import preprocessing
from tensorflow.keras.layers.experimental.preprocessing import TextVectorization

from gensim.models import Word2Vec, KeyedVectors


import os
import re
import shutil
import string
import numpy as np

MAX_SEQUENCE_LENGTH = 1000
MAX_NUM_WORDS = 10000 
EMBEDDING_DIM = 300 
VALIDATION_SPLIT = 0.2


# Definindo os caminhos para a base de dados de depressivos e o embedding escolhido pré-treinado. Para uso das funcionalidades do Tensorflow, o dataset foi formatado:
# 
# 
# 
# *   Os conjuntos de treinamento e de teste foram separados previamente em diretórios diferentes
# *   Em cada um dos diretórios, existe um diretório para cada classe
# *   Dentro dos diretórios de classes estão os documentos (postagens) pertencentes à classe, cada um em um arquivo de texto próprio.
# 
# 

# Carregando o dataset de treinamento, com uma divisão adicional para conjuntos de treinamento e de validação.

# In[14]:


batch_size = 32
seed = 42

raw_train_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dataset_dir,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=seed
)


# In[15]:


raw_val_ds = tf.keras.preprocessing.text_dataset_from_directory(
    train_dataset_dir, 
    batch_size=batch_size, 
    validation_split=0.2, 
    subset='validation', 
    seed=seed)


# In[16]:


raw_test_ds = tf.keras.preprocessing.text_dataset_from_directory(
    test_dataset_dir, 
    batch_size=batch_size)


# Vetorização dos textos em tokens. Ao contrário dos outros experimentos, a tokenização e vetorização não resultam em um BoW; em vez de cada dado significar a presença ou não (ou a frequência ou o valor tf-idf) de cada palavra em dado documento, o vetor resultante é uma lista de índices de token, ordenados na lista de acordo com sua aparição no documento. Para isso, é criada também uma lista adicional contendo o token e o índice pelo qual ele é representado. Neste caso, é necessário definir um número máximo para esse vocabulário e um tamanho constante do vetor de cada documento.

# In[17]:


vectorize_layer = TextVectorization(
    max_tokens = MAX_NUM_WORDS,
    output_mode = 'int',
    output_sequence_length = MAX_SEQUENCE_LENGTH
)


# In[18]:


# separa os textos dos rótulos
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)


# In[19]:


# Vetoriza os textos de acordo com as informações anteriores e reúne vetores e rótulos

def vectorize_text(text, label):
    text = tf.expand_dims(text, -1)
    return vectorize_layer(text), label


# In[ ]:


text_batch, label_batch = next(iter(raw_train_ds))
first_doc, first_label = text_batch[0], label_batch[0]


# In[21]:


train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)


# Carrega arquivo de embeddings a ser usado

# In[23]:


embeddings_filename = 'cbow_s300.txt'
get_ipython().run_line_magic('time', 'w2v_model = KeyedVectors.load_word2vec_format(os.path.join(project_path, w2v_model_filename), binary=False) # txt format')


# Cria uma matriz de embeddings específica para o problema, que representa os tokens do vocabulário da vetorização do conjunto de treinamento e o vetor de embedding relacionado encontrado na matriz de embeddings pré-treinada. Caso o token não esteja presente no embedding pré-treinado, o vetor estará zerado.

# In[27]:


# vocabulario "inverso": palavras sao o indice da lista, o conteudo sao os indices
word_index = dict(zip(vectorize_layer.get_vocabulary(), range(len(vectorize_layer.get_vocabulary()))))

#talvez de erro se numero maximo de tokens for menor do que o tamanho do vocabulario no conjunto de treinamento
embedding_matrix = np.zeros((MAX_NUM_WORDS, EMBEDDING_DIM))

hits = 0
misses = 0

for word, i in word_index.items():
    if word in w2v_model:
        embedding_matrix[i] = w2v_model[word]
        hits+=1
    else:
        misses +=1

print("Converted %d words (%d misses)" % (hits, misses))


# Cria a camada de embeddings, a partir dos valores da matriz obtida no passo anterior

# In[28]:


embedding_layer = Embedding(MAX_NUM_WORDS,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable=False)
print("Preparing of embedding matrix is done")


# Criação e teste com uma rede convolutiva.

# In[29]:


cnnmodel = Sequential()
cnnmodel.add(embedding_layer)
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPool1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(MaxPool1D(5))
cnnmodel.add(Conv1D(128, 5, activation='relu'))
cnnmodel.add(GlobalMaxPooling1D())
cnnmodel.add(Dense(128, activation='relu'))
cnnmodel.add(Dense(len(raw_train_ds.class_names), activation='softmax'))

cnnmodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['acc']
)


# In[32]:


# cnnmodel.summary()
cnnmodel.fit(train_ds, batch_size=128, epochs=5, validation_data=val_ds) 


# In[33]:


score, acc = cnnmodel.evaluate(test_ds)
print('Test accuracy with CNN:', acc)


# 
# Criação e teste com uma rede LSTM.

# In[34]:


rnnmodel = Sequential()
rnnmodel.add(embedding_layer)
rnnmodel.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
rnnmodel.add(Dense(len(raw_train_ds.class_names), activation='sigmoid'))
rnnmodel.compile(
    loss='sparse_categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)


# In[36]:


rnnmodel.fit(
    train_ds,
    batch_size=32,
    epochs=5,
    validation_data=val_ds
)


# In[37]:


score, acc = rnnmodel.evaluate(test_ds,
                            batch_size=32)
print('Test accuracy with RNN:', acc)

