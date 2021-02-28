#!/usr/bin/env python

import pandas as pd
import numpy as np
import tensorflow as tf
import spacy
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model

from gensim.models import KeyedVectors


def word_to_index(data): 
    '''
    Input:
        data - numpy array with str sentences 
    Output:
        word_index_arr - numpy array with word indexes
    '''
    word_index_list = []
    for sentence in data:
        line = []
        for word in str(sentence).split():
            if word in word2vec_model.vocab:
                line.append(word2vec_model.vocab[word].index)
            else:
                line.append(199999)
        word_index_list.append(line)
    word_index_arr = np.array(word_index_list)
    return word_index_arr


def save_history(history, filename):
    '''
    Input:
        history - keras history instance
        filename - string filename to save
    Output:
       None
    '''
    np.save('../model_data/' + filename + 'hist.npy', history.history)
    return


def load_history(path):
    '''
    Input:
        path - string path of history file, filename included
    Output:
       history - keras history instance
    '''
    history = np.load(path, allow_pickle='TRUE').item()
    return history


def load_model(path):
    '''
    Input:
        path - string path of model file, filename included
    Output:
       history - keras history instance
    '''
    saved_model = load_model(path)
    return saved_model


def display_loss_plot(history, filename):
    '''
    Input:
        history - keras history instance
        filename - string filename to save
    Output:
       None
    '''
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(loss) + 1)
    nb_epochs = len(loss)
  
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} 

    plt.figure(figsize=(20,10))
    plt.xlabel(str(nb_epochs) + ' Epochs', **font)
    plt.ylabel('Loss', **font)
    plt.plot(epochs, loss, 'r', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss', **title_font)
    plt.legend()
    plt.savefig('../rel/figuras/' + filename + 'loss.png')


def display_acc_plot(history, filename):
    '''
    Input:
        history - keras history instance
        filename - string filename to save
    Output:
       None
    '''
    accuracy = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    epochs = range(1, len(accuracy) + 1)
    nb_epochs = len(accuracy)
  
    font = {'family' : 'normal',
            'weight' : 'normal',
            'size'   : 14}
    title_font = {'fontname':'Arial', 'size':'16', 'color':'black', 'weight':'normal',
                  'verticalalignment':'bottom'} 

    plt.figure(figsize=(20,10))
    plt.xlabel(str(nb_epochs) + ' Epochs', **font)
    plt.ylabel('Accuracy', **font)
    plt.plot(epochs, accuracy, 'r', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation acc', **title_font)
    plt.legend()
    plt.savefig('../rel/figuras/' + filename + 'acc.png')

    
sns.set_theme()

# Tamanho máximo de uma sentença
SEQUENCE_MAXLEN = 50

# Carrega os embeddings do word2vec
word2vec_model = KeyedVectors.load_word2vec_format("../data/word2vec_200k.txt")

# Carrega os datasets
train = pd.read_csv('../data/train.csv', sep=';')
val = pd.read_csv('../data/val.csv', sep=';')
test = pd.read_csv('../data/test.csv', sep=';')

x_train = train['review_text'].values
y_train = train['overall_rating'].values
x_train = word_to_index(x_train)
x_train = keras.preprocessing.sequence.pad_sequences(x_train, value=0, maxlen=SEQUENCE_MAXLEN, padding='post', truncating='post')

x_val = val['review_text'].values
y_val = val['overall_rating'].values
x_val = word_to_index(x_val)
x_val = keras.preprocessing.sequence.pad_sequences(x_val, value=0, maxlen=SEQUENCE_MAXLEN, padding='post', truncating='post')

x_test = test['review_text'].values
y_test = test['overall_rating'].values
x_test = word_to_index(x_test)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, value=0, maxlen=SEQUENCE_MAXLEN, padding='post', truncating='post')

print(x_train.shape)
print(x_val.shape)
print(x_test.shape)

# Modelo LSTM

def get_lstm_model(dropout_prob=0.0):   
    model = keras.Sequential()
    model.add(layers.Input(shape=(SEQUENCE_MAXLEN, )))
    embedding_layer = word2vec_model.get_keras_embedding()
    embedding_layer.trainable = False
    model.add(embedding_layer)
    model.add(layers.LSTM(64))
    model.add(layers.Dropout(dropout_prob))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


#LSTM 1

print("Modelo LSTM com Dropout 0.0\n")
name = 'm1_lstm_drop0.0'
model = get_lstm_model(dropout_prob=0.0)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)

print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()

# LSTM 2

print("Modelo LSTM com Dropout 0.25\n")
name = 'm2_lstm_drop0.25'
model = get_lstm_model(dropout_prob=0.25)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)


print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()

# LSTM 3

print("Modelo LSTM com Dropout 0.5\n")
name = 'm3_lstm_drop0.5'
model = get_lstm_model(dropout_prob=0.5)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)

print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()
# Modelo Bidirecional

def get_bidirectional_model(dropout_prob=0.0):   
    model = keras.Sequential()
    model.add(layers.Input(shape=(SEQUENCE_MAXLEN, )))
    embedding_layer = word2vec_model.get_keras_embedding()
    embedding_layer.trainable = False
    forward_layer = keras.layers.LSTM(64)
    backward_layer = keras.layers.LSTM(64, go_backwards=True)
    model.add(embedding_layer)
    model.add(keras.layers.Bidirectional(forward_layer, backward_layer=backward_layer))
    model.add(layers.Dropout(dropout_prob))
    model.add(keras.layers.Dense(5, activation='softmax'))
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# Bidirecional 1

print("Modelo Bidirecional com Dropout 0.0\n")
name = 'm1_bidirectional_drop0.0'
model = get_bidirectional_model(dropout_prob=0.0)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)

print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()
# Bidirecional 2

print("Modelo Bidirecional com Dropout 0.25\n")
name = 'm2_bidirectional_drop0.25'
model = get_bidirectional_model(dropout_prob=0.25)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)

print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()
# Bidirecional 3
print("Modelo Bidirecional com Dropout 0.5\n")
name = 'm3_bidirectional_drop0.5'
model = get_bidirectional_model(dropout_prob=0.5)
es = EarlyStopping(monitor='val_accuracy', mode='max', verbose=1, patience=10)
mc = ModelCheckpoint('../model_data/' + name + 'best_model.h5', monitor='val_accuracy', mode='max', save_best_only=True)
history = model.fit(x_train, y_train, batch_size=32, epochs=100, validation_data=(x_val, y_val), callbacks=[es, mc])
save_history(history, name)
model.evaluate(x_test, y_test)
display_loss_plot(history, name)
display_acc_plot(history, name)

print()
print('Acurácia e loss conjunto de testes')
model.evaluate(x_test, y_test)

print()
print('Acurácia e loss conjunto de testes - modelo melhor desempenho')
best_model_lstm = model.load_weights('../model_data/' + name + 'best_model.h5')
model.evaluate(x_test, y_test)
print()
