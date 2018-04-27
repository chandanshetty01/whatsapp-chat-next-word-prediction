import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
from keras.models import Sequential, load_model , Model
from keras.layers import Dense, Activation
from keras.layers import Input, Reshape, LSTM, Dropout
from keras.layers import TimeDistributed, Flatten
from keras.layers.core import Dense, Activation, Dropout, RepeatVector
from keras.optimizers import RMSprop
from keras.engine import InputLayer
from keras.utils import np_utils
import re
#import tfcoreml as tf_converter

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams
import coremltools

path = 'whatspp_chat_1.txt'
text = open(path).read().lower()

text = text.replace("chandan:", "")
text = text.replace("chethan sp:", "")
text = text.replace("missed voice cal", "")

 #remove everything between []
regex = '\[.*?\]'
text = re.sub(regex,"",text)

 #remove everything between <>
regex = '\<.*?\>'
text = re.sub(regex,"",text)

#Removing all special characters for now
text = re.sub(r'\W+', ' ', text)

text = text.rstrip() #remove \n in last line
words = text.split()

#removing texts containing  http
words = [ x for x in words if "http" not in x ]

#removing texts containing  www
words = [ x for x in words if "www" not in x ]

#removing special character -----
words = [ x for x in words if "————" not in x ]

#Remove words containing attachment
words = [ x for x in words if ".pdf" not in x ]

#Remove words with single character
words = [ x for x in words if len(x) > 1]

print(words)

n_words = len(words)
unique_words = sorted(list(set(words)))
words_indices = dict((c, i) for i, c in enumerate(unique_words))
indices_words = dict((i, c) for i, c in enumerate(unique_words))

next_word = []
sentences = []
SEQUENCE_LENGTH = 2

for i in range(0, n_words - SEQUENCE_LENGTH):
    seq_in  = words[i: i+SEQUENCE_LENGTH]
    seq_out = words[i+SEQUENCE_LENGTH]
    sentences.append([words_indices[word] for word in seq_in])
    next_word.append(words_indices[seq_out])
    print(f'{seq_in} => {seq_out}')

# Reshape X to size of [samples, time steps, features] and scale it to 0-1
# Represent Y as one hot encoding
X = np.reshape(sentences, (len(sentences), SEQUENCE_LENGTH, 1))
Y = np_utils.to_categorical(next_word)

print(X.shape)
print(f'num training examples: {len(sentences)}')

model_name = 'keras_word_predictor.h5'
coremodel_name = 'coremodel_word_predictor.mlmodel'

def generateModel():
    model = Sequential()
    model.add(LSTM(256, input_shape=(X.shape[1], X.shape[2])))
    model.add(Dropout(0.2))
    model.add(Dense(Y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = model.fit(X, Y, validation_split=0.05, batch_size=128, epochs=20).history
    model.save(model_name)
    pickle.dump(history, open("history.p", "wb"))
    return model

def generateCoreMLModel():
    model = load_model(model_name)
    #model = load_model('reshaped-model.h5')
    mlmodel = coremltools.converters.keras.convert(model)
    #coreml_model = coremltools.converters.sklearn.convert(model, ["bedroom", "bath", "size"], "price")
    mlmodel.save(coremodel_name)

generateModel()
generateCoreMLModel()

model = load_model(model_name)
coreml_model = coremltools.converters.keras.convert(model)

def getNextWord(words, count = 4):
    result = []
    pattern = []

    for word in words:
        pattern.append(words_indices[word])

    x = np.reshape(pattern, (1, len(pattern), 1))
    #x = x/float(len(words))
    prediction = model.predict(x)
    for i in range(0, count):
        index = np.argmax(prediction)
        result.append(indices_words[index])
        prediction = np.delete(prediction, 0)
    return result

def generate_suggestions(words):
    words = words[:SEQUENCE_LENGTH]
    next_words = getNextWord(words)
    print(f' suggestions for {words} => {next_words}')

generate_suggestions(['dude', 'stated', 'office'])
print("Completion of prgaram")
