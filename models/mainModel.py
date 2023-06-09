from collections import Counter
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer
from keras.layers import concatenate, Dense
from keras.models import Model

# [mlp.output, cnn.output]


def get_combined_model(models):
    inputs = [i.input for i in models]
    outputs = [i.output for i in models]
    combinedInput = concatenate(outputs)
    x = Dense(4, activation="relu")(combinedInput)
    x = Dense(1, activation="linear")(x)
    model = Model(inputs=inputs, outputs=x)
    return model, x
