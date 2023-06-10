from keras.layers import Dense
from keras.layers import Input
from keras.models import Model, Sequential
from utils.parserUtils import load_doc, prepare_data, process_docs

# vocab_filename = 'vocab.txt'
# vocab = load_doc(vocab_filename)
# vocab = vocab.split()
# vocab = set(vocab)

# lines = process_docs('data/inputDescription.txt', vocab)
# modes = ['binary', 'count', 'tfidf', 'freq']
# n_words = prepare_data(lines, modes[0]).shape[1]


def create_mlp(dim):
    model = Sequential()
    model.add(Dense(128, input_dim=dim, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(4, activation="relu"))

    return model


def create_category_mlp(dim):
    model = Sequential()
    model.add(Dense(8, input_dim=dim, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))

    return model
