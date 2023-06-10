from collections import Counter
from string import punctuation
from os import listdir
from nltk.corpus import stopwords
from keras.preprocessing.text import Tokenizer

def get_tokenizer(tokens):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(tokens)
    return tokenizer

def tokenize(tokenizer, text):
    return tokenizer.texts_to_matrix(text.split("\n"), mode='freq')

def load_tokens(f):
    text = load_doc(f)
    tokens = clean_doc(text)
    return tokens

def load_doc(filename):
    file = open(filename, 'r')
    text = file.read()
    file.close()
    return text


def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    return tokens


def add_doc_to_vocab(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    vocab.update(tokens)


def doc_to_line(filename, vocab):
    doc = load_doc(filename)
    tokens = clean_doc(doc)
    tokens = [w for w in tokens if w in vocab]
    return ' '.join(tokens)


def process_docs(directory, vocab):
    lines = list()
    for filename in listdir(directory):
        path = directory + '/' + filename
        line = doc_to_line(path, vocab)
        lines.append(line)
    return lines


def prepare_data(train_docs, mode):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_docs)
    Xtrain = tokenizer.texts_to_matrix(train_docs, mode=mode)
    return Xtrain


def save_file(lines, filename):
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


def make_vocab(fName, out):
    vocab = Counter()
    process_docs(fName, vocab)
    save_file(out)
