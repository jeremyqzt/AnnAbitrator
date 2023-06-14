from collections import defaultdict
from utils.parserUtils import get_tokenizer, load_tokens, tokenize, load_doc, make_input_vector
from keras.layers import Dense, Input
from keras.models import Model, load_model
from numpy import array
from models.textModel import create_category_mlp, create_mlp
from keras.layers import concatenate
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelBinarizer
from sklearn.feature_extraction.text import CountVectorizer

types = {"Description": "str", "Input": "str", "Status": 'int', "Type": 'int', "Amount": "float",
         "ShipAmount": "float", "TaxAmount": "float", "TotalAmount": "float", "Success": "int", "SKU": "str"}
csv_train = pd.read_csv("./data/amount.csv",
                        names=["Description", "Input", "SKU", "Status", "Type", "Amount", "ShipAmount", "TaxAmount",
                               "TotalAmount", "Success"],
                        encoding="utf-8",
                        dtype=types, keep_default_na=False)

csv_train.head()
scaler = MinMaxScaler()

csv_train[['Amount', 'ShipAmount', 'TaxAmount', 'TotalAmount']] = scaler.fit_transform(
    csv_train[['Amount', 'ShipAmount', 'TaxAmount', 'TotalAmount']])

in_vec = csv_train[['Status', 'Type', 'Amount',
                    'ShipAmount', 'TaxAmount', 'TotalAmount']].to_numpy()

to_train = []
for vec in in_vec:
    to_convert = make_input_vector([
        {"size": 50, "to_set": vec[1]}, {
            "size": 1, "to_set": vec[3]}, {"size": 1, "to_set": vec[4]},
        {"size": 1, "to_set": vec[5]}])
    to_train.append(to_convert)

csv_train.to_csv(
    'data/des.txt', columns=["Description"], index=False, header=False)
csv_train.to_csv('data/inp.txt', columns=["Input"], index=False, header=False)
csv_train.to_csv('data/sku.txt', columns=["SKU"], index=False, header=False)

expected = csv_train[['Success']]
create_model = True
trainingDataIn = load_tokens("data/des.txt")
trainingDataOut = load_tokens("data/inp.txt")
all_tokens = trainingDataIn + trainingDataOut


tokenizer = get_tokenizer(all_tokens)
trainingDataIn = tokenize(tokenizer, load_doc("data/des.txt").rstrip())
trainingDataOut = tokenize(tokenizer, load_doc("data/inp.txt").rstrip())
trainingDataSKU = tokenize(tokenizer, load_doc("data/sku.txt").rstrip())

n_words = trainingDataIn.shape[1]
n_words2 = trainingDataOut.shape[1]
n_words3 = trainingDataSKU.shape[1]

pos = create_mlp(n_words)
neg = create_mlp(n_words2)
sku = create_category_mlp(n_words3)
other = create_category_mlp(len(to_train[0]))

combinedInput = concatenate([pos.output, neg.output, sku.output, other.output])

x = Dense(16, activation="relu")(combinedInput)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

if create_model:

    model = Model(inputs=[pos.input, neg.input,
                  sku.input, other.input], outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=['accuracy'])

    print(trainingDataIn)
    model.fit(x=[trainingDataIn, trainingDataOut, trainingDataSKU, array(to_train)],
              y=array(expected), epochs=500, verbose=2)
    model.save('saved_model/my_model')
else:
    model = load_model('saved_model/my_model')
a = model.predict(x=[trainingDataIn, trainingDataOut,
                  trainingDataSKU, array(to_train)])
print(a)
