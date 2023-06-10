from utils.parserUtils import get_tokenizer, load_tokens, tokenize, load_doc
from keras.layers import Dense, Input
from keras.models import Model
from numpy import array
from models.textModel import create_mlp
from keras.layers import concatenate

all_tokens = load_tokens("data/pos.txt") + load_tokens("data/neg.txt") + load_tokens("data/test/pos.txt") + load_tokens("data/test/neg.txt")

trainingDataIn = load_tokens("data/pos.txt")
trainingDataOut = load_tokens("data/neg.txt")

testIn = load_tokens("data/test/pos.txt")
testOut = load_tokens("data/test/neg.txt")

tokenizer = get_tokenizer(all_tokens)
trainingDataIn = tokenize(tokenizer,load_doc("data/pos.txt"))
trainingDataOut = tokenize(tokenizer,load_doc("data/neg.txt"))
testIn = tokenize(tokenizer,load_doc("data/test/pos.txt"))
testOut = tokenize(tokenizer,load_doc("data/test/neg.txt"))

n_words = trainingDataIn.shape[1]
n_words2 = trainingDataOut.shape[1]

pos = create_mlp(n_words, False)
neg = create_mlp(n_words2, False)

combinedInput = concatenate([pos.output, neg.output])

x = Dense(16, activation="relu")(combinedInput)
x = Dense(8, activation="relu")(x)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[pos.input, neg.input], outputs=x)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(x=[trainingDataIn, trainingDataOut], y=array([1,0,0,1]), epochs=500, verbose=2)

a = model.predict(x=[testIn, testOut])
print(a)