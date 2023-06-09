from utils.parserUtils import read_csv
from keras.layers import Dense
from keras.layers import Input
from keras.models import Model, Sequential
from numpy import array

trainingDataIn = read_csv("data/pos.txt")
trainingDataOut = read_csv("data/neg.txt")

n_words = trainingDataIn.shape[1]
print(trainingDataIn.shape)

# define network
model = Sequential()
model.add(Dense(50, input_shape=(n_words,), activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile network
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(trainingDataIn, array([1,0,0,1]), epochs=500, verbose=2)

a = model.predict(trainingDataIn)
print(a)