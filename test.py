from keras.models import Sequential
from keras.layers import Dense
import numpy

result = {}

for m in [100, 200, 500]:
    dataset = numpy.loadtxt('data/' + str(m) + '.txt', delimiter=",")
    X = dataset[:,0:32]
    Y = dataset[:,32]

    model = Sequential()
    model.add(Dense(32, input_dim=32, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

    model.fit(x = dataset[:,0:32], y = dataset[:,32], epochs=150, batch_size=16)
    solution = [0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1]
    prediction = model.predict(numpy.array([solution]))
    result[m] = prediction[0][0]
    print(prediction)

print(result)