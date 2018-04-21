from keras.models import Sequential
from keras.layers import Dense
import numpy as np


def find_errors_for_rule(rule, sizes, layer_count):
    errors = {}
    for m in sizes:
        dataset = np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_' + str(m) + '.txt', delimiter=",")
        X = dataset[:,0:32]
        Y = dataset[:,32]

        model = Sequential()
        model.add(Dense(32, input_dim=32, activation='relu'))

        for i in range(0, layer_count):
            model.add(Dense(32, activation='relu'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(x = dataset[:,0:32], y = dataset[:,32], epochs=150, batch_size=32)
        testset = np.loadtxt('data/testsets/' + str(rule) + '_test.txt', delimiter=",")
        X_test = testset[:,0:32]
        Y_test = testset[:,32]
        predictions = model.predict(np.array(X_test))
        errors[m] = np.square(np.subtract(Y_test, predictions)).mean()
    return errors

def find_errors_for_rules(rules, sizes, layer_count):
    errors = {}
    for r in rules:
        errors[r] = find_errors_for_rule(r, sizes, layer_count)
    return errors


layer_count = 3
rules = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
sizes = [1000 + 1000*x for x in range(8)]


print(find_errors_for_rules(rules, sizes, layer_count))
