from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

import sys


def make_plot(errors, rule):
    plt.plot(errors.keys(), errors.values(), 'ro')
    plt.xlabel('size of the training set')
    plt.ylabel('square error')
    plt.savefig('graphs/' + str(rule) + '.pdf')
    plt.close()

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

        model.fit(x = dataset[:,0:32], y = dataset[:,32], epochs=150, batch_size=16)
        testset = np.loadtxt('data/testsets/150_test.txt', delimiter=",")
        X_test = testset[:,0:32]
        Y_test = testset[:,32]
        predictions = model.predict(np.array(X_test))
        errors[m] = np.square(np.subtract(Y_test, predictions)).mean()
        print(predictions)
        print(errors)
    return errors

def make_graphs_for_rules(rules, sizes, layer_count):
    for r in rules:
        errors = find_errors_for_rule(r, sizes, layer_count)
        make_plot(errors, r)

layer_count = 2
#rules = [95, 110, 150, 222]
rules = [150]
sizes = [1000 + 1000*x for x in range(9)]
#sizes = [10000 + 10000*x for x in range(9)]

make_graphs_for_rules(rules, sizes, layer_count)
