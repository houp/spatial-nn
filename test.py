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
        dataset = np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_10000.txt', delimiter=",")

        model = Sequential()
        model.add(Dense(64, input_dim=32, activation='tanh'))

        for i in range(0, layer_count):
            model.add(Dense(64, activation='tanh'))

        model.add(Dense(1, activation='sigmoid'))

        model.compile(loss='mean_squared_error', optimizer='adam')

        model.fit(x = dataset[m:,0:32], y = dataset[m:,32], epochs=287, batch_size=200, verbose=1)
        testset = np.loadtxt('data/testsets/' + str(rule) + '_test.txt', delimiter=",")
        X_test = testset[:,0:32]
        Y_test = testset[:,32]
        predictions = model.predict(np.array(X_test))
        errors[m] = np.square(np.subtract(Y_test, predictions)).mean()
    return errors

def make_graphs_for_rules(rules, sizes, layer_count):
    for r in rules:
        errors = find_errors_for_rule(r, sizes, layer_count)
        make_plot(errors, r)

layer_count = 10
rules = [r for r in range(8)]
#rules = [150]
#sizes = [1000 + 1000*x for x in range(8)]
sizes = [500 + 500*x for x in range(6)]

make_graphs_for_rules(rules, sizes, layer_count)
