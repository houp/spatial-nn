from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout

from keras import regularizers
from keras import models

import numpy as np
import sys

import matplotlib.pyplot as plt

def find_error_for_rule(model, rule, packs, epc, bs):
    if(packs > 8):
        packs = 8
        
    datasets = [np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_'+str(i)+'.txt', delimiter=",") for i in range(1,packs+1)]
    
    dataset = np.concatenate(tuple(datasets))

    X = dataset[:,0:32]
    Y = dataset[:,32]

    model.fit(x = X, y = Y, epochs=epc, batch_size=bs, verbose=0, shuffle=True)

    testset = np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_9.txt', delimiter=",")
    X_test = testset[:,0:32]
    Y_test = testset[:,32]

    predictions = model.predict(X_test)
    plt.plot(predictions,'r',Y_test,'b')
    plt.title('Rule '+str(rule),fontsize=12)
    plt.savefig('diff-plot/rule-'+str(rule)+'.png')
    plt.clf()

    error_vector = np.square(np.subtract(Y_test, predictions))
    return (error_vector.min(), error_vector.mean(), error_vector.max(), error_vector.std(), np.median(error_vector), Y_test.min(), Y_test.max(), predictions.min(), predictions.max())

def get_model(afun, layer_count):
    model = Sequential()
    model.add(Dense(32, input_dim=32, activation=afun, kernel_initializer='lecun_normal'))

    drop_base = 0.05
    exp_base = max(6, layer_count)

    for layer in range(0, layer_count):
        drop = drop_base / (layer+1)
        model.add(Dense(2**min(9,exp_base-layer), activation=afun, kernel_initializer='lecun_normal'))
        if(layer != layer_count - 1):
            model.add(AlphaDropout(drop))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


afun = 'tanh'
try:
    afun = sys.argv[1]
except:
    pass

rule_start = 0
rule_count = 255

try:
    rule_start, rule_count = int(sys.argv[2]), int(sys.argv[3])
except:
    pass


print("rule, training set size, batch size, epochs, activation function, (min error, avg. error, max error, std. dev. of err, meadian error)")

for rule in range(rule_start, rule_count):
    for packs in [8]:
        for batch_size in [64]:
            for epochs in [512]:
                print(rule, packs, batch_size, epochs, afun, find_error_for_rule(get_model(afun, 6), rule, packs, epochs, batch_size))
                sys.stdout.flush()
