from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout

from keras import regularizers
from keras import models

import numpy as np
import sys

import matplotlib.pyplot as plt

def find_error_for_rule(model, rule, packs, epc, bs, name):
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
    plt.plot(predictions,'r', Y_test,'b')
    plt.savefig('diff-plot-no/rule-'+str(rule)+'-'+name+'.png', dpi=300)
    plt.clf()

    error_vector = np.abs(np.subtract(Y_test.reshape((1,128)), predictions.reshape((1,128))))
    
    return (error_vector.min(), error_vector.mean(), error_vector.max(), error_vector.std(), np.median(error_vector), Y_test.min(), Y_test.max(), predictions.min(), predictions.max())

def get_model(afun, layer_count):
    model = Sequential()
    model.add(Dense(32, input_dim=32, activation=afun))

    for layer in range(0, layer_count):
        model.add(Dense(256, activation=afun))

    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model


rule_start = 0
rule_count = 255

try:
    rule_start, rule_count = int(sys.argv[2]), int(sys.argv[3])
except:
    pass


print("rule, training set size, batch size, epochs, activation function, layers, (min error, avg. error, max error, std. dev. of err, meadian error)")


for rule in range(0,256):
    for layers in [1,2]:
        for packs in [8]:
            for batch_size in [64]:
                for epochs in [256,512]:
                    for afun in ['relu','selu','tanh']:
                        print(rule, packs, batch_size, epochs, afun, layers, find_error_for_rule(get_model(afun, layers), rule, packs, epochs, batch_size, afun+"_"+str(batch_size)+"_"+str(layers)+"_"+str(epochs)))
                        sys.stdout.flush()
