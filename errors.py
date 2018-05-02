from keras.models import Sequential
from keras.layers import Dense
from keras.layers import AlphaDropout

from keras import regularizers

import numpy as np
import sys

def find_error_for_rule(rule, size, layer_count, afun, epc, bs):
    if(size > 10000):
        size = 10000
        
    dataset = np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_10000.txt', delimiter=",")

    model = Sequential()
    model.add(Dense(32, input_dim=32, activation=afun, kernel_initializer='lecun_normal', kernel_regularizer=regularizers.l2(0.05)))

    reg_base = 0.1
    drop_base = 0.1
    exp_base = max(6, layer_count)

    for layer in range(0, layer_count):
        reg = reg_base / (layer+1)
        drop = drop_base / (layer+1)
        model.add(Dense(2**(exp_base-layer), activation=afun, kernel_initializer='lecun_normal', activity_regularizer=regularizers.l2(reg), bias_regularizer=regularizers.l2(reg), kernel_regularizer=regularizers.l2(reg)))
        if(layer != layer_count - 1):
            model.add(AlphaDropout(drop))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='mean_squared_error', optimizer='adam')
    model.fit(x = dataset[0:size,0:32], y = dataset[0:size,32], epochs=epc, batch_size=bs, verbose=0)
    testset = np.loadtxt('data/testsets2/' + str(rule) + '_test.txt', delimiter=",")
    X_test = testset[:,0:32]
    Y_test = testset[:,32]

    predictions = model.predict(np.array(X_test))

    error_vector = np.square(np.subtract(Y_test, predictions))
    return (error_vector.min(), error_vector.mean(), error_vector.max(), error_vector.std(), np.median(error_vector))


afun = 'tanh'
try:
    afun = sys.argv[1]
except:
    pass

rule_start = 0
rule_count = 256

try:
    rule_start, rule_count = int(sys.argv[2]), int(sys.argv[3])
except:
    pass

print("rule, training set size, batch size, epochs, activation function, (min error, avg. error, max error, std. dev. of err, meadian error)")

for rule in range(rule_start, rule_count):
    for size in [500]:
        for batch_size in [50]:
            for epochs in [250]:
                print(rule,size,batch_size,epochs,afun,find_error_for_rule(rule,size,5,afun,epochs,batch_size))
                sys.stdout.flush()
