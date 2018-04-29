from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import sys

def find_error_for_rule(rule, size, layer_count, afun, epc, bs):
    if(size>1000):
        size=1000
        
    dataset = np.loadtxt('data/' + str(rule) + '/' + str(rule) + '_1000.txt', delimiter=",")

    model = Sequential()
    model.add(Dense(32, input_dim=32, activation=afun))

    for _ in range(0, layer_count):
        model.add(Dense(64, activation=afun))

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

step=0
try:
    step=int(sys.argv[2])
except:
    pass

    
for rule in range(step*64,(step+1)*64-1):
    for size in [100, 250, 500, 1000]:
        for batch_size in [100, 50, 5]:
            for epochs in [10, 50, 500, 1000]:
                print(rule,size,batch_size,epochs,afun,find_error_for_rule(rule,size,1,afun,epochs,batch_size))
                sys.stdout.flush()
