#gradient descent for linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from random import shuffle

boston = load_boston()
data = boston.data

x1 = []
for i in range(0, data.shape[0]):
    x1.append(data[i, :])

## normalize data
for i in range(0, len(x1[1])):
    minx = 999999999.0
    maxx = -999999999.0
    for j in range(0, len(x1)):
        if minx > x1[j][i]:
            minx = x1[j][i]
        if maxx < x1[j][i]:
            maxx = x1[j][i]
    for k in range(0, len(x1)):
        x1[k][i] = (x1[k][i] - minx) / (maxx - minx)

maxy = -9999999999999.0
miny = 9999999999999.0

for i in range(0, len(boston.target)):
    if maxy < boston.target[i]:
        maxy = boston.target[i]
    if miny > boston.target[i]:
        miny = boston.target[i]

y = []
for i in range(0, len(boston.target)):
    y.append((boston.target[i] - miny) / (maxy - miny))

## insert 1 in position 0 for each observation as parameter of b[0]
x = []
for i in range(0, len(x1)):
    x.append(np.insert(x1[i], 0, 1))

### set training and test data
testx = []
testy = []
alltrain = []
for i in range(0, len(x)):
    if i % 10 == 0:
        testx.append(x[i])
        testy.append(y[i])
    else:
        alltrain.append(np.insert(x[i], 0, y[i]))


### make a model to calculate the function 
def model(b, x):
    y = 0.0
    for i in range(0, len(b)):
        y = y + b[i] * x[i]
    return y


###calculate rmse
def rmse(p, y):
    a = 0.0
    b = 0.0
    for i in range(0, len(p)):
        a = a + (p[i] - y[i]) ** 2
    b = (a / len(p)) ** 0.5
    return b


####one epoch of learning
def epoch(b, x, y, lr):
    # b = []
    # for i in range(0, 14):
    #	b.append(0.0)
    # error = 0.0
    for i in range(0, len(x)):
        error = model(b, x[i]) - y[i]
        for j in range(0, len(x[i])):
            b[j] = b[j] - lr * error * x[i][j]
    return b


###run multiple times with threshhold
lr = [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
for k in range(0, 6):
    rmse1 = []
    epochs = 0
    error = 0.0
    b = []
    for j in range(0, 14):
        b.append(0.0)
    while epochs <= 10:
        shuffle(alltrain)
        trainx = []
        trainy = []
        # b1 = []
        for i in range(0, len(alltrain)):
            trainx.append(alltrain[i][1:len(alltrain)])
            trainy.append(alltrain[i][0])
        b = epoch(b, trainx, trainy, lr[k])
        predictions = []
        for i in range(0, len(testx)):
            prediction = model(b, testx[i])
            predictions.append(prediction)
        rmse1.append(rmse(predictions, testy))
        epochs = epochs + 1
    plt.plot(rmse1)
    plt.title('Learning Rate: %f' % lr[k])
    plt.savefig('Question1<%f>.png' % lr[k])
    plt.clf()

###is rmse calculated only on test dataset? do we record rmse to find the best model?
###what is the cutoff of rmse?
###what is the other dimention of the plot?
