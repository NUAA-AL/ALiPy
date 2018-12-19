import scipy.io as scio
import numpy as np
import math

from acepy.utils.misc import randperm
from acepy.query_strategy.query_features import AFASMC_mc

ld = scio.loadmat('C:\\Code\\acepy-additional-methods-source\\feature querying\\AFASMC_code\\data_file.mat')
dataset = ld['dataset']
trainI = ld['trainI'].flatten()-1
testI = ld['testI'].flatten()-1
Omega = ld['Omega'].flatten()-1

struct = dataset[0][0]
data = struct[0]
target = struct[1].flatten()

# print(data)
# print(target)
lambda1 = 1
lambda2 = 1

tr = 0.7
sr = 0.6
queryRound = 100
testAcc = np.zeros((5,queryRound))
queryNum = 50
m,n = data.shape
mtr = math.floor(m*tr)

for t in range(10):
    # trainI = randperm(m-1, mtr)
    testI = list(set(range(m)) - set(trainI))

    Xtr = data[trainI,:]
    Xte = data[testI,:]
    Ytr = target[trainI]
    Yte = target[testI]

    # Omega = randperm(mtr * n-1, math.floor(sr * mtr * n))
    X_all = []

    for i in range (queryRound):
        Xmc, model, dist = AFASMC_mc(Xtr, Ytr, Omega)
        print(Xmc, model, dist)
