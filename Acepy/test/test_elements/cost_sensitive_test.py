import os 
os.chdir(r'C:\Users\31236\Desktop\al_tools\acepy\test\test_elements')
os.environ["TF_CPP_MIN_LOG_LEVEL"]="3"

import scipy.io as scio
import pandas as pd
import numpy as np 
import h5py


filename = 'yeast-go.mat'
data = h5py.File(filename, 'r')
# yg_data = scio.loadmat(filename)

# print(type(data))
print([i for i in data.keys()])
print(data.values())
# ins_dis = data['ins_dis']
# print(np.shape(ins_dis))
# print(ins_dis[0])
# print(yg_data.keys())