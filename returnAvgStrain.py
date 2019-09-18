import numpy as np
import os

data_dir = 'new_sample/'
filenames =  os.listdir(data_dir)

def avgstrain(filename):
    temp = np.load(data_dir + filename,'r')
    geo = temp[0]
    strain = temp[1:]
    avg_strain = np.mean(strain, axis=(1,2,3))
    data = np.array((geo,avg_strain))
    return data

if not os.path.exists('avgData/'):
    os.mkdir('avgData/')

for filename in filenames:
    data = avgstrain(filename)
    data.dump('avgData/'+filename)
