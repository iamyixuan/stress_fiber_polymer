import numpy as np
import os

data_dir = 'new_sample/'
filenames =  os.listdir(data_dir)
def avgstrain(filename):
    temp = np.load(data_dir + filename,'r')
    geo = temp[0]
    strain_z = temp[6]
    data = np.array((geo,strain_z))
    return data

if not os.path.exists('z_Data/'):
    os.mkdir('z_Data/')

for filename in filenames:
    data = avgstrain(filename)
    data.dump('z_Data/'+filename)
