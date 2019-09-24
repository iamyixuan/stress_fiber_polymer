import numpy as np
import os

data_dir = 'sample32/'
filenames =  os.listdir(data_dir)
def maxstrain(filename):
    temp = np.load(data_dir + filename,'r')
    geo = temp[0]
    max_strain = np.max(temp[1:])
    index = np.where(temp == max_strain)
    coord =np.array(index[1:]).reshape(-1,)
    data = np.array((geo,coord))
    return data

if not os.path.exists('max_Data/'):
    os.mkdir('max_Data/')

for filename in filenames:
    data = maxstrain(filename)
    data.dump('max_Data/'+filename)
