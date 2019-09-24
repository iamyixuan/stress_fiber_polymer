import numpy as np
import os
data = np.load('data/data.npy','r')
print(data.shape)
# data = data[:,60:300,:160,:]
# set sub sample size as (7,16,16,16)
size = 32
'''
sampleCord = np.random.randint(240, size = (10000,3))
print(data.shape)
# start sampling
def sampling(input,coord,size, fileNum):
    if not os.path.exists('sample32'):
        os.makedirs('sample32')

    if coord[0] <= 240- size and coord[1] <= 160-size  and coord[2] <= 152-size :
        new = input[:,coord[0]:coord[0]+size, coord[1]:coord[1]+size,coord[2]:coord[2]+size]
        np.save('sample32/sample_' + str(fileNum)+'.npy', new)
    else:
        pass

for i, coord in enumerate(sampleCord):
    sampling(data, coord, 32, i)
'''
from scipy import ndimage, misc
import matplotlib.pyplot as plt

img_45 = ndimage.rotate(data, 15, axes = (1, 3), reshape=False)
plt.imshow(img_45[0, 299, :, :])
plt.show()
#sampleCord = np.random.randint(240, size = (5000,3))

# start sampling
#def sampling(input,coord,size, fileNum):
#    if not os.path.exists('sample32_rot'):
#        os.makedirs('sample32_rot')
#
#    if coord[0] <= 150- size and coord[1] <= 300-size  and coord[2] <= 152-size :
#        new = input[:,coord[0]:coord[0]+size, coord[1]:coord[1]+size,coord[2]:coord[2]+size]
#        np.save('sample32_rot/' + str(fileNum)+'.npy', new)
#    else:
#        pass
#
#for i, coord in enumerate(sampleCord):
#    sampling(img_45[:, 150:,:,:], coord, 32, i)
