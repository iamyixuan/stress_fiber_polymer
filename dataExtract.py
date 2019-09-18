import numpy as np
import pandas as pd
import os

'''
#data = pd.read_csv('strainData.csv')
data = pd.read_csv('cellStrainCoord.csv') # 891 x-values; 897 y-values; 456 z-values
points1= sorted(list(set(data['Points_0'])))
points2= sorted(list(set(data['Points_1'])))
points3= sorted(list(set(data['Points_2'])))
x_coords = range(len(points1))
y_coords = range(len(points2))
z_coords = range(len(points3))
data['Points_0'] = data['Points_0'].apply(lambda x: x_coords[points1.index(x)])
data['Points_1'] = data['Points_1'].apply(lambda x: y_coords[points2.index(x)])
data['Points_2'] = data['Points_2'].apply(lambda x: z_coords[points3.index(x)])

data.to_csv('filterdCoord.csv', index = False)
'''

# df1 = pd.read_csv('filterdCoord.csv')
# data = np.zeros((891,897,456))

# for i in range(len(df1)):
#     x = df1.iloc[i,7]
#     y = df1.iloc[i,8]
#     z = df1.iloc[i,9]
#     Ezz = df1.iloc[i,6]
#     data[x][y][z] = Ezz

# np.save('Eyz.npy', data)

from scipy.ndimage import zoom

def interpolate_image(filename):
    f = np.load(filename,'r')
    zoom_img = zoom(f, (300/891.0, 300/897.0, 152/456.0))
    return zoom_img


Exx = interpolate_image('Exx.npy') 
Exy = interpolate_image('Exy.npy') 
Exz = interpolate_image('Exz.npy') 
Eyy = interpolate_image('Eyy.npy') 
Eyz = interpolate_image('Eyz.npy') 
Ezz = interpolate_image('Ezz.npy') 
np.save('Exx_scale.npy', Exx)
np.save('Exy_scale.npy', Exy)
np.save('Exz_scale.npy', Exz)
np.save('Eyy_scale.npy', Eyy)
np.save('Eyz_scale.npy', Eyz)
np.save('Ezz_scale.npy', Ezz)