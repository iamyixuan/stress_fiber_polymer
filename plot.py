import numpy as np
import matplotlib.pyplot as plt

a = np.load('data.npy','r')
a1 = a[0,:,:,1]
a2 = a[1,:,:,1]*10e3
a3 = a[2,:,:,1]*10e3
a4 = a[3,:,:,1]*10e3
a5 = a[4,:,:,1]*10e3
a6 = a[5,:,:,1]*10e3
a7 = a[6,:,:,1]*10e3

#plt.imshow(a1)
#plt.axis('off')
#plt.savefig('micro.png',format = 'png', dpi=300)
#
ax1=plt.subplot(2,3,1)
ax1.imshow(a2)
ax1.axis('off')
ax2= plt.subplot(2,3,2)
ax2.imshow(a3)
ax2.axis('off')
ax3=plt.subplot(2,3,3)
ax3.imshow(a4)
ax3.axis('off')
ax4 =plt.subplot(2,3,4)
ax4.imshow(a5)
ax4.axis('off')
ax5=plt.subplot(2,3,5)
ax5.imshow(a6)
ax5.axis('off')
ax6=plt.subplot(2,3,6)
ax6.imshow(a7)
ax6.axis('off')
plt.savefig('strains.png',format='png',dpi=300)
