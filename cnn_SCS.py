import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random
import time

num_epochs = 2000
batch_size = 32
decay_steps = 500
decay_rate = 0.98
starter_learning_rate = 1e-3
train_ratio = 0.99
# load the data
dir_path = 'samples'
data = []
reg_constant = 0.01

# for f in os.listdir(dir_path):
#     item = np.load(dir_path + '/' + f,'r')
#     data.append(item)
# data = np.array(data)

# train_size = int(0.8*len(data))
# random.shuffle(data)
# train = data[:train_size]
# dat_mean = np.mean(train[:,0,:,:,:] ,axis=0, keepdims=True)
# dat_std = np.std(train[:,0,:,:,:], axis=0, keepdims=True)
# val = data[train_size:]
# X_train = ((train[:,0,:,:,:]-dat_mean)/dat_std).reshape(-1,1,16,16,16)
# y_train = train[:,1:,:,:,:]*10e2
# X_val = ((val[:,0,:,:,:]-dat_mean)/dat_std).reshape(-1,1,16,16,16)
# y_val = val[:,1:,:,:,:]*10e2

train_path = './3DGAN/generatedData'
val_path = 'samples'
data = []
val=[]
for f in os.listdir(train_path)[:5000]:
	item = np.load(train_path + '/' + f,'r')
	item=item.reshape(16,16,16,7)
	data.append(item)

for f in os.listdir(val_path)[:500]:
	item = np.load(val_path + '/' + f,'r')
	if item.shape==(7,16,16,16):
		val.append(item)

data = np.array(data) 
val = np.array(val)
val = np.moveaxis(val,1,-1)


dat_mean = np.mean(data[:,:,:,:,0] ,axis=0, keepdims=True)
dat_std = np.std(data[:,:,:,:,0], axis=0, keepdims=True)
X_train = ((data[:,:,:,:,0]-dat_mean)/dat_std).reshape(-1,16,16,16,1)

y_train = data[:,:,:,:,1:]*10e2
X_val = ((val[:,:,:,:,0]-dat_mean)/dat_std).reshape(-1,16,16,16,1)

y_val = val[:,:,:,:,1:]*10e2

keep_prob = 0.2 # keep probability for dropout
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

reset_graph()
# build the network
X = tf.placeholder(tf.float32, shape = [None, 16, 16, 16,1], name = 'geometry')
y = tf.placeholder(tf.float32, shape = [None, 16, 16, 16,6], name = 'strains')
training = tf.placeholder_with_default(False, shape=(), name='training')

# CNN1
C_1 = tf.layers.conv3d(inputs = X, kernel_size = 3, filters = 32, padding = 'same',activation = tf.nn.relu)
C_1 = tf.layers.max_pooling3d(inputs = C_1, pool_size = 2, strides = 2, padding = 'same')

# CNN2
C_2 = tf.layers.conv3d(inputs = C_1, kernel_size = 3, filters = 64, padding = 'same', activation = tf.nn.relu)
C_2 = tf.layers.max_pooling3d(inputs = C_2, pool_size = 2, strides = 2, padding = 'same')
C_2 = tf.layers.batch_normalization(C_2, training = training)

# FC1
F_1 = tf.layers.flatten(C_2)
F_1 = tf.layers.dense(F_1, units = 1024, activation = tf.nn.relu)
F_1 = tf.layers.dropout(F_1, rate=keep_prob, training = training)


# FC2
F_2 = tf.layers.dense(F_1, units = 30, activation = tf.nn.relu)
F_2 = tf.layers.dropout(F_2, rate=keep_prob, training = training)

# Fc3 
F_3 = tf.layers.dense(F_2, units = 1024, activation = tf.nn.relu)
F_3 = tf.layers.dropout(F_3, rate=keep_prob, training = training)
# FC4
F_4  = tf.layers.dense(F_3, units = 3072, activation = tf.nn.relu)
F_4 = tf.layers.dropout(F_4, rate=keep_prob, training = training)
F_4 = tf.layers.batch_normalization(F_4, training = training)
# CNN3
C_3 = tf.reshape(F_4, [-1, 4,4,4,48])
C_3 = keras.layers.UpSampling3D(size=(2,2,2))(C_3)
C_3 = tf.layers.conv3d(C_3, kernel_size = 3, filters = 32, padding = 'same', activation = tf.nn.relu)


# CNN 4
C_4 = keras.layers.UpSampling3D(size=(2,2,2))(C_3)
C_4 = tf.layers.conv3d(C_4, kernel_size = 3, filters = 16, padding = 'same',  activation = tf.nn.relu)
C_4 = tf.layers.batch_normalization(C_4, training = training)
# CNN 5
output = tf.layers.conv3d(C_4, kernel_size = 3, filters = 6, padding = 'same')
print(output.get_shape())


# metrics
with tf.variable_scope('loss'):
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.losses.mean_squared_error(y, output) + reg_constant*sum(reg_losses)
	train_step = tf.train.AdamOptimizer(learning_rate = starter_learning_rate).minimize(loss)
	SS_tol = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
	SS_res = tf.reduce_sum(tf.square(y-output))
	R_squared = 1. - SS_res/SS_tol



global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step=global_step,
                                               decay_steps=decay_steps,decay_rate=decay_rate)
train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
    
   
#set GPU
config = tf.ConfigProto() 
#config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
saver = tf.train.Saver()    
start_time = time.localtime()
print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))
extra_graphkeys_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.Session(config=config) as sess:
	sess.run(tf.global_variables_initializer())
	num= int(len(X_train)/batch_size)
	best_validation = 0
	for epoch in range(num_epochs):
		train_loss = 0
		train_mae = 0
		for itr in range(num):
			X_batch = X_train[itr*batch_size:(itr+1)*batch_size]
			y_batch = y_train[itr*batch_size:(itr+1)*batch_size]
			_train_step, _loss, _r2 , _update= sess.run([train_step, loss, R_squared, extra_graphkeys_update_ops], feed_dict = {training: True, X: X_batch, y: y_batch})
			train_loss += _loss
			train_mae += _r2
		val_loss, val_r2 = sess.run([loss, R_squared], feed_dict = {X: X_val, y: y_val})
		if val_r2 > best_validation:
			saved_model = saver.save(sess, './saved_model_SCS')
			best_validation = val_r2
		print('Epoch: {} Training Loss: {:.4f} Trianing R2: {:.4f} Val Loss: {:.4f} Val R2: {:.4f} Best So Far {:.4f}'.format(epoch+1,train_loss/num, _r2, val_loss, val_r2, best_validation))
