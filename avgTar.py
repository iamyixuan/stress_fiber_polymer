import tensorflow as tf
from tensorflow import keras
import os
import numpy as np
import random
import time
from tensorflow.contrib.layers import conv3d_transpose

num_epochs = 5000
batch_size =10
decay_steps = 500
decay_rate = 0.98
starter_learning_rate = 1e-4
train_ratio = 0.99
reg_constant = 0.01
# load the data
train_path = './avgData/'
data = []

for filename in os.listdir(train_path):
    data.append(np.load(train_path + filename,'r'))

INPUT = np.array([i[0] for i in data])
OUTPUT = np.array([i[1] for i in data])*10e3
train = INPUT[:int(0.8*len(OUTPUT))]
val = INPUT[int(0.8*len(OUTPUT)):]

dat_mean = np.mean(train, axis = 0, keepdims = True)
dat_std = np.std(train, axis=0, keepdims=True)
X_train = ((train-dat_mean)/dat_std).reshape(-1,32,32,32,1)
y_train =OUTPUT[:int(0.8*len(OUTPUT))]
X_val = ((val-dat_mean)/dat_std).reshape(-1,32,32,32,1)
y_val = OUTPUT[int(0.8*len(OUTPUT)):]

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)

	return(tf.Variable(initial))

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return(tf.Variable(initial))

def conv3d(x, W, s=[1,1,1,1,1], padding='SAME'):
	if (padding.upper() == 'VALID'):
		return (tf.nn.conv3d(x,W,strides=s,padding='VALID'))
	# SAME
	return (tf.nn.conv3d(x,W,strides=s,padding='SAME'))

def seblock(x, in_cn):

	squeeze = tf.keras.layers.GlobalAveragePooling3D()(x)

	with tf.variable_scope('sq'):
		w = weight_variable([in_cn, in_cn//16])
		b = bias_variable([in_cn//16])
		h = tf.matmul(squeeze, w) + b
		excitation = tf.nn.relu(h)

	with tf.variable_scope('ex'):
		w = weight_variable([in_cn//16, in_cn])  # None1*128
		b = bias_variable([in_cn])
		h = tf.matmul(excitation, w) + b
		excitation = tf.nn.sigmoid(h)  # None*128
		excitation = tf.reshape(excitation, [-1, 1, 1,1, in_cn])
	return x * excitation

def residual_block(x, cn, scope_name, training):
	with tf.variable_scope(scope_name):
		shortcut = x
		w1 = weight_variable([3, 3, 3, cn, cn])
		b1 = bias_variable([cn])
		x1 = tf.layers.batch_normalization(tf.nn.relu(conv3d(x, w1) + b1), training = training)
		w2 = weight_variable([3, 3,3, cn, cn])
		b2 = bias_variable([cn])
		x2 = tf.layers.batch_normalization(conv3d(x1, w2) + b2, training = training)
		x3 = seblock(x2, cn)

	return x3 + shortcut

reset_graph() # this is very important for batchNorm to work

X = tf.placeholder(tf.float32, shape = [None,32, 32, 32, 1], name = 'geometry')
y = tf.placeholder(tf.float32, shape = [None, 6], name = 'strains')
training = tf.placeholder_with_default(False, shape=(), name='training')


C_1 = tf.layers.conv3d(inputs = X, kernel_size = 3, filters = 32, padding = 'same',activation = tf.nn.relu)
C_1 = tf.layers.max_pooling3d(inputs = C_1, pool_size = 2, strides = 2, padding = 'same')


C_2 = tf.layers.conv3d(inputs = C_1, kernel_size = 3, filters = 64, padding = 'same', activation = tf.nn.relu)
C_2 = tf.layers.max_pooling3d(inputs = C_2, pool_size = 2, strides = 2, padding = 'same')

x4 = residual_block(C_2, 64, 'res1', training = training)
x5 = residual_block(x4, 64, 'res2',training = training)
x6 = residual_block(x5, 64, 'res3',training = training)
x7 = residual_block(x6, 64, 'res4',training = training)
x8 = residual_block(x7, 64, 'res5',training = training)

x9 = conv3d_transpose(x8, 64, kernel_size = 3, stride = 2, padding = 'SAME',)
x9 = tf.nn.relu(x9)
x9 = tf.layers.batch_normalization(x9, training = training)

x10 = conv3d_transpose(x9,32, kernel_size = 3, stride=(2,2,2), padding='SAME')
x10 = tf.nn.relu(x10)
x10 = tf.layers.batch_normalization(x10, training = training)
output = tf.layers.conv3d(x10, filters = 1, kernel_size = 3, padding = 'SAME', activation = tf.nn.relu)
output = tf.layers.flatten(output)
output = tf.layers.dense(output, units = 100, activation = tf.nn.relu)
output = tf.layers.dense(output, units = 6, activation = None)


global_step = tf.Variable(0, trainable=False)
add_global = global_step.assign_add(1)
learning_rate = tf.train.exponential_decay(starter_learning_rate,
                                               global_step=global_step,
											   decay_steps=decay_steps,decay_rate=decay_rate)


with tf.variable_scope('loss'):
	reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
	loss = tf.losses.mean_squared_error(y, output) + reg_constant*sum(reg_losses)
	train_step = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(loss)
	SS_tol = tf.reduce_sum(tf.square(y - tf.reduce_mean(y)))
	SS_res = tf.reduce_sum(tf.square(y-output))
	R_squared = 1. - SS_res/SS_tol


extra_graphkeys_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

#set GPU
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# config.gpu_options.allow_growth = True
# saver = tf.train.Saver()
# start_time = time.localtime()
# print('Computing starts at: ', time.strftime('%Y-%m-%d %H:%M:%S', start_time))

with tf.Session() as sess:
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
            best_validation = val_r2
        print('Epoch: {} Training Loss: {:.4f} Trianing R2: {:.4f} Val Loss: {:.4f} Val R2: {:.4f} Best So Far {:.4f}'.format(epoch+1,train_loss/num, _r2, val_loss, val_r2, best_validation))
