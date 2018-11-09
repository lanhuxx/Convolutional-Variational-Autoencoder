# Author: Y. Li

# Language: Python 3.6.5
# Environment: tensorflow-gpu 1.8.0

from __future__ import division, print_function, absolute_import

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import tensorflow.contrib.slim as slim

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# Visualize decoder setting
# Parameters
training_epochs = 200
batch_size = 5
display_step = 100

LR = 1e-3
# LEARNING_RATE_BASE = 0.0001
# LEARNING_RATE_DECAY = 0.99

min_after_dequeue = 1000
TRAINING_SAMPLE_SIZE = 5000

IMAGE_SIZE1 = 243
IMAGE_SIZE2 = 729
IMAGE_CHANNEL = 1
DEEP_SIZE = 8

fc1_nodes = 2048
fc2_nodes = 512
fc3_nodes = 128
fc4_nodes = 64

# -------------------------------Import Training Samples---------------------------
files = tf.train.match_filenames_once('MMC_tfrecords/tr/train_data.tfrecords')
filename_queue = tf.train.string_input_producer(files, shuffle=False)

reader = tf.TFRecordReader()
_,serialized_example = reader.read(filename_queue)
features = tf.parse_single_example(serialized_example, features={
	'img_raw':tf.FixedLenFeature([], tf.string)
	})
image = features['img_raw']

image = tf.decode_raw(image, tf.uint8)
image = tf.reshape(image, [656, 875, IMAGE_CHANNEL])

# decoded_image = tf.cast(decoded_image, tf.float32) * (1. / 255) - 1
# decoded_image = tf.cast(decoded_image, tf.float32) / 255

image = tf.image.convert_image_dtype(image, dtype=tf.float32)
# surrounding blank is useless
destorted_image = tf.image.crop_to_bounding_box(image, 195, 90, IMAGE_SIZE1, IMAGE_SIZE2)
destorted_image = tf.image.resize_images(destorted_image, (185, 366), method=1)
image = tf.cast(destorted_image, tf.float32)

capacity = min_after_dequeue + 3 * batch_size
images = tf.train.shuffle_batch(
	[image], batch_size=batch_size,
	capacity = capacity, min_after_dequeue=min_after_dequeue)
# ---------------------------------------------------------------------------------

def lrelu(x, leak=0.2, name="lrelu"):
	return tf.maximum(x, leak*x)

# Building the encoder
def encoder(x, reuse=False):
	with tf.variable_scope('encoder') as scope:
		if reuse:
			scope.reuse_variables()
		with slim.arg_scope([slim.conv2d], padding='VALID',
			weights_initializer=tf.contrib.layers.xavier_initializer(),  # xavier initializer
			weights_regularizer=slim.l2_regularizer(0.0005),
			activation_fn=lrelu,
			kernel_size=[5, 5],
			stride=2):
			net = slim.conv2d(x, DEEP_SIZE, scope='d_conv1')
			net = slim.batch_norm(net, decay=0.9, epsilon=1e-5, scale=True, scope='d_bd1')
			net = slim.conv2d(net, DEEP_SIZE * 2, scope='d_conv2')
			net = slim.batch_norm(net, decay=0.9, epsilon=1e-5, scale=True, scope='d_bd2')
			net = slim.conv2d(net, DEEP_SIZE * 4, scope='d_conv3')
			net = slim.batch_norm(net, decay=0.9, epsilon=1e-5, scale=True, scope='d_bd3')
			net = slim.conv2d(net, DEEP_SIZE * 8, scope='d_conv4')
			pool_shape = net.get_shape().as_list()
			nodes = pool_shape[1] * pool_shape[2] * pool_shape[3]
			net = tf.reshape(net, [-1, nodes])

			e_fc1 = slim.fully_connected(net, fc1_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc1')
			e_fc1 = lrelu(e_fc1)

			e_fc2 = slim.fully_connected(e_fc1, fc2_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc2')
			e_fc2 = lrelu(e_fc2)

			e_fc3 = slim.fully_connected(e_fc2, fc3_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc3')
			e_fc3 = lrelu(e_fc3)

			fc_mean = slim.fully_connected(e_fc3, fc4_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc4')
			fc_std = slim.fully_connected(e_fc3, fc4_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='e_fc4')
	return fc_mean, fc_std

# sampler
def sampler(fc_mean, fc_std):
	eps = tf.random_normal(tf.shape(fc_std), dtype=tf.float32, mean=0, stddev=1.0, name='epsilon')
	z = fc_mean + tf.exp(fc_std / 2) * eps
	return z

# Building the decoder
def decoder(x):
	with tf.variable_scope('decoder') as scope:
		with slim.arg_scope([slim.batch_norm],
			decay=0.9,
			epsilon=1e-5,
			scale=True):
			d_fc1 = slim.fully_connected(x, fc3_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc1')
			d_fc1 = lrelu(d_fc1)

			d_fc2 = slim.fully_connected(d_fc1, fc2_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc2')
			d_fc2 = lrelu(d_fc2)

			d_fc3 = slim.fully_connected(d_fc2, fc1_nodes, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc3')
			d_fc3 = lrelu(d_fc3)

			d_fc4 = slim.fully_connected(d_fc3, 10240, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc4')
			d_fc4 = lrelu(d_fc4)
			net = tf.reshape(d_fc4, [-1, 8, 20, DEEP_SIZE*8])

			net = lrelu(slim.batch_norm(net, scope='g_bd1'))
			# The default of slim.conv2d_transpose is weights_initializer=initializers.xavier_initializer()
			net = slim.conv2d_transpose(net, num_outputs=DEEP_SIZE * 8, kernel_size=[6, 5], stride=2, activation_fn=None, scope='g_conv_tran1', padding='VALID')
			net = lrelu(slim.batch_norm(net, scope='g_bd2'))
			net = slim.conv2d_transpose(net, num_outputs=DEEP_SIZE * 4, kernel_size=[6, 5], stride=2, activation_fn=None, scope='g_conv_tran2', padding='VALID')
			net = lrelu(slim.batch_norm(net, scope='g_bd3'))
			net = slim.conv2d_transpose(net, num_outputs=DEEP_SIZE * 2, kernel_size=[5, 5], stride=2, activation_fn=None, scope='g_conv_tran4', padding='VALID')
			net = lrelu(slim.batch_norm(net, scope='g_bd4'))
			net = slim.conv2d_transpose(net, num_outputs=IMAGE_CHANNEL, kernel_size=[5, 6], stride=2, activation_fn=None, scope='g_conv_tran5', padding='VALID')
			net = lrelu(net)
	return net

# Construct model
def train():
	# tf Graph input (only pictures)
	X = tf.placeholder("float", [None, 185, 366, IMAGE_CHANNEL])

	with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
		fc_mean, fc_std = encoder(X)
		# Hidden space of sampling
		sample_latent = sampler(fc_mean, fc_std)
		# reconstruct output
		decoder_op = decoder(sample_latent)

		# Prediction
		y_pred = decoder_op
		# Targets (Labels) are the input data.
		y_true = X

		global_step = tf.Variable(0, trainable = False)

		# learning_rate = tf.train.exponential_decay(
		# 	LEARNING_RATE_BASE,
		# 	global_step,
		# 	TRAINING_SAMPLE_SIZE / batch_size,
		# 	LEARNING_RATE_DECAY,
		# 	name='learning_rate')

	# Define loss and optimizer
	def vae_loss(x_reconstructed, x_true, fc_mean, fc_std):
		encode_decode_loss = x_true * tf.log(1e-10+x_reconstructed) + (1 - x_true) * tf.log(1e-10+1-x_reconstructed)
		encode_decode_loss =- tf.reduce_sum(encode_decode_loss, 1)
		#KL loss
		kl_div_loss = 1 + fc_std - tf.square(fc_mean) - tf.exp(fc_std)
		kl_div_loss =- 0.5 * tf.reduce_sum(kl_div_loss, 1)
		return tf.reduce_mean(encode_decode_loss+kl_div_loss)
	loss_op = vae_loss(decoder_op, y_true, fc_mean, fc_std)
	
	with tf.variable_scope(tf.get_variable_scope(), reuse=False):
		optimizer = tf.train.RMSPropOptimizer(LR).minimize(loss_op)

	saver=tf.train.Saver(tf.global_variables())
	# Launch the graph
	with tf.Session() as sess:
		sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])
		coord = tf.train.Coordinator()
		threads = tf.train.start_queue_runners(coord=coord)
		
		total_batch = TRAINING_SAMPLE_SIZE // batch_size
		# Training cycle
		for epoch in range(training_epochs):
			# Loop over all batches
			for i in range(total_batch):
				batch_xs = sess.run(images)  # max(x) = 1, min(x) = 0
				# Run optimization op (backprop) and cost op (to get loss value)
				_, c = sess.run([optimizer, loss_op], feed_dict={X: batch_xs})
				# Display logs per epoch step
				if (i+1) % display_step == 0:
					print(epoch+1,'\t', i+1, '\t', "{:.9f}".format(c))
		saver.save(sess,'MMC_ckpt/CVA/CVA.ckpt')

		print("Optimization Finished!")

		# Compare original images with their reconstructions
		a = 0
		for j in range(50):
			xs = sess.run(images)
			encode_decode = sess.run(y_pred, feed_dict={X: xs})
			for i in range(batch_size):
				plt.imshow(sess.run(images)[i, :, :].reshape(185, 366, IMAGE_CHANNEL))
				plt.xticks([])    # close x & y axes
				plt.yticks([])
				# plt.show()
				plt.savefig('MMC_new_images/CVA/real_images/%d.jpg' % a)

				plt.imshow(encode_decode[i, :, :].reshape(185, 366, IMAGE_CHANNEL))
				plt.xticks([])    # close x & y axes
				plt.yticks([])
				# plt.show()
				plt.savefig('MMC_new_images/CVA/pred_images/%d.jpg' % a)

				a = a + 1
train()
