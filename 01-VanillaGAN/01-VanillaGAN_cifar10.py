"""
	**01-VanillaGAN**

	[paper] https://arxiv.org/pdf/1406.2661.pdf
	[dataset] cifar10
	[reference] https://github.com/ckmarkoh/GAN-tensorflow 
"""
import os, sys
import shutil
import argparse
import numpy as np
import tensorflow as tf
import urllib.request
from skimage.io import imsave

from load_data import load_cifar10


img_height	= 32
img_width	= 32
img_channel	= 3
img_size	= img_height * img_width * img_channel
img_classes	= 10
total_data	= 60000

to_train = True
to_restore = False
output_path = "cifar10_output"

max_epoch = 500
adam_learning_rate = 0.0001

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 256

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

def build_generator(z_prior):
	"""
			||      ||      ||      ||
			||  w1  ||  w2  ||  w3  ||
		z-> || ---> || ---> || ---> ||--> x
			||      ||      ||      ||
			||      ||      ||      ||
			z       h1      h2      h3
	
		z:  noise(=random samples) input layer
		h1: RELU(w1.T * z + b1)
		h2: RELU(w2.T * h1 + b2)
		h3: w3.T * h2 + b3
		x_fake:  tanh(h3)
	"""
	w1 = tf.Variable(tf.truncated_normal([z_size, h1_size], stddev=0.1), name="g_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h1_size]), name="g_b1", dtype=tf.float32)
	h1 = tf.nn.relu(tf.matmul(z_prior, w1) + b1)
	w2 = tf.Variable(tf.truncated_normal([h1_size, h2_size], stddev=0.1), name="g_w2", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([h2_size]), name="g_b2", dtype=tf.float32)
	h2 = tf.nn.relu(tf.matmul(h1, w2) + b2)
	w3 = tf.Variable(tf.truncated_normal([h2_size, img_size], stddev=0.1), name="g_w3", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([img_size]), name="g_b3", dtype=tf.float32)
	h3 = tf.matmul(h2, w3) + b3
	x_generate = tf.nn.tanh(h3)
	g_params = [w1, b1, w2, b2, w3, b3]
	return x_generate, g_params


def build_discriminator(x_data, x_generated, keep_prob):
	"""
		------      ||      ||      ||      ||       --------
		x_data      ||  w1  ||  w2  ||  w3  ||        y_data
		        --> || ---> || ---> || ---> ||--> y = 
		x_fake      ||      ||      ||      ||        y_fake
		------      ||      ||      ||      ||       --------
		           x_in     h1      h2      h3
	
		x_in:  input layer
		h1: RELU(w1.T * z + b1)	 + dropout
		h2: RELU(w2.T * h1 + b2)	+ dropout
		h3: w3.T * h2 + b3
		y:  predict the input data is true or fake (~0:fake, ~1:true)
		|-- y_data = D(x)
		\-- y_fake = D(G(z))
	"""
	x_in = tf.concat([x_data, x_generated], 0)
	w1 = tf.Variable(tf.truncated_normal([img_size, h2_size], stddev=0.1), name="d_w1", dtype=tf.float32)
	b1 = tf.Variable(tf.zeros([h2_size]), name="d_b1", dtype=tf.float32)
	h1 = tf.nn.dropout(tf.nn.relu(tf.matmul(x_in, w1) + b1), keep_prob)
	w2 = tf.Variable(tf.truncated_normal([h2_size, h1_size], stddev=0.1), name="d_w2", dtype=tf.float32)
	b2 = tf.Variable(tf.zeros([h1_size]), name="d_b2", dtype=tf.float32)
	h2 = tf.nn.dropout(tf.nn.relu(tf.matmul(h1, w2) + b2), keep_prob)
	w3 = tf.Variable(tf.truncated_normal([h1_size, 1], stddev=0.1), name="d_w3", dtype=tf.float32)
	b3 = tf.Variable(tf.zeros([1]), name="d_b3", dtype=tf.float32)
	h3 = tf.matmul(h2, w3) + b3
	y_data = tf.nn.sigmoid(tf.slice(h3, [0, 0], [batch_size, -1], name=None))
	y_generated = tf.nn.sigmoid(tf.slice(h3, [batch_size, 0], [-1, -1], name=None))
	d_params = [w1, b1, w2, b2, w3, b3]
	return y_data, y_generated, d_params


def show_result(batch_res, fname, grid_size=(8, 8), grid_pad=5):
	"""
		Save generted image
	"""
	#####
	## interval [-1,1] mapping to the interval [0,1], then multiply 255
	#####
	batch_res = 0.5 * batch_res.reshape((batch_res.shape[0], img_height, img_width, img_channel)) + 0.5
	img_h, img_w = batch_res.shape[1], batch_res.shape[2]
	grid_h = img_h * grid_size[0] + grid_pad * (grid_size[0] - 1)
	grid_w = img_w * grid_size[1] + grid_pad * (grid_size[1] - 1)
	img_grid = np.zeros((grid_h, grid_w, img_channel), dtype=np.uint8)
	for i, res in enumerate(batch_res):
		if i >= grid_size[0] * grid_size[1]:
			break
		img = (res) * 255
		img = img.astype(np.uint8)
		row = (i // grid_size[0]) * (img_h + grid_pad)
		col = (i % grid_size[1]) * (img_w + grid_pad)
		img_grid[row:row + img_h, col:col + img_w, :] = img
	imsave(fname, img_grid)


def train():
	cifar10 = load_cifar10('../cifar10_data')

	#####
	## Model setting
	#####
	x_data = tf.placeholder(tf.float32, [batch_size, img_size], name="x_data")
	z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
	keep_prob = tf.placeholder(tf.float32, name="keep_prob")
	global_step = tf.Variable(0, name="global_step", trainable=False)

	x_generated, g_params = build_generator(z_prior)	# `x_generated(tf.Variable)`: output of generator, i.e. x = G(z)
														# `g_params(tf.Variable)`: weights and biases of generator
	y_data, y_generated, d_params = build_discriminator(x_data, x_generated, keep_prob)

	## discriminator loss function (max. `log D(x) + log(1-D(G(z)))` <===> min. `-(log D(x) + log(1-D(G(z)))`)
	d_loss = - tf.reduce_mean(tf.log(y_data) + tf.log(1 - y_generated))
	## generator loss function (min. log(1-D(G(z))) <===> max. log D(G(z)), since 0 < D(G(z)) < 1 <===> min. -log D(G(z)) )
	g_loss = - tf.reduce_mean(tf.log(y_generated))
	## value of C(G)
	value_of_c = tf.reduce_mean(tf.log(y_data) / tf.log(2.)) + tf.reduce_mean(tf.log(1 - y_generated) / tf.log(2.))

	## Gradient descent by Adam optimization method
	optimizer = tf.train.AdamOptimizer(adam_learning_rate)

	## adjust the parameters [ W1, b1, W2, b2, W3, b3 ] of discriminator to minimize `d_loss function
	d_trainer = optimizer.minimize(d_loss, var_list=d_params)
	## adjust the parameters [ W1, b1, W2, b2, W3, b3 ] of generator to minimize `g_loss function
	g_trainer = optimizer.minimize(g_loss, var_list=g_params)

	init = tf.initialize_all_variables()

	saver = tf.train.Saver()

	sess = tf.Session(config=config)

	sess.run(init)

	if to_restore:
		chkpt_fname = tf.train.latest_checkpoint(output_path)
		saver.restore(sess, chkpt_fname)
	else:
		if os.path.exists(output_path):
			shutil.rmtree(output_path)
		os.mkdir(output_path)

	#####
	## Start to train
	#####

	## Generate validation noise samples z_v
	z_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)

	## go training with 500 epoch
	for i in range(sess.run(global_step), max_epoch):
		## 233 iteration in each epoch 
		for j in range(int(total_data / batch_size)):
			print("epoch:%s, iter:%s" % (i, j))
			x_value, _ = cifar10.train.next_batch(batch_size)	# return image (shape=(batch, 3072)) and label (shape=(batch,10))
			x_value	= 2 * x_value.astype(np.float32) - 1		# centralize
			z_value	= np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)   # generate noise samples z
			#####
			## Update the discriminator
			#####
			_, d_minloss, Dx = sess.run([d_trainer, d_loss, y_data],
								 feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
			print("[batch {0}] discriminator loss: {1}".format(j, d_minloss))
			print("[batch {0}] D(x) = {1}".format(j, np.mean(Dx)))
			if j % 1 == 0:	  # [NOTICE] `j % 1 == 0` always TRUE
				#####
				## Update the generator
				#####
				_, g_minloss, value_c_min = sess.run([g_trainer, g_loss, value_of_c],
									 feed_dict={x_data: x_value, z_prior: z_value, keep_prob: np.sum(0.7).astype(np.float32)})
				print("[batch {0}] generator loss: {1}".format(j, g_minloss))
				print("[batch {0}] C(G) = {1}".format(j, value_c_min))
		## Validation
		x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_sample_val})	# `x_gen_val` = G(z_v)
		show_result(x_gen_val, "{0}/sample{1}.jpg".format(output_path,i))
		## Random sample validation
		z_random_sample_val = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
		x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_random_sample_val})
		show_result(x_gen_val, "{0}/random_sample{1}.jpg".format(output_path,i))

		sess.run(tf.assign(global_step, i + 1))
		saver.save(sess, os.path.join(output_path, "model"), global_step=global_step)


def test():
	z_prior = tf.placeholder(tf.float32, [batch_size, z_size], name="z_prior")
	x_generated, _ = build_generator(z_prior)
	chkpt_fname = tf.train.latest_checkpoint(output_path)

	init = tf.initialize_all_variables()
	sess = tf.Session(config=config)
	saver = tf.train.Saver()
	sess.run(init)
	saver.restore(sess, chkpt_fname)
	z_test_value = np.random.normal(0, 1, size=(batch_size, z_size)).astype(np.float32)
	x_gen_val = sess.run(x_generated, feed_dict={z_prior: z_test_value})
	show_result(x_gen_val, "{0}/test_result.jpg".format(output_path))

def main():
	global max_epoch
	global output_path
	parser = argparse.ArgumentParser(description="%s uasge:" % sys.argv[0], formatter_class=argparse.RawTextHelpFormatter)
	parser.add_argument('command', nargs='?', type=str,
						choices=['train', 'test'],
						help="train: training phase.\n" 
							 "test: testing the latest training result.")
	parser.add_argument('--max_epoch',  help='the maximum epoch to train [default: %(default)s]', type=int, default=max_epoch)
	parser.add_argument('--out_dir',	help='the folder saved output images [default: %(default)s]', type=str, default=output_path)
	args = parser.parse_args()

	command = args.command
	max_epoch = args.max_epoch
	output_path = args.out_dir

	if command == 'train':
		train()
	elif command == 'test':
		test()
if __name__ == '__main__':
	main()
