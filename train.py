import tensorflow as tf
import os
import pandas as pd
from PIL import Image
import numpy as np
from numpy import array

flags = tf.app.flags
flags.DEFINE_integer("iterations",450000,"Number of iterations to train")
flags.DEFINE_integer("batch_size",100,"Number of batch size in each iteration")
flags.DEFINE_float("starting_learning_rate",0.01,"starting learning rate")
flags.DEFINE_string("train_models","./train_models","path to save checkpoints")
FLAGS = flags.FLAGS
MOMENTUM = 0.9#0.0005
LEARNING_RATE_DECAY = 0.1
NUM_EPOCHS_PER_LEARNING_RATE_DECAY = 100000
WEIGHT_DECAY_FACTOR = 0.0005#0.9
TRAINING_SIZE = 513


class PretrainedAlexNet(object):


	def __init__(self,images,_keep_prob,_is_test=False):
		self.is_test = _is_test
		self.x = images
		self.keep_prob = _keep_prob
		self.fc8 = None
		self.pre_trained_model = np.load(open(os.getcwd() + "/dataset/bvlc_alexnet.npy", "rb"), encoding="latin1").item()
		self.create()


	def create(self):
		conv1 = tf.layers.conv2d(
			inputs=self.x,
			filters=96,
			strides=4,
			kernel_size=[11, 11],
			kernel_initializer=tf.constant_initializer(self.pre_trained_model["conv1"][0]),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
			bias_initializer=tf.constant_initializer(self.pre_trained_model["conv1"][1]),
			padding="VALID",
			activation=tf.nn.relu, name='conv1')
		pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3, 3], padding="VALID", strides=2, name='pool1')
		norm1 = tf.nn.lrn(pool1, depth_radius=2, bias=1.0, alpha=2.0 / 100000.0, beta=0.75,
						  name='norm1')

		conv2 = tf.layers.conv2d(
			inputs=pool1,
			filters=256,
			strides=1,
			kernel_size=[5, 5],
			kernel_initializer=tf.constant_initializer(self.pre_trained_model["conv2"][0]),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
			bias_initializer=tf.constant_initializer(self.pre_trained_model["conv2"][1]),
			padding="same",
			activation=tf.nn.relu, name='conv2')
		pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3, 3], padding="valid", strides=2, name='pool2')
		norm2 = tf.nn.lrn(pool2, depth_radius=2, bias=1.0, alpha=2.0 / 100000.0, beta=0.75, name='norm2')

		conv3 = tf.layers.conv2d(
			inputs=norm2,
			filters=384,
			strides=1,
			kernel_size=[3, 3],
			kernel_initializer=tf.constant_initializer(self.pre_trained_model["conv3"][0]),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
			bias_initializer=tf.constant_initializer(self.pre_trained_model["conv3"][1]),
			padding="SAME",
			activation=tf.nn.relu, name='conv3')
		conv4 = tf.layers.conv2d(
			inputs=conv3,
			filters=384,
			strides=1,
			kernel_size=[3, 3],
			kernel_initializer=tf.constant_initializer(self.pre_trained_model["conv4"][0]),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
			bias_initializer=tf.constant_initializer(self.pre_trained_model["conv4"][1]),
			padding="SAME",
			activation=tf.nn.relu, name='conv4')
		conv5 = tf.layers.conv2d(
			inputs=conv4,
			filters=256,
			strides=1,
			kernel_size=[3, 3],
			kernel_initializer=tf.constant_initializer(self.pre_trained_model["conv5"][0]),
			kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
			bias_initializer=tf.constant_initializer(self.pre_trained_model["conv5"][1]),
			padding="SAME",
			activation=tf.nn.relu, name='conv5')
		pool5 = tf.layers.max_pooling2d(inputs=conv5, pool_size=[3, 3], padding="VALID", strides=2, name='pool5')

		pool_fc_flat = tf.reshape(pool5, [-1, 6 * 6 * 256])

		fc6 = tf.layers.dense(inputs=pool_fc_flat, units=4096,
							  kernel_initializer=tf.constant_initializer(self.pre_trained_model["fc6"][0]),
							  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
							  bias_initializer=tf.constant_initializer(self.pre_trained_model["fc6"][1]),
							  activation=tf.nn.relu, name='fc6')
		fc6_drop = tf.nn.dropout(fc6, self.keep_prob)

		fc7 = tf.layers.dense(inputs=fc6, units=4096,
							  kernel_initializer=tf.constant_initializer(self.pre_trained_model["fc7"][0]),
							  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
							  bias_initializer=tf.constant_initializer(self.pre_trained_model["fc7"][1]),
							  activation=tf.nn.relu, name='fc7')
		fc7_drop = tf.nn.dropout(fc7, self.keep_prob)
		self.fc8 = tf.layers.dense(inputs=fc7_drop, units=30,
							  # kernel_initializer= tf.constant_initializer(pre_trained_model["fc8"][0]),
							  # bias_initializer = tf.constant_initializer(pre_trained_model["fc8"][1]),
							  kernel_regularizer=tf.contrib.layers.l2_regularizer(0.0005),
							  activation=tf.nn.relu, name='fc8')




def get_data(is_test=False):
	data_path = "train"
	if is_test:
		data_path = "test"
	return pd.read_csv('dataset/book30-listing-'+data_path+'.csv', encoding="ISO-8859-1",
				names=['ASIN', 'FILENAME', 'IMAGE_URL', 'TITLE', 'AUTHOR', 'CATEGORY_ID', 'CATEGORY'])


def loadImages(fnames,is_test):
	path = os.getcwd() + "/dataset/train_images/"
	if is_test:
		path = os.getcwd() + "/dataset/test_images/"
	loadedImages = []
	for image in fnames:
		img = Image.open(path + image)
		loadedImages.append(img)
	return loadedImages


def get_pixels(fnames,is_test):
	imgs = loadImages(fnames, is_test)
	pixel_list = []
	for img in imgs:
		img = img.resize((227, 227), Image.ANTIALIAS)
		arr = array(img, dtype="float32")
		pixel_list.append(list(arr))
	return np.array(pixel_list)


def label_from_category(category_id=None):
	label_list = np.zeros(30)
	label_list[category_id] = 1
	return list(label_list)


def features_from_data(data, is_test=False):
	pixels = get_pixels(data.FILENAME, is_test)
	labels = np.array(data["CATEGORY_ID"].apply(label_from_category).tolist())
	return pixels, labels


def train():

	with tf.Session() as sess:
		x_ = tf.placeholder(tf.float32, [None, 227, 227, 3])
		x_image = tf.reshape(x_, [-1, 227, 227, 3])
		y_ = tf.placeholder(tf.float32, [None, 30])
		keep_prob = tf.placeholder(tf.float32)
		model = PretrainedAlexNet(images=x_image,_keep_prob=keep_prob)

		global_step = tf.get_variable("global_step",initializer=0)
		lr = tf.train.exponential_decay(FLAGS.starting_learning_rate, global_step, NUM_EPOCHS_PER_LEARNING_RATE_DECAY,
										LEARNING_RATE_DECAY, staircase=True)
		regularization_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

		cross_entropy = (tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=model.fc8))+
						 regularization_loss)

		train_step  = tf.train.MomentumOptimizer(lr, WEIGHT_DECAY_FACTOR).minimize(cross_entropy,global_step=global_step)

		correct_prediction = tf.equal(tf.argmax(model.fc8, 1), tf.argmax(y_, 1))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

		sess.run(tf.global_variables_initializer())


		data_list = np.array_split(get_data(),TRAINING_SIZE)

		saver = tf.train.Saver(var_list=tf.trainable_variables())
		ckpt = tf.train.get_checkpoint_state(FLAGS.train_models)
		if ckpt and ckpt.model_checkpoint_path:
			saver.restore(sess, ckpt.model_checkpoint_path)
			print("restore done , global_step = %d" % sess.eval(global_step) )#tf.train.global_step(sess, global_step))

		from datetime import datetime
		start_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')

		for i in range(sess.eval(global_step)+1,FLAGS.iterations):
			x, train_labels = features_from_data(data_list[i % TRAINING_SIZE])
			train_step.run(feed_dict={x_: x, y_: train_labels, keep_prob: 0.5})

			current_time_str = datetime.now().strftime('%Y-%m-%d %H:%M')

			if i % 100 == 0 :
				train_accuracy = accuracy.eval(feed_dict={
					x_: x, y_: train_labels, keep_prob: 1.0})
				print("step %d, training accuracy %g, current_:%s , start:%s , loss = %f , "
					  "global_step = %d, learning_rate=%f " % (i, train_accuracy, current_time_str,
					start_time_str, sess.run(cross_entropy, feed_dict={x_: x, y_: train_labels, keep_prob: 1.0})[1],
					sess.run(global_step), sess.run(lr)))
			if i % 5000 == 0:
				model_path = os.path.join(FLAGS.train_models,'model.ckpt')
				saver.save(sess,model_path,global_step=global_step)
				print("checkpoint saved")

		model_path = os.path.join(FLAGS.train_models, 'model.ckpt')
		saver.save(sess,model_path,global_step=global_step)


def main(argv=None):
	if not tf.gfile.Exists(FLAGS.train_models):
		tf.gfile.MakeDirs(FLAGS.train_models)
	train()

if __name__ == "__main__":
	tf.app.run()
