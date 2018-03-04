import tensorflow as tf
import numpy as np
import sift_feature_extractor as sfe

class BackProp:

	# initialize variables
	def __init__(self, layerSizes):

		self.w1 = tf.Variable(tf.truncated_normal([layerSizes[0],
			layerSizes[1]]))
		self.b2 = tf.Variable(tf.truncated_normal([1, layerSizes[1]]))

		self.w2 = tf.Variable(tf.truncated_normal([layerSizes[1],
			layerSizes[2]]))
		self.b2 = tf.Variable(tf.truncated_normal([1, layerSizes[2]]))

	def feedForward(self, x):
		z1 = tf.nn.sigmoid(tf.matmul(x, self.w1) + self.b1)
		yHat = tf.matmul(z1, self.w2) + self.b2
		return yHat