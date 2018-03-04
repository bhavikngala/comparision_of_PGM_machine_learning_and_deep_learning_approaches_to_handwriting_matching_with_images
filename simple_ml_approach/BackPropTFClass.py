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

	# feed forward the input and compute output
	def feedForward(self, x):
		z1 = tf.nn.sigmoid(tf.matmul(x, self.w1) + self.b1)
		yHat = tf.matmul(z1, self.w2) + self.b2
		return yHat

	# training the network
	def trainNetwork(self, xTrain, yTrain, xVali, yVali, epochs,
		learningRate, miniBatchSize):
		# input output placeholder Variables
		x = tf.placeholder(tf.float32, [None, 768])
		y = tf.placeholder(tf.float32, [None, 2])

		# feed forward
		yHat = feedForward(x)
		# class of prediction
		predict = tf.argmax(yHat, axis=1)

		# cross entropy cost function
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=y, logits=yHat))
		# weight updates
		updates = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

		# session object
		sess = tf.Session()
		# initiablize variables
		init = tf.global_variables_initializer()
		# run init
		sess.run(init)

		# run epochs for training
		for epoch in range(epochs):
			# train in minibatches
			for i in range(xTrain.shape[0]/miniBatchSize)
				# sample indices in minibatch
				lowerBound = i*miniBatchSize
				upperBound = min((i+1)*miniBatchSize, xTrain.shape[0])

				# update weights on minibatch
				sess.run(updates, feed_dict={x:xTrain[lowerBound:upperBound],
					y:yTrain[lowerBound:upperBound]})

			# evaluate on validation set
			if xVali is not None and yVali is not None:
				if epoch % 100 == 0 or (epoch % 10 == 0 and epoch < 100):
					# compute test accuracy
					test_accuracy = np.mean(np.argmax(yVali, axis=1) == sess.run(predict, feed_dict={x:xVali, y:yVali}))

					print('Epoch = %05d, test accuracy = %.2f%%' % (epoch+1, 100. * test_accuracy))