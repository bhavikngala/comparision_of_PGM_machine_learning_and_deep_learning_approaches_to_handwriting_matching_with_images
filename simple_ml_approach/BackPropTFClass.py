import tensorflow as tf
import numpy as np
import sift_feature_extractor as sfe

class BackProp:

	# initialize variables
	def __init__(self, layerSizes, modelCheckPointFileName=None):
		# layerSizes
		self.layerSizes = layerSizes

		# weight prototype, name attribute for saving weight
		self.w1 = tf.Variable(tf.truncated_normal([layerSizes[0],
			layerSizes[1]]), name='w1')
		# bias prototype, name attribute for saving bias
		self.b2 = tf.Variable(tf.truncated_normal([1, layerSizes[1]]),
			name='b1')

		# weight prototype, name attribute for saving weight
		self.w2 = tf.Variable(tf.truncated_normal([layerSizes[1],
			layerSizes[2]]), name='w2')
		# bias prototype, name attribute for saving bias
		self.b2 = tf.Variable(tf.truncated_normal([1, layerSizes[2]]),
			name='b2')

		# class scope session object
		self.sess = tf.Session()
		self.sess.run(tf.global_variables_initializer())

		# if model checkpoint filename is provided then  restore
		# parameters from file
		if modelCheckPointFileName is not None:
			restoreNetwork(modelCheckPointFileName)

	# feed forward the input and compute output
	def feedForward(self, x):
		z1 = tf.nn.sigmoid(tf.matmul(x, self.w1) + self.b1)
		yHat = tf.matmul(z1, self.w2) + self.b2
		return yHat

	# training the network
	def trainNetwork(self, epochs, learningRate, miniBatchSize, xTrain, yTrain, xVali=None, yVali=None):
		# input output placeholder Variables
		x = tf.placeholder(tf.float32, [None, self.layerSizes[0]])
		y = tf.placeholder(tf.float32, [None, self.layerSizes[2]])

		# feed forward
		yHat = feedForward(x)
		# class of prediction
		predict = tf.argmax(yHat, axis=1)

		# cross entropy cost function
		cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
			labels=y, logits=yHat))
		# weight updates
		updates = tf.train.GradientDescentOptimizer(learningRate).minimize(cost)

		# run epochs for training
		for epoch in range(epochs):
			# train in minibatches
			for i in range(xTrain.shape[0]/miniBatchSize)
				# sample indices in minibatch
				lowerBound = i*miniBatchSize
				upperBound = min((i+1)*miniBatchSize, xTrain.shape[0])

				# update weights on minibatch
				self.sess.run(updates, feed_dict={x:xTrain[lowerBound:upperBound], y:yTrain[lowerBound:upperBound]})

			# evaluate on validation set
			if xVali is not None and yVali is not None:
				if epoch % 100 == 0 or (epoch % 10 == 0 and epoch < 100):
					# compute test accuracy
					test_accuracy = np.mean(np.argmax(yVali, axis=1) == self.sess.run(predict, feed_dict={x:xVali, y:yVali}))

					print('Epoch = %05d, test accuracy = %.2f%%' % (epoch+1, 100. * test_accuracy))

	# predict class:
	def predict(self, xInput):
		x = tf.placeholder(tf.float32, [None, 768])
		yHat = feedForward(x)
		# prediction
		predict = tf.argmax(yHat, axis=1)
		
		# run prediction
		self.sess.run(predict, feed_dict={x:xInput})
		return predict

	# evaluate network
	def evaluateNetwork(self, xInput, yInput):
		x = tf.placeholder(tf.float32, [None, 768])
		yHat = feedForward(x)
		# prediction
		predict = tf.argmax(yHat, axis=1)

		# compute accuracy
		accuracy = np.mean(np.argmax(yInput, axis=1) == self.sess.run(predict, feed_dict={x:xInput}))
		print('Network accuracy = %.2f%%' % (100. * accuracy))

	# save network weights and bias
	def saveNetwork(self, checkpointFilename):
		# saver object to save parameters to file
		saver = tf.train.Saver([self.w1, self.w2, self.b1, self.b2])
		saver.save(self.sess, checkpointFilename)

	# restore network from a checkpoint
	def restoreNetwork(self, checkpointFilename):
		# saver object to restore parameters to file
		saver = tf.train.Saver([self.w1, self.w2, self.b1, self.b2])
		saver.restore(self.sess, checkpointFilename)