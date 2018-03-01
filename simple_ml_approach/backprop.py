import numpy as np

import sys
sys.path.insert(0, './../helpers')

import math_helper as misc

class BackPropNN:

	def __init__(self, layerSizes):
		self.numLayers = len(layerSizes)

		self.weights = [(2 * 0.05 * np.random.rand(i, j) - 0.05) for i,j in zip(layerSizes[:-1], layerSizes[1:])]
		self.biases = [(2 * 0.05 * np.random.rand(1, i) - 0.05) for i in layerSizes[1:]]

		for weight in self.weights:
			print('sum of weight:', np.sum(weight))
		for bias in self.biases:
			print('sum of bias:', np.sum(bias))

	def feedForward(self, x):
		a = []
		a.append(x)
		for i in range(1, self.numLayers):
			z = np.dot(a[i-1], self.weights[i-1]) + self.biases[i-1]
			a.append(misc.sigmoid(z))
		# for aa in a:
		# 	print('sum of a:', np.sum(aa))	
		# quit()
		
		return a

	def computeErrorInNeurons(self, a, y):
		# consider there is only one input
		# np.dot function will compute deltas for each input and each neuron 
		deltas = [None] * (self.numLayers - 1)
		deltas[-1] = (a[-1] - y) * (a[-1] * (1 - a[-1]))

		for i in range(2, self.numLayers):
			delta = (np.dot(deltas[-i+1], self.weights[-i + 1].T)) * (a[-i] * (1 - a[-i]))
			deltas[-i] = delta
		# for delta in deltas:
		# 	print('sum of delta:', np.sum(delta))
		# quit()

		return deltas

	def computeGradientInCostFuncWRTWeightsAndBiases(self, a, deltas):
		gradientsWRTWeights = [np.zeros(np.shape(w)) for w in self.weights]
		
		gradientIndex = 0
		for a1, delta in zip(a[:-1], deltas):
			for a1_row, detla_row in zip(a1, delta):
				gradientsWRTWeights[gradientIndex] = gradientsWRTWeights[gradientIndex] + np.reshape(a1_row, [a1_row.shape[0], 1]) * detla_row
			gradientsWRTWeights[gradientIndex] = gradientsWRTWeights[gradientIndex] / a1.shape[0]
			gradientIndex += 1

		deltas = [np.reshape(np.mean(delta, axis=0), [1, delta.shape[1]]) for delta in deltas]

		return gradientsWRTWeights, deltas

	def updateWeights(self, gradientsWRTWeights, gradientsWRTBiases, learningRate):
		self.weights = [(w - (learningRate * gradientsWRTWeight)) for w, gradientsWRTWeight in zip(self.weights, gradientsWRTWeights)]
		self.biases = [(b - (learningRate * gradientsWRTBias)) for b, gradientsWRTBias in zip(self.biases, gradientsWRTBiases)]

	def train(self, x, y, learningRate, epochs, miniBatchSize, vali_x=None, vali_y=None):
		for epoch in range(epochs):
			for i in range(int(x.shape[0]/miniBatchSize)):
				lowerBound = i*miniBatchSize
				upperBound = min((i+1)*miniBatchSize, x.shape[0])

				a = self.feedForward(x[lowerBound:upperBound, :])
				deltas = self.computeErrorInNeurons(a, y[lowerBound:upperBound, :])
				[gradientsWRTWeights, gradientsWRTBiases] = self.computeGradientInCostFuncWRTWeightsAndBiases(a, deltas)
				self.updateWeights(gradientsWRTWeights, gradientsWRTBiases, learningRate)

			if vali_x is not None and vali_y is not None:
				if epoch % 100 == 0 or epoch % 10 == 0:
					print('epoch:', format(epoch, '05d'), 'classification error:', self.evaluateNetwork(vali_x, vali_y))

	def evaluateNetwork(self, x, y):
		t = self.feedForward(x)
		return np.sum(np.argmax(t[-1], axis=1) != np.argmax(y, axis=1))/y.shape[0]

	def describeNetwork(self):
		print('Number of layers in network:', self.numLayers)

		for i in range(0, self.numLayers - 1):
			print('layer', str(i+1), 'weight:', np.shape(self.weights[i]), 'bias:', np.shape(self.biases[i]))