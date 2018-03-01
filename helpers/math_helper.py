import numpy as np

def sigmoid(x):
	return (1/(1 + np.exp(-x)))

def sigmoidPrime(x):
	sigx = sigmoid(x)
	return sigx * (1 - sigx)