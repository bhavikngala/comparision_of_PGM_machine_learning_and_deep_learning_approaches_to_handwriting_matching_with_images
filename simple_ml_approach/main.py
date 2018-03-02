import numpy as np
import random
from simple_ml_approach import sift_feature_extractor as sfe
from simple_ml_approach import BackPropNN

# filename of vector added SIFT descriptors
vectorAddedSIFTDescriptorsFile = './../data/vectorAddedSIFTDescriptorsDictionary.npy'

learningRate = 0.5
epochs = 1000
miniBatchSize = 100

def fitVectorAddedSIFTDescriptorsData():
	print('~~~~~~~~inside main:::function:::fitVectorAddedSIFTDescriptorsData')

	# form same writer different writer pairs for input to network
	inputs, outputs = sfe.generateInputOutputDataset(vectorAddedSIFTDescriptorsFile, 10)

	# shuffle randomly
	inputs, outputs = shuffleLists(inputs, outputs)

	numInputs = len(inputs)
	inpuVectorLength = inputs[0].shape[0]

	# intialize network
	# [256, 128, 2]
	feedForwardNN = BackPropNN([inputVectorLength, inputVectorLength/2, 2])

	# describe network
	feedForwardNN.describeNetwork()

	# train network
	feedForwardNN.train(inputs[0:int(numInputs * 0.8)],
		outputs[0:int(numInputs * 0.8)], learningRate, epochs, miniBatchSize,
		inputs[int(numInputs * 0.8):int(numInputs * 0.9)],
		outputs[int(numInputs * 0.8):int(numInputs * 0.9)])

# shuffle input lists
def shuffleLists(*lists):
	# pack the lists using zip
	packedLists = list(zip(*lists))
	# shuffle the packed list
	random.shuffle(packedLists)
	# return the lists by unpacking them
	return zip(*packedLists)


def main():
	print('~~~~~~~~inside main:::function:::main')

	# train network using vector added SIFT descriptors
	fitVectorAddedSIFTDescriptorsData()

if __name__ == "__main__":
	main()