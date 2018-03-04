import numpy as np
import random
import sift_feature_extractor as sfe
from backprop import BackPropNN

# filename of vector added SIFT descriptors
vectorAddedSIFTDescriptorsFile = './../data/vectorAddedSIFTDescriptorsDictionary.npy'
# filename of clusters of SIFT descriptors
clustersOfSIFTDescriptorsFile = './../data/centroidsOfSIFTDescriptorsDictionary_3.npy'

learningRate = 0.5
epochs = 5000
miniBatchSize = 500

# function trains the network with given input-output and network size
# input is specified in the siftDescriptorFile which is the file name
def fitNetWorkWithSIFTDescriptorRepresentation(siftDescriptorFile, numPairs, networkSize):
	print('~~~~~~~inside main:::function:::fitNetWorkWithSIFTDescriptorRepresentation')

	# form same writer different writer pairs for input to network
	inputs, outputs = sfe.generateInputOutputDataset(siftDescriptorFile, numPairs)

	# shuffle randomly
	inputs, outputs = shuffleLists(inputs, outputs)

	# converting to np.array
	inputs = np.array(inputs)
	outputs = np.array(outputs)

	# total number of inputs
	numInputs = len(inputs)
	
	# initialize network
	feedForwardNN = BackPropNN(networkSize)

	# describe network
	feedForwardNN.describeNetwork()

	# train network with validation data
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
	print('~~~~~~~inside main:::function:::main')

	# train network using vector added SIFT descriptors
	# fitNetWorkWithSIFTDescriptorRepresentation(vectorAddedSIFTDescriptorsFile, 10, [256, 128, 2])

	# train network using clusters of SIFT descriptors
	fitNetWorkWithSIFTDescriptorRepresentation(clustersOfSIFTDescriptorsFile, 10, [768, 256, 2])

if __name__ == "__main__":
	main()