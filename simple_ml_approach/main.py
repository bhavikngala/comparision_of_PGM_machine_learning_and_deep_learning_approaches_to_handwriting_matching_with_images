import argparse
import numpy as np
import sift_feature_extractor as sfe
from BackPropTFClass import BackProp

import sys
sys.path.insert(0, './../helpers')
import file_helper as postman

def main(args):
	# read the pair names in the list
	pairNamesList = postman.readPairNamesInList(
		args['filepath'] + '/MLTestPairs.csv')

	# extract sift features, cluster them, store in dictionary
	siftFeatureClusterDict = sfe.readImagesExtractAndClusterSift(
		args['filepath'] + '/MLTestData/')

	# network parameters
	networkSize = [768, 256, 2]
	# model checkpoint filename to restore parameters
	weightFileName = './tmp/model.ckpt'

	# initialize network
	# provide model checkpoint name in constructor to restore weights, baises
	feedForwardNN = BackProp(networkSize, weightFileName)

	# 2D list for saving output in csv
	output = []
	output.append(['', 'FirtImage', 'SecondImage', 'SameOrDifferent'])

	for i in range(len(pairNamesList)):
		# getting features for writer one
		writer1Features = siftFeatureClusterDict.get(pairNamesList[i][0])
		writer2Features = siftFeatureClusterDict.get(pairNamesList[i][1])

		xInput = np.reshape(
			np.concatenate([writer1Features, writer2Features]),
			[1,768])

		prediction = feedForwardNN.predict(xInput)

		# prediction is subtracted from 1 because during training the network
		# same writer class was assigned [1, 0] i.e. 0 value
		# and [0, 1] i.e 1 was assigned for different class
		# which is opposite to the output required for the test data
		output.append([str(i), pairNamesList[i][0], pairNamesList[i][1], 
			str(1 - prediction[0])])

	# save the output list to csv file
	postman.saveToCSVFile(args['filepath'] + '/MLTestOutput.csv', output)

if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--filepath", required=True,
		help="path of the test data folder")
	args = vars(ap.parse_args())

	main(args)