import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
import random
import os

# directory of images
dataDir = './../../AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
# filename of SIFT descriptors
siftDescriptorsFile = './../data/SIFTDescriptorDictionary.npy'
# filename of vector added SIFT descriptors
vectorAddedSIFTDescriptorsFile = './../data/vectorAddedSIFTDescriptorsDictionary.npy'
# filename of centroids of SIFT descriptors
centroidsOfSIFTDescriptorsFile = './../data/centroidsOfSIFTDescriptorsDictionary_3.npy'

# function extracts all the SIFT features of all the images in a directory
# stores the features in a dictionary
# stores the dictionary in a file for later use
def extractSIFTFeatures():
	print('\n~~~~~~~inside sift_feature_extractor:::function:::extractSIFTFeatures\n')
	# dictionary to store descriptors with filenames
	descriptorDict = {}
	
	# list of all filenames in the directory
	imgFilenames = os.listdir(dataDir)

	# SIFT feature extractor
	sift = cv2.xfeatures2d.SIFT_create()

	# looping over all the images in the directory
	# extracting SIFT features
	# storing the feature in the dictionary
	for imgFileName in imgFilenames:
		# reading image as grayscale image
		img = cv2.imread(dataDir + '/' + imgFileName, 0)
	
		# extracting keypoints and descriptors of image
		_, desc = sift.detectAndCompute(img, None)

		writerID = imgFileName[0:4]
		# initialize list or get existing list from dict
		if writerID not in descriptorDict:
			descList = [desc]
		else:
			descList = descriptorDict.get(writerID)
			descList.append(desc)

		# update dictionary
		descriptorDict.update({writerID:descList})

	# store dictionary in file
	np.save(siftDescriptorsFile, descriptorDict)

# perform vector addition of all descriptors in an image
def vectorAdditionOfSIFTDescriptors():
	print('\n~~~~~~~inside sift_feature_extractor:::function:::vectorAdditionOfSIFTDescriptors\n')
	# read the descriptors stored in file
	descriptorDict = np.load(siftDescriptorsFile).item()

	# iterate over the descriptors, add them, normalize them
	for k, v in descriptorDict.items():
		vectorAddedSIFTDescriptors = []

		for descArray in v:
			# adding desccriptors of a single image
			resultantVector = np.sum(descArray, axis=0)
			# normalizing the vector
			resultantVector = resultantVector / np.linalg.norm(resultantVector, ord=2)

			# appending the resultant vector to list
			vectorAddedSIFTDescriptors.append(resultantVector)

		descriptorDict.update({k:vectorAddedSIFTDescriptors})

	# save dictionary to file
	np.save(vectorAddedSIFTDescriptorsFile, descriptorDict)

# clustering the descriptors in the image using Kmeans
def clusterDescriptorsUsingKmeans(numCluster):
	print('\n~~~~~~~inside sift_feature_extractor:::function:::clusterDescriptorsUsingKmeans\n')
	# read the descriptors stored in file
	descriptorDict = np.load(siftDescriptorsFile).item()

	# iterate over the descriptors, find their cluster centroids
	for k, v in descriptorDict.items():
		centroidsOfSIFTDescriptors = []

		# looping over descriptors of each image, clustering them
		for descArray in v:
			# clustering the descriptors
			centroids, _ = kmeans2(descArray, numCluster, minit='points')
			# normalizing the centroids
			centroidMag = np.reshape(np.linalg.norm(centroids, axis=1, ord=2),
				[-1, 1])
			centroids = centroids/centroidMag

			# appending the resultant vector to list
			centroidsOfSIFTDescriptors.append(centroids)

		descriptorDict.update({k:centroidsOfSIFTDescriptors})

	# save dictionary to file
	np.save(centroidsOfSIFTDescriptorsFile, descriptorDict)

# generate inputs and output labels from the extracted SIFT descriptors
def generateInputOutputDataset(descriptorsFile):
	print('\n~~~~~~~inside sift_feature_extractor:::function:::generateInputOutputDataset\n')
	# lists to store inputs and outputs
	inputs = []
	outputs = []

	# read the descriptor file in dictionary
	descriptorDict = np.load(descriptorsFile).item()

	# fetch all the keys in a list
	keysList = [*descriptorDict]

	# iterating all the keys in the dictionary
	# forming same writer/different writer pairs
	for k, v in descriptorDict.items():
		# number of images for a writer
		numImages = len(v)
		# if > 1 then form same writer pairs
		if numImages > 1:
			# random sampling of images to make pairs
			# max number of pairs = 5
			l1 = random.sample(range(numImages), min(numImages, 5))
			l2 = random.sample(range(numImages), min(numImages, 5))
			random.shuffle(l2)

			# forming pairs
			for (i1, i2) in zip(l1, l2):
				if i1 != i2:
					inputs.append(np.concatenate([v[i1].flatten(),
						v[i2].flatten()]))
					outputs.append([1,0])

		# fetching 5 writers from the keysList
		negativeSamples = [keysList[i1] for i1 in random.sample(range(len(keysList)), 5)]
		# iterating to form different writer pairs
		for key in negativeSamples:
			# getting descriptors list for different writer
			descList = descriptorDict.get(key)
			# randomly sampling an image for each writer
			i1 = random.randint(0, len(descList)-1)
			i2 = random.randint(0, len(v)-1)
			# forming pairs
			inputs.append(np.concatenate([v[i2].flatten(),
				descList[i1].flatten()]))
			outputs.append([0, 1])

	print('num Inputs:', len(inputs))
	print('length of input:', inputs[0].shape)
	print('same writer:', outputs.count([1,0]), 'different writer:',
		outputs.count([0,1]))
	return inputs, outputs

def main():
	print('\n~~~~~~~inside sift_feature_extractor:::function:::main\n')
	# extract SIFT features from batch of images
	# extractSIFTFeatures()

	# vector addition of descriptors in an image
	# vectorAdditionOfSIFTDescriptors()

	# cluster SIFT descriptors
	# clusterDescriptorsUsingKmeans(3)

	# test run for generating input, output for network
	generateInputOutputDataset(centroidsOfSIFTDescriptorsFile)

if __name__ == "__main__":
	main()