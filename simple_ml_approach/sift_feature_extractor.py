import cv2
import numpy as np
from scipy.cluster.vq import kmeans2
import os

# directory of images
dataDir = './../../AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
# filename of SIFT descriptors
siftDescriptorsFile = './../data/SIFTDescriptorDictionary.npy'
vectorAddedSIFTDescriptorsFile = './../data/vectorAddedSIFTDescriptorsDictionary.npy'
centroidsOfSIFTDescriptorsFile = './../data/centroidsOfSIFTDescriptorsDictionary_3.npy'

# function extracts all the SIFT features of all the images in a directory
# stores the features in a dictionary
# stores the dictionary in a file for later use
def extractSIFTFeatures():
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

def main():
	# extract SIFT features from batch of images
	# extractSIFTFeatures()

	# vector addition of descriptors in an image
	# vectorAdditionOfSIFTDescriptors()

	# cluster SIFT descriptors
	clusterDescriptorsUsingKmeans(3)

if __name__ == "__main__":
	main()