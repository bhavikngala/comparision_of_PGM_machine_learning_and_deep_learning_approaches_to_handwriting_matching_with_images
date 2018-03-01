import cv2
import numpy as np
import os

# directory of images
dataDir = './../../AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
# filename of SIFT descriptors
siftDescriptorsFile = './../data/SIFTDescriptorDictionary.npy'

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

def main():
	extractSIFTFeatures()

if __name__ == "__main__":
	main()