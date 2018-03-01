import cv2
import numpy as np
import os

def extractSIFTFeatures():
	dataDir = './../../AND_dataset/Dataset[Without-Features]/AND_Images[WithoutFeatures]/'
	
	imgFilenames = os.listdir(dataDir)
	sift = cv2.xfeatures2d.SIFT_create()
	flen = 100000
	
	print('number of images:', len(imgFilenames))

	for imgFileName in imgFilenames:
		img = cv2.imread(dataDir + '/' + imgFileName, 0)
	
		kp, desc = sift.detectAndCompute(img, None)
		if len(kp) < flen:
			flen = len(kp)

		if len(kp) < 10:
			print(imgFileName, ':::', len(kp))

	print('min number of descriptors:', flen)

def main():
	extractSIFTFeatures()

if __name__ == "__main__":
	main()