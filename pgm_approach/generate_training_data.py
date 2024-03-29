import csv
import numpy as np
import random

# filename with path of the features file
filename = './../data/pgm/features_pgm.csv'

# read the features in dictionary
# authorname is the key
# list of features is the value
def readDataInDict(filename=None):
	# initialize empty dictioanary
	dataDict = {}

	if filename is None:
		filename = './../data/pgm/features_pgm.csv'

	# open csv file, read into list of rows
	with open(filename, 'r') as csvfile:
		# header to ignore header row
		header = True
		reader = csv.reader(csvfile)

		# loop each row of the file
		for row in reader:
			# ignore header row
			if header:
				header = False
				continue

			# read authore name from the row, delete last character
			authorName = row[0]
			authorName = authorName[:-1]

			# initialize empty list
			features = []
			# if authorname is  present in dict then read it into features
			if authorName in dataDict:
				features = dataDict.get(authorName)

			# update features list and update dictionary
			features.append(np.array(row[1:], dtype='int8'))
			dataDict.update({authorName:features})

	return dataDict

# form same writer, different writer features pairs
def formSameWriterDiffWriterInputOutputFeaturePairs(numPairs, matchH0H1Prior,
	filename=None):
	# read the features data into dictionary
	dataDict = readDataInDict(filename)

	# initialisze empty lists for same writer pairs and different writer pairs separately
	sameWriterInputPairs = []
	sameWriterOutputPairs = []
	diffWriterInputPairs = []
	diffWriterOutputPairs = []

	# fetch all the keys in a list
	keysList = [*dataDict]

	# iterate over key value pairs in the dictionary to form pairs
	for k, v in dataDict.items():
		# number of feature lists for a writer
		numFeatureLists = len(v)

		# if > 1 then form same writer pairs
		if numFeatureLists > 1:
			# random sampling of images to make pairs
			# max number of pairs = 5
			l1 = random.sample(range(numFeatureLists), min(numFeatureLists, numPairs))
			l2 = random.sample(range(numFeatureLists), min(numFeatureLists, numPairs))
			random.shuffle(l2)

			# forming pairs
			for (i1, i2) in zip(l1, l2):
				if i1 != i2:
					sameWriterInputPairs.append(np.concatenate(
						[v[i1].flatten(), v[i2].flatten()]))
					sameWriterOutputPairs.append([1])

		# fetching 5 writers from the keysList
		negativeSamples = [keysList[i1] for i1 in random.sample(range(len(keysList)), numPairs)]
		
		# iterating to form different writer pairs
		for key in negativeSamples:
			if key != k:
				# getting descriptors list for different writer
				featureList = dataDict.get(key)
				
				# randomly sampling an image for each writer
				i1 = random.randint(0, len(featureList)-1)
				i2 = random.randint(0, len(v)-1)
				
				# forming pairs
				diffWriterInputPairs.append(
					np.concatenate([v[i2].flatten(),
						featureList[i1].flatten()]))
				diffWriterOutputPairs.append([0])

	# shuffle data
	sameWriterInputPairs, sameWriterOutputPairs = \
		shuffleLists(sameWriterInputPairs, sameWriterOutputPairs)
	diffWriterInputPairs, diffWriterOutputPairs = \
		shuffleLists(diffWriterInputPairs, diffWriterOutputPairs)

	if matchH0H1Prior:
		# take min of same and different pairs
		p = min(len(sameWriterInputPairs), len(diffWriterInputPairs))

		# only use p number of pairs from same and diff pairs list
		# this is done to keep the prior prob of H0 and H1 to be
		# equal to half
		trainingInputs = np.concatenate((sameWriterInputPairs[:(p-10)],
			diffWriterInputPairs[:(p-10)]), axis=0)
		trainingOutputs = np.concatenate((sameWriterOutputPairs[:(p-10)],
			diffWriterOutputPairs[:(p-10)]), axis=0)

		testingInputs = np.concatenate((sameWriterInputPairs[(p-10):], diffWriterInputPairs[(p-10):]), axis=0)
		testingOutputs = np.concatenate((sameWriterOutputPairs[(p-10):],
			diffWriterOutputPairs[(p-10):]), axis=0)
	else:
		l1 = len(sameWriterInputPairs)
		l2 = len(diffWriterInputPairs)
		trainingInputs = np.concatenate((sameWriterInputPairs[:int(0.8*l1)],
			diffWriterInputPairs[:int(0.8*l2)]), axis=0)
		trainingOutputs = np.concatenate((sameWriterOutputPairs[:int(0.8*l1)],
			diffWriterOutputPairs[:int(0.8*l2)]), axis=0)

		testingInputs = np.concatenate((sameWriterInputPairs[int(0.8*l1):], diffWriterInputPairs[int(0.8*l2):]), axis=0)
		testingOutputs = np.concatenate((sameWriterOutputPairs[int(0.8*l1):],
			diffWriterOutputPairs[int(0.8*l2):]), axis=0)

	return trainingInputs, trainingOutputs, testingInputs, testingOutputs

# shuffle data
def shuffleLists(*lists):
	# pack the lists using zip
	packedLists = list(zip(*lists))
	# shuffle the packed list
	random.shuffle(packedLists)
	# return the lists by unpacking them
	return zip(*packedLists)

def main():
	inputs, outputs = formSameWriterDiffWriterInputOutputFeaturePairs(5)
	print("total pairs:", inputs.shape[0])
	print("same writer pairs:", np.sum(outputs[:] == [1]))
	print("same writer paits:", np.sum(outputs[:] == [0]))

if __name__ == "__main__":
	main()