import csv

def readCSVIntoListAsDict(filename):
	data = []

	with open(filename, 'rb') as csvfile:
		reader = csv.DictReader(csvfile)
		for row in reader:
			data.append(row)

	return data

def readCSVIntoList(filename):
	data = []

	with open(filename, 'r') as csvfile:
		reader = csv.reader(csvfile)

		firstLine = True
		for row in reader:
			if firstLine:
				firstLine = False
				continue
				
			data.append(row)

	return data

def readPairNamesInList(filename):
	# read csv file for pairs data
	pairNamesList = []

	with open(filename, 'r') as csvfile:
		# ignore header row
		header = True
		reader = csv.reader(csvfile)

		# loop each row of the file
		for row in reader:
			# ignore header row
			if header:
				header = False
				continue

			pairNamesList.append([row[1].split('.')[0], row[2].split('.')[0]])

	return pairNamesList

def readWriterFeaturesInDict(filename):
	# read the features in the CSV for an img in dictionary
	writerFeaturesDict = {}

	with open(filename, 'r') as csvfile:
		# ignore header row
		header = True
		reader = csv.reader(csvfile)

		# loop each row of the file
		for row in reader:
			# ignore header row
			if header:
				header = False
				continue

			# converting the type of feature values to integer and storing in dictionary as list
			writerFeaturesDict.update({row[1]:[int(x) for x in row[2:]]})

	return writerFeaturesDict

def saveToCSVFile(filename, outputList):
	with open(filename, 'w') as csvfile:
		wr = csv.writer(csvfile, delimiter=',')
		
		for row in outputList:
			wr.writerow(row)