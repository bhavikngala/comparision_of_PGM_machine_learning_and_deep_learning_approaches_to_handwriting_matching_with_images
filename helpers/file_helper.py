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

	with open(filename, 'rb') as csvfile:
		reader = csv.reader(csvfile)

		firstLine = True
		for row in reader:
			if firstLine:
				firstLine = False
				continue
				
			data.append(row)

	return data