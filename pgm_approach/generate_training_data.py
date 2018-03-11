import csv
import numpy as np

# filename with path of the features file
filename = './../data/pgm/features_pgm.csv'

# read the features in dictionary
# authorname is the key
# list of features is the value
def readDataInDict():
	# initialize empty dictioanary
	dataDict = {}

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
				features = dataDict[authorName]

			# update features list and update dictionary
			features.append(np.array(row[1:], dtype='int8'))
			dataDict.update({authorName:features})

def main():

if __name__ == "__main__":
	main()