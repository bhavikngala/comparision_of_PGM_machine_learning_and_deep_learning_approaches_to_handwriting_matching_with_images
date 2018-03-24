import numpy as np
import csv
import argparse
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, BicScore
from pgmpy.inference import VariableElimination

import model_learning as pgmModelConstructor

import sys
sys.path.insert(0, './../helpers')
import file_helper as postman

def main(args):
	# read the pair names in the list
	pairNamesList = postman.readPairNamesInList(
		args['filepath'] + '/PGMTestPairs.csv')

	# reading the feature values into list and storing them into dictionary with writer id as key 
	writerFeaturesDict = postman.readWriterFeaturesInDict(
		args['filepath'] + '/PGMTestData.csv')

	# 2D list for saving output in csv
	output = []
	output.append(['', 'FirtImage', 'SecondImage', 'LLR', 'SameOrDifferent'])

	# PGM model
	model = pgmModelConstructor.finalModel()
	
	# object to perform inference
	model_infer = VariableElimination(model)

	for i in range(len(pairNamesList)):

		# getting features in a lists
		writer1 = writerFeaturesDict.get(pairNamesList[i][0])
		writer2 = writerFeaturesDict.get(pairNamesList[i][1])

		# forming the evidence dictionary to use in infer.query function
		evidence = {}
		for j in range(1, len(writer1)):
			featurename = 'f' + str(j)
			evidence.update({featurename:writer1[j-1]})
			featurename = 'f1' + str(j)
			evidence.update({featurename:writer2[j-1]})

		# querying the model
		q = model_infer.query(variables=['h'], evidence=evidence)

		# getting the probability values
		phi = q['h']
		# compute log likelihood
		LLR = np.log(phi.values[1] / phi.values[0])
		
		# predict the value
		predictedValue = 0
		if LLR >= 0:
			predictedValue = 1

		output.append([str(i), pairNamesList[i][0], pairNamesList[i][1],
			str(LLR), str(predictedValue)])

	# save the output list to csv file
	postman.saveToCSVFile(args['filepath'] + '/PGMTestOutput.csv', output)
	
if __name__ == '__main__':
	ap = argparse.ArgumentParser()
	ap.add_argument("-f", "--filepath", required=True,
		help="path of the test data folder")
	args = vars(ap.parse_args())

	main(args)