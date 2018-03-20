import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import generate_training_data as gtd

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, BicScore
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination

def computeDifferenceVariables(inputs, numValuesForEachVariable):
	# 2D arrray for the difference variables
	diff = []
	# number of rows in the inputs arrray
	numRows = inputs.shape[0]
	# number of features
	numFeatures = len(numValuesForEachVariable)

	# iterating over each row
	for rowIndex in range(numRows):
		# new row for each pair, gets appended to diff array
		diffRow = []
		
		# iterating each features pair
		for columnIndex in range(numFeatures):
			# feature k from one source
			xk = inputs[rowIndex, columnIndex]
			# feature k from another source
			yk = inputs[rowIndex, columnIndex + 9]
			
			# if they are same then append it to diffRow array
			if xk == yk:
				diffRow.append(xk)
			# otherwise append sum of nk + min(xk, yk)
			else:
				diffRow.append(numValuesForEachVariable[columnIndex] + \
					min(xk, yk))

		# append the 
		diff.append(diffRow)

	# return the difference of the variables
	return diff

def main():
	# all values features can take
	stateValues = [
		[0, 1, 2, 3],
		[0, 1, 2, 3, 4],
		[0, 1, 2],
		[0, 1, 2, 3, 4],
		[0, 1, 2, 3],
		[0, 1, 2, 3],
		[0, 1, 2, 3],
		[0, 1, 2, 3, 4],
		[0, 1, 2],
	]

	# number of values each feature can take
	stateValuesK = [4, 5, 3, 5, 4, 4, 4, 5, 3]

	# input/output data
	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(20, False)

	trainingInputs = np.array(trainingInputs)
	trainingOutputs = np.array(trainingOutputs)

	print('trainingInputs shape:', trainingInputs.shape)
	print('trainingOutputs shape:', trainingOutputs.shape)

	# separating pairs for h0 hypithesis and h1 hypothesis
	# h0: same writer pairs
	# h1: different writer pairs
	h0Data = trainingInputs[(trainingOutputs[:] == 1).nonzero()[0]]
	h1Data = trainingInputs[(trainingOutputs[:] == 0).nonzero()[0]]

	print('h0 shape:', h0Data.shape)
	print('h1 shape:', h1Data.shape)

	# computing new variable difference variables
	h0Diff = computeDifferenceVariables(h0Data, stateValuesK)
	h1Diff = computeDifferenceVariables(h1Data, stateValuesK)

	# correlation matrix
	h0Correlation = np.corrcoef(h0Diff, rowvar=False)
	h1Correlation = np.corrcoef(h1Diff, rowvar=False)

	print('h0Correlation:', h0Correlation.shape)
	print('\nplotting heatmap for h0 correlation\n')
	sns.heatmap(h0Correlation, annot=True)

	'''
	print('\nplotting heatmap for h1 correlation\n')
	sns.heatmap(h1Correlation, annot=True)
	plt.show()
	'''

	# converting to pandas data frame
	h0Diff = pd.DataFrame(h0Diff, columns = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])

	print('\nestimating PGM\n')
	# using hill climbing algo
	hc = HillClimbSearch(h0Diff)
	# estimating model
	model = hc.estimate(max_indegree = 5)
	print(model.edges())

	plt.show()

if __name__ == '__main__':
	sns.set()
	main()