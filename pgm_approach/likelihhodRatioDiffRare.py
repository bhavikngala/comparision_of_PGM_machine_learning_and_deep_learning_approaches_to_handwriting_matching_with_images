import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import generate_training_data as gtd

from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, BicScore
from pgmpy.estimators import K2Score
from pgmpy.estimators import HillClimbSearch
from pgmpy.inference import VariableElimination

def computeDifferenceAndRarityVariables(inputs, numValuesForEachVariable):
	# 2D arrray for the difference variables
	diff = []
	# 2D arrray for the difference variables
	rarity = []

	# number of rows in the inputs arrray
	numRows = inputs.shape[0]
	# number of features
	numFeatures = len(numValuesForEachVariable)

	# iterating over each row
	for rowIndex in range(numRows):
		# new row for each pair, gets appended to diff array
		diffRow = []
		# new row for each pair, gets appended to diff array
		rarityRow = []

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

			rarityRow.append(int((xk + yk) / 2))

		# append the diff row
		diff.append(diffRow)
		# append the rarity row
		rarity.append(rarityRow)

	# return the difference of the variables
	return diff, rarity

def generateDiffAndRarityData():
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
	h0Diff, h0Rarity = computeDifferenceAndRarityVariables(h0Data, stateValuesK)
	h1Diff, h1Rarity = computeDifferenceAndRarityVariables(h1Data,
		stateValuesK)

	return h0Diff, h0Rarity, h1Diff, h1Rarity

def generateDiffAndRarityModel(h0Diff, h0Rarity):
	# correlation matrix
	h0DiffCorrelation = np.corrcoef(h0Diff, rowvar=False)
	h0RarityCorrelation = np.corrcoef(h0Rarity, rowvar=False)

	# converting to pandas data frame
	h0Diff = pd.DataFrame(h0Diff, columns = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])
	h0Rarity = pd.DataFrame(h0Rarity, columns = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'])

	print('\nestimating PGM\n')
	# using hill climbing algo
	hc = HillClimbSearch(h0Diff)
	
	# estimating model
	diffModel = hc.estimate(max_indegree = 40)
	print('difference model:\n', diffModel.edges())

	print('\nplotting heatmap for h0Diff correlation\n')
	sns.heatmap(h0DiffCorrelation, annot=True)
	plt.show()

	# using hill climbing algo
	hc = HillClimbSearch(h0Rarity)
	
	# estimating model
	rarityModel = hc.estimate(max_indegree = 20)
	print('rarity model:\n', rarityModel.edges())

	print('\nplotting heatmap for h0Rarity correlation\n')
	sns.heatmap(h0RarityCorrelation, annot=True)
	plt.show()

	return h0DiffModel, h0RarityModel

def scoreModels(h0Diff, h0Rarity):
	diffModel0 = [('d5', 'd9'), ('d5', 'd3'), ('d3', 'd4'), ('d3', 'd8'), 
				  ('d9', 'd6'), ('d9', 'd1'), ('d9', 'd7'), ('d9', 'd8')]

	diffModel1 = [('d2', 'd5'), ('d5', 'd9'), ('d5', 'd3'), ('d3', 'd4'),
				  ('d3', 'd8'), ('d9', 'd6'), ('d9', 'd1'), ('d9', 'd7'),
				  ('d9', 'd8')]

	diffModel2 = [('d1', 'd2'), ('d5', 'd9'), ('d5', 'd3'), ('d3', 'd4'),
				  ('d3', 'd8'), ('d9', 'd6'), ('d9', 'd1'), ('d9', 'd7'),
				  ('d9', 'd8')]

	print(' \nestimating K2/BIC score of difference structures\n')
	print('k2score model0: {0}		BicScore model0: {1}'.format(
		K2Score(h0Diff).score(BayesianModel(diffModel0)),
		BicScore(h0Diff).score(BayesianModel(diffModel0))))
	print('k2score model1: {0}		BicScore model1: {1}'.format(
		K2Score(h0Diff).score(BayesianModel(diffModel1)),
		BicScore(h0Diff).score(BayesianModel(diffModel1))))
	print('k2score model2: {0}		BicScore model2: {1}'.format(
		K2Score(h0Diff).score(BayesianModel(diffModel2)),
		BicScore(h0Diff).score(BayesianModel(diffModel2))))

	rarityModel0 = [('r5', 'r9'), ('r5', 'r3'), ('r9', 'r1'), ('r8', 'r3'),
					('r6', 'r9'), ('r6', 'r3')]


	rarityModel1 = [('r6', 'r9'), ('r7', 'r9'), ('r3', 'r4'), ('r3', 'r5'),
					('r3', 'r9'), ('r2', 'r9'), ('r5', 'r9'), ('r9', 'r8'),
					('r9', 'r1')]

	rarityModel2 = [('r7', 'r9'), ('r4', 'r3'), ('r4', 'r9'), ('r1', 'r2'),
					('r1', 'r9'), ('r2', 'r9'), ('r5', 'r9'), ('r9', 'r8'),
					('r9', 'r6')]

	print(' \nestimating K2/BIC score of rarity structures\n')
	print('k2score model0: {0}		BicScore model0: {1}'.format(
		K2Score(h0Rarity).score(BayesianModel(rarityModel0)),
		BicScore(h0Rarity).score(BayesianModel(rarityModel0))))
	print('k2score model1: {0}		BicScore model1: {1}'.format(
		K2Score(h0Rarity).score(BayesianModel(rarityModel1)),
		BicScore(h0Rarity).score(BayesianModel(rarityModel1))))
	print('k2score model2: {0}		BicScore model2: {1}'.format(
		K2Score(h0Rarity).score(BayesianModel(rarityModel2)),
		BicScore(h0Rarity).score(BayesianModel(rarityModel2))))

def evaluateNetwrok(model, inputData, hypothesis):
	predictedHypothesis = []

	for i in range(len(evaluationData.index)):

def main():
	# h0 difference and h0 rarity data
	h0Diff, h0Rarity, h1Diff, h1Rarity = generateDiffAndRarityData()	

	# converting to pandas data frame
	h0Diff = pd.DataFrame(h0Diff, columns = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])
	h0Rarity = pd.DataFrame(h0Rarity, columns = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'])
	h1Diff = pd.DataFrame(h1Diff, columns = ['d1', 'd2', 'd3', 'd4', 'd5', 'd6', 'd7', 'd8', 'd9'])
	h1Rarity = pd.DataFrame(h1Rarity, columns = ['r1', 'r2', 'r3', 'r4', 'r5', 'r6', 'r7', 'r8', 'r9'])

	# h0 difference model and h0 rarity model
	# h0DiffModel, h0RarityModel = generateDiffAndRarityModel(h0Diff, h0Rarity)

	# compute scores for differemt models
	# scoreModels(h0Diff, h0Rarity)



if __name__ == '__main__':
	sns.set()
	main()