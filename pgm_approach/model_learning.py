import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, BicScore
from pgmpy.estimators import HillClimbSearch
import generate_training_data as gtd
from pgmpy.inference import VariableElimination
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

featuresLabelList = ['f1','f2','f3','f4','f5','f6','f7','f8','f9']
featuresLabelList2 = ['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
	'f10', 'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18']

def differenceBetweenFeatures(absoluteFlag):
	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, True)

	trainingInputs = trainingInputs[:, :9] - trainingInputs[:, 9:]
	testingInputs = testingInputs[:, :9] - testingInputs[:, 9:]

	if absoluteFlag:
		trainingInputs = np.absolute(trainingInputs)
		testingInputs = np.absolute(testingInputs)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','h'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','h'])

	return trainingData, testingData

def naiveModel():
	trainingData, testingData = differenceBetweenFeatures(True)

	# create model
	'''model = BayesianModel(
		[('f10','f1'), ('f10','f2'), ('f10','f3'),
		 ('f10','f4'), ('f10','f5'), ('f10','f6'),
		 ('f10','f7'), ('f10','f8'), ('f10','f9')])'''

	model = BayesianModel(
		[('f1','h'), ('f2','h'), ('f3','h'),
		 ('f4','h'), ('f5','h'), ('f6','h'),
		 ('f7','h'), ('f8','h'), ('f9','h')])

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu')

	# inference object
	# computing probability of Hyothesis given evidence
	evaluateModel(model, testingData, 'h', featuresLabelList)

def naiveModel2():
	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, True)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19'])

	# create model
	model = BayesianModel(
		[('f19','f1'), ('f19','f2'), ('f19','f3'),
		 ('f19','f4'), ('f19','f5'), ('f19','f6'),
		 ('f19','f7'), ('f19','f8'), ('f19','f9'),
		 ('f19','f10'), ('f19','f11'), ('f19','f12'),
		 ('f19','f13'), ('f19','f14'), ('f19','f15'),
		 ('f19','f16'), ('f19','f17'), ('f19','f18')])

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu')

	# inference object
	# computing probability of Hyothesis given evidence
	evaluateModel(model, testingData, 'f19', featuresLabelList2)

def naiveModel3():
	print("~~~~~~~in function naiveModel3\n")

	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, True)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	# create model
	model = BayesianModel(
		[('f1', 'F1'), ('f11', 'F1'),
		 ('f2', 'F2'), ('f12', 'F2'),
		 ('f3', 'F3'), ('f13', 'F3'),
		 ('f4', 'F4'), ('f14', 'F5'),
		 ('f5', 'F5'), ('f15', 'F5'),
		 ('f6', 'F6'), ('f16', 'F6'),
		 ('f7', 'F7'), ('f17', 'F7'),
		 ('f8', 'F8'), ('f18', 'F8'),
		 ('f9', 'F9'), ('f19', 'F9'),
		 ('F1', 'h'),
		 ('F2', 'h'),
		 ('F3', 'h'),
		 ('F4', 'h'),
		 ('F5', 'h'),
		 ('F6', 'h'),
		 ('F7', 'h'),
		 ('F8', 'h'),
		 ('F9', 'h')])

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu')

	featuresLabelList = ['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']

	# inference object
	# computing probability of Hyothesis given evidence
	evaluateModel(model, testingData, 'h', featuresLabelList)

def finalModel():
	print("~~~~~~~in function naiveModel3\n")

	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(10, True)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	# droping f9 and f18 for now
	trainingData = trainingData.drop(['f9', 'f19'], axis=1)
	testingData = testingData.drop(['f9', 'f19'], axis=1)

	model = BayesianModel(
		[# features in first image
		 # features of a
		 ('f1', 'f2'),
		 # features of n
		 ('f2', 'f4'), ('f4', 'f5'), ('f3', 'f5'),
		 # features of d
		 ('f6', 'f8'), ('f7', 'f8'), ('f4', 'f7'), ('f5', 'f7'), ('f8', 'h'),
		 # features in second image
		 # features of a
		 ('f11', 'f12'),
		 # features of n
		 ('f12', 'f14'), ('f14', 'f15'), ('f13', 'f15'),
		 # features of d
		 ('f16', 'f18'), ('f17', 'f18'), ('f14', 'f17'), ('f15', 'f17'),
		 ('f18', 'h')
		 ])

	# feature values dictionary
	state_names = {
		'f1':[0, 1, 2, 3],
		'f2':[0, 1, 2, 3, 4],
		'f3':[0, 1, 2],
		'f4':[0, 1, 2, 3, 4],
		'f5':[0, 1, 2, 3],
		'f6':[0, 1, 2, 3],
		'f7':[0, 1, 2, 3],
		'f8':[0, 1, 2, 3, 4],
		'f11':[0, 1, 2, 3],
		'f12':[0, 1, 2, 3, 4],
		'f13':[0, 1, 2],
		'f14':[0, 1, 2, 3, 4],
		'f15':[0, 1, 2, 3],
		'f16':[0, 1, 2, 3],
		'f17':[0, 1, 2, 3],
		'f18':[0, 1, 2, 3, 4],
		'h':[0, 1]
	}

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu', state_names=state_names)

	# list of input features
	featuresLabelList = ['f1','f2','f3','f4','f5','f6','f7','f8',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18']

	# inference object
	# computing probability of Hyothesis given evidence
	# infer probabilities for h based on 
	# evaluateModel(model, testingData, 'h', featuresLabelList)

	return model

def naiveModel5():
	print("~~~~~~~in function naiveModel5\n")

	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, True)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',\
		'h'])

	# droping f9 and f18 for now
	trainingData = trainingData.drop(['f9', 'f19'], axis=1)
	testingData = testingData.drop(['f9', 'f19'], axis=1)

	model = BayesianModel(
		[# features in first image
		 # features of a
		 ('f1', 'f2'), ('f2', 'h'),
		 # features of n
		 ('f2', 'f4'), ('f4', 'f5'), ('f3', 'f5'), ('f5', 'h'), 
		 # features of d
		 ('f6', 'f8'), ('f7', 'f8'), ('f4', 'f7'), ('f5', 'f7'), ('f8', 'h'),
		 # features in second image
		 # features of a
		 ('f11', 'f12'), ('f12', 'h'),
		 # features of n
		 ('f12', 'f14'), ('f14', 'f15'), ('f13', 'f15'), ('f15', 'h'),
		 # features of d
		 ('f16', 'f18'), ('f17', 'f18'), ('f14', 'f17'), ('f15', 'f17'),
		 ('f18', 'h')
		 ])

	# feature values dictionary
	state_names = {
		'f1':[0, 1, 2, 3],
		'f2':[0, 1, 2, 3, 4],
		'f3':[0, 1, 2],
		'f4':[0, 1, 2, 3, 4],
		'f5':[0, 1, 2, 3],
		'f6':[0, 1, 2, 3],
		'f7':[0, 1, 2, 3],
		'f8':[0, 1, 2, 3, 4],
		'f11':[0, 1, 2, 3],
		'f12':[0, 1, 2, 3, 4],
		'f13':[0, 1, 2],
		'f14':[0, 1, 2, 3, 4],
		'f15':[0, 1, 2, 3],
		'f16':[0, 1, 2, 3],
		'f17':[0, 1, 2, 3],
		'f18':[0, 1, 2, 3, 4],
		'h':[0, 1]
	}

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu', state_names=state_names)

	# list of input features
	featuresLabelList = ['f1','f2','f3','f4','f5','f6','f7','f8',\
		'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18']

	# inference object
	# computing probability of Hyothesis given evidence
	# infer probabilities for h based on 
	evaluateModel(model, testingData, 'h', featuresLabelList)

def learnedStructureModel():
	# trainingData, testingData = differenceBetweenFeatures(True)
	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, True)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19', 
			'h'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19',
			'h'])

	#trainingData = trainingData.drop(['f9', 'f18'], axis=1)
	#testingData = testingData.drop(['f9', 'f18'], axis=1)
	
	hc = HillClimbSearch(trainingData, scoring_method=BicScore(trainingData))
	model = hc.estimate(max_indegree = 20)

	state_names = {
		'f1':[0, 1, 2, 3],
		'f2':[0, 1, 2, 3, 4],
		'f3':[0, 1, 2],
		'f4':[0, 1, 2, 3, 4],
		'f5':[0, 1, 2, 3],
		'f6':[0, 1, 2, 3],
		'f7':[0, 1, 2, 3],
		'f8':[0, 1, 2, 3, 4],
		'f9':[0, 1, 2],
		'f11':[0, 1, 2, 3],
		'f12':[0, 1, 2, 3, 4],
		'f13':[0, 1, 2],
		'f14':[0, 1, 2, 3, 4],
		'f15':[0, 1, 2, 3],
		'f16':[0, 1, 2, 3],
		'f17':[0, 1, 2, 3],
		'f18':[0, 1, 2, 3, 4],
		'f19':[0, 1, 2],
		'h':[0, 1]
	}

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator, prior_type='BDeu', state_names=state_names)

	print(model.edges())

	# inference object
	# computing probability of Hyothesis given evidence
	evidenceNodes = ['f1','f2','f3','f4','f5','f6','f7','f8','f9',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19']
	evaluateModel(model, testingData, 'h', evidenceNodes)

def evaluateModel(model, evaluationData, queryNode, evidenceNodes):
	print("~~~~~~~in function evaluateModel\n")

	model_infer = VariableElimination(model)

	actualValue = []
	predictedValue = []

	for i in range(len(evaluationData.index)):
		e = evaluationData.loc[i, evidenceNodes].to_dict()
		
		actualValue.append(evaluationData.get_value(i, queryNode))
		q = model_infer.query(variables=[queryNode], evidence=e)
		phi = q[queryNode]

		predictedValue.append(1 - np.argmax(phi.values))

	print('accuracy:', np.mean(np.array(actualValue) == np.array(predictedValue)))

def correlationOfVariabledInPairs():
	'''trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, False)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19'])
	'''
	trainingData, testingData = differenceBetweenFeatures(True)
	correlationMatrix = trainingData.corr()
	print(correlationMatrix)
	
	sns.heatmap(correlationMatrix, annot=True)
	plt.show()

def main():
	learnedStructureModel()
	# naiveModel()
	# correlationOfVariabledInPairs()
	# naiveModel3()
	# print('\nmodel 4\n')
	# finalModel()
	# print('\nmodel 5\n')
	# naiveModel5()

if __name__ == '__main__':
	main()


'''
model = BayesianModel(
		[# features in first image
		 # features of a
		 ('f1', 'f2'),
		 # features of n
		 ('f2', 'f4'), ('f4', 'f5'), ('f3', 'f5'),
		 # features of d
		 ('f6', 'f8'), ('f7', 'f8'), ('f4', 'f7'), ('f5', 'f7'), ('f8', 'h'),
		 # features in second image
		 # features of a
		 ('f11', 'f12'),
		 # features of n
		 ('f12', 'f14'), ('f14', 'f15'), ('f13', 'f15'),
		 # features of d
		 ('f16', 'f18'), ('f17', 'f18'), ('f14', 'f17'), ('f15', 'f17'),
		 ('f18', 'h')
		 ])

'''
''' Model 5
model = BayesianModel(
		[
		 ('f1', 'f2'), ('f2', 'f4'), ('f4', 'f5'), ('f3', 'f5'),
		 ('f4', 'f7'), ('f5', 'f7'), ('f7', 'f8'), ('f6', 'f8'),
		 ('f11', 'f12'), ('f12', 'f14'), ('f14', 'f15'), ('f13', 'f15'),
		 ('f14', 'f17'), ('f15', 'f17'), ('f17', 'f18'), ('f16', 'f18'),
		 ('f1', 'f11'), ('f2', 'f12'), ('f3', 'f13'), ('f4', 'f14'),
		 ('f5', 'f15'), ('f6', 'f16'), ('f7', 'f17'), ('f8', 'f18'),
		 ('f11', 'h'), ('f12', 'h'), ('f13', 'h'), ('f14', 'h'), ('f15', 'h'),
		 ('f16', 'h'), ('f17', 'h'), ('f18', 'h')
		 ])
'''