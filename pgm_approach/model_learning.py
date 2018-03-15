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
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])

	testingData = pd.DataFrame(
		data = np.concatenate((testingInputs, testingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])

	return trainingData, testingData

def naiveModel():
	trainingData, testingData = differenceBetweenFeatures(True)

	# create model
	model = BayesianModel(
		[('f10','f1'), ('f10','f2'), ('f10','f3'),
		 ('f10','f4'), ('f10','f5'), ('f10','f6'),
		 ('f10','f7'), ('f10','f8'), ('f10','f9')])

	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator,
		prior_type='BDeu')

	# inference object
	# computing probability of Hyothesis given evidence
	evaluateModel(model, testingData, 'f10', featuresLabelList)

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

def learnedStructureModel():
	trainingData, testingData = differenceBetweenFeatures(False)

	hc = HillClimbSearch(trainingData, scoring_method=BicScore(trainingData))
	model = hc.estimate()
	# fit model and data, compute CPDs
	model.fit(trainingData, estimator=BayesianEstimator, prior_type='BDeu')

	print(model.edges())

	# inference object
	# computing probability of Hyothesis given evidence
	evaluateModel(model, testingData)

def evaluateModel(model, evaluationData, queryNode, evidenceNodes):
	model_infer = VariableElimination(model)

	actualValue = []
	predictedValue = []

	for i in range(len(evaluationData.index)):
		e = evaluationData.loc[i, evidenceNodes].to_dict()
		print(queryNode, evaluationData.get_value(i, queryNode))
		actualValue.append(evaluationData.get_value(i, queryNode))
		q = model_infer.query(variables=[queryNode], evidence=e)
		phi = q[queryNode]
		print(phi.values)
		print('prediction:', np.argmax(phi.values))
		predictedValue.append(np.argmax(phi.values))

	print('accuracy:', np.mean(np.array(actualValue) == np.array(predictedValue)))

def correlationOfVariabledInPairs():
	trainingInputs, trainingOutputs, testingInputs, testingOutputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5, False)

	trainingData = pd.DataFrame(
		data = np.concatenate((trainingInputs, trainingOutputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10',\
			'f11', 'f12', 'f13', 'f14', 'f15', 'f16', 'f17', 'f18', 'f19'])

	correlationMatrix = trainingData.corr()
	print(correlationMatrix)
	
	sns.heatmap(correlationMatrix, annot=True)
	plt.show()

def main():
	# learnedStructureModel()
	# naiveModel()
	correlationOfVariabledInPairs()

if __name__ == '__main__':
	main()