import numpy as np
import pandas as pd
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
import generate_training_data as gtd
from pgmpy.inference import VariableElimination

def main():
	# getting same writer different writer pairs
	inputs, outputs = \
		gtd.formSameWriterDiffWriterInputOutputFeaturePairs(5)

	# subtracting features of 2 images
	inputs = np.absolute(inputs[:, :9] - inputs[:, 9:])

	# forming pandas dataframe
	data = pd.DataFrame(data = np.concatenate((inputs, outputs), axis=1),
		columns=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10'])

	# create model
	model = BayesianModel([('f10','f1'), ('f10','f2'), ('f10','f3'),
		('f10','f4'), ('f10','f5'), ('f10','f5'), ('f10','f6'),
		('f10','f7'), ('f10','f8'), ('f10','f9')])

	# fit model and data, compute CPDs
	model.fit(data, estimator=BayesianEstimator, prior_type='BDeu')

if __name__ == '__main__':
	main()