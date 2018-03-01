import json
from libpgm.pgmlearner import PGMLearner

import sys
sys.path.insert(0, './../helpers')

import file_helper as postmaster


def main():
	# filename
	features_file = './../data/features.csv'

	# read data into list
	handwriting_features = postmaster.readCSVIntoListAsDict(features_file)

	# learn structure
	# instantiate learner
	learner = PGMLearner()

	pvalue = 0.25
	indegree = 1
	# estimate structure
	#result = learner.discrete_constraint_estimatestruct(
	#	handwriting_features, pvalue, indegree)
	result = learner.discrete_estimatebn(handwriting_features)

	#result = learner.discrete_condind(handwriting_features, 'f1', 'f2',
	#	['f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9'])
	# output 
	#print result.chi, result.pval, result.U
	#print json.dumps(result.E, indent=2)
	print json.dumps(result.Vdata, indent=2)

if __name__ == "__main__":
	main()