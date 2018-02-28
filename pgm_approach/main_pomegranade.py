import numpy as np
from pomegranate import BayesianNetwork
import seaborn

import sys
sys.path.insert(0, './../helpers')

import file_helper as postmaster

seaborn.set_style('whitegrid')

def pomegranadeMethod():
	# filename
	features_file = './../data/features.csv'

	# reading data
	data = postmaster.readCSVIntoList(features_file)
	data = np.array(data, dtype='int32')

	# learn model
	model = BayesianNetwork.from_samples(data, algorithm='exact')
	print model.structure
	model.plot()

def main():
	pomegranadeMethod()

if __name__ == "__main__":
	main()