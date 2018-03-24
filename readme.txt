Folder structure:

-code/
	--data/
	--helpers/
	--pgm_approach/
	--simple_ml_approach/
	--deep_learning_approach/

Requirements: conda

1. steps to run the PGM approach solution
a. cd ./pgm_apprach
b. conda env create -f pgmpy.yml
c. activate pgmpy (source activate pgmpy|for linux)
d. python main.py -f <path of the hidden data folder>
   eg if the path is say ./../pgmtestdata
   then run python main.py -f ./../pgmtestdata

2. steps to run the simple ML solution
a. cd ./simple_ml_solution
b. conda env create -f py3.yml
c. activate py3 (source activate py3|for linux)
d. python main.py -f <path of the hidden data folder>
   eg if the path is say ./../mltestdata
   then run python main.py -f ./../mltestdata

3. steps to run the deep learning approach:

Note 1: Make sure the root of the project contains the AND dataset.
Note 2 :Make sure the folder titled ‘deep-learning-approach’ contains the hidden data set according to the given format.

a. cd ./deep_learning_approach
b. conda create --name dlenv --file package-list.txt
c. activate dlenv (source activate dlenv|for linux)
d. python main.py