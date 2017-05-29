from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

import Util

def warn(*args,**kwargs):
	pass
import warnings
warnings.warn = warn

def main():
	results = {}
	training_samples, training_classes = Util.read_base('abalone-processed.data')

	# ==================================================
	# Decision tree classifier
	# ==================================================

	# Decision tree parameters
	params = {
	'criterion' : ['gini','entropy'],
	'max_features' : ['auto', 'sqrt', 'log2', None],
	'min_samples_split' : np.array([2,4,6]),
	'min_samples_leaf' : np.array([1,2,3])
	}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = DecisionTreeClassifier()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		print("> Best score: "+str(grid.best_score_))
		print("> Best param: "+str(getattr(grid.best_estimator_,param)))
		results[param] = str(getattr(grid.best_estimator_,param))

	print("\n\n")
	for arg in results.keys():
		print("Best value for '"+arg+"': "+results[arg])

main()