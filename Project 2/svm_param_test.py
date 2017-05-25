from sklearn.svm import SVC
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
	# Support vector machine classifier
	# ==================================================

	# SVM parameters
	params = {
	'C' : np.array([1.0,10.0,100.0]),
	'kernel' : ["linear","poly","rbf","sigmoid"],
	'gamma' : np.array([1e-3,1e-4,1e-5])}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = SVC()
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