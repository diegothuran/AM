from sklearn.neural_network import MLPClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

import Util
import ignore_warnings

def main():
	results = {}
	training_samples, training_classes = Util.read_base('abalone-ACNN96.data')

	# ==================================================
	# MLP classifier
	# ==================================================
	params = {
	'hidden_layer_sizes' : np.array([(50,),(100,),(200)]),
	'activation' : np.array(['identity','logistic','tanh','relu']),
	#'solver' : np.array(['lbfgs','sgd','adam']),
	#'alpha' : np.array([0.001,0.0001,0.00001]),
	#'learning_rate' : ['constant','invscaling','adaptive'],
	'learning_rate_init' :  np.array([0.01,0.001,0.0001])}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = MLPClassifier()
		grid = GridSearchCV(estimator=classifier, #verbose=10,
			param_grid={param:params[param]})
		grid.fit(training_samples, training_classes)
		print("> Best score: "+str(grid.best_score_))
		print("> Best param: "+str(getattr(grid.best_estimator_,param)))
		results[param] = str(getattr(grid.best_estimator_,param))
		input()
	print("\n\n")

	for arg in results.keys():
		print("Best value for '"+arg+"': "+results[arg])

	classifier = MLPClassifier()
	classifier.fit(training_samples, training_classes)

main()