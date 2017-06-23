from sklearn.neighbors import KNeighborsClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import confusion_matrix
import numpy as np

import Util
import ignore_warnings

def main():
	results = {}
	training_samples, training_labels = Util.read_base('abalone-ACNN96.data')

	# ==================================================
	# K Nearest Neighbors classifier
	# ==================================================

	# KNN parameters
	params = {
	'n_neighbors' : np.array([2,5,10]),
	'weights' : np.array(["uniform","distance"]),
	'algorithm' : ["ball_tree","kd_tree","brute"],
	'leaf_size' : np.array([10,20,30,40])}

	for param in params.keys():
		print("========================================")
		print("Testing values for '"+param+"'")
		print("========================================")
		classifier = KNeighborsClassifier()
		grid = GridSearchCV(estimator=classifier, #scoring="accuracy",
			param_grid={param:params[param]})
		grid.fit(training_samples, training_labels)
		print("> Best score: "+str(grid.best_score_))
		print("> Best param: "+str(getattr(grid.best_estimator_,param)))
		results[param] = str(getattr(grid.best_estimator_,param))

	print("\n\n")
	for arg in results.keys():
		print("Best value for '"+arg+"': "+results[arg])

main()