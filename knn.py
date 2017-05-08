import math
import operator
import random
import numpy as np
from Util import readBase

def normalize(raw_set):
    normalized = raw_set - raw_set.mean() / raw_set.std()
    return normalized

def getEuclideanDistance(instance_1, instance_2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance_1[i] - instance_2[i]), 2)
	return math.sqrt(distance)

def getNeighbors(test_instance, train_set, train_labels, k):
    distances = []
    view_length = len(test_instance) - 1
    for i in range(len(train_set)):
        distance = getEuclideanDistance(test_instance, train_set[i], view_length)
        distances.append((distance, train_labels[i]))
    distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def getVotes(neighbors):
    votes = {}
    for i in range(len(neighbors)):
        if neighbors[i] in votes:
            votes[neighbors[i]] += 1
        else:
            votes[neighbors[i]] = 1
    votes = sorted(votes.items(), key=operator.itemgetter(0), reverse=True)
    return votes[0][0]

def getAccuracy(test_labels, predictions):
    matches = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            matches += 1
    return (matches / float(len(test_labels))) * 100.0

def knn(test_set, test_labels, train_set, train_labels, k):
    predictions = []
    # train_set = normalize(train_set)
    # test_set = normalize(test_set)
    print("====== starting... ======")
    for i in range(len(test_set)):
        neighbors = getNeighbors(test_set[i], train_set, train_labels, k)
        result = getVotes(neighbors)
        predictions.append(result)
        # print("instance " + repr(i) + ":")
        # print("neighbors = " + repr(neighbors))
        # print('predicted = ' + repr(result) + ', actual = ' + repr(test_labels[i]))
    accuracy = getAccuracy(test_labels, predictions)
    print("accuracy: " + repr(accuracy) + "%")
    print("====== end ======")
    return predictions

def getValidationSet(train_set, train_labels):
    validation_set = []
    validation_labels_set = []
    indexes_to_remove = []
    train_length = len(train_set)
    validation_length = int(0.3 * train_length)
    for i in range(validation_length):
        index = random.randint(0, train_length - 1)
        while index in indexes_to_remove:
            index = random.randint(0, train_length - 1)
        validation_set.append(train_set[index])
        validation_labels_set.append(train_labels[index])
        indexes_to_remove.append(index)
    train_set = np.delete(train_set, indexes_to_remove, axis=0)
    train_labels = np.delete(train_labels, indexes_to_remove, axis=0)
    return np.array(validation_set).astype(np.float), np.array(validation_labels_set).astype(np.str), train_set, train_labels

def main():
    shape_train, rgb_train, labels_train = readBase('segmentation.test')
    shape_test, rgb_test, labels_test = readBase('segmentation.data')

    shape_validation, shape_validation_labels, shape_train_set, labels_train_set = getValidationSet(shape_train, labels_train)
    rgb_validation, rgb_validation_labels, rgb_train_set, labels_train_set = getValidationSet(rgb_train, labels_train)

    k = 7

    knn(shape_test, labels_test, shape_train, labels_train, k)
    knn(rgb_test, labels_test, rgb_train, labels_train, k)

    # knn(shape_validation, shape_validation_labels, shape_train_set, labels_train_set, k)
    # knn(rgb_validation, rgb_validation_labels, rgb_train_set, labels_train_set, k)

main()