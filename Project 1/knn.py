import random
import math
import operator
import numpy as np
import Util

def normalize(data_set):
    return data_set - data_set.mean() / data_set.std()

def getEuclideanDistance(instance1, instance2, length):
	distance = 0
	for i in range(length):
		distance += pow((instance1[i] - instance2[i]), 2)
	return math.sqrt(distance)

def getVotes(neighbors):
    votes = {}
    for i in range(len(neighbors)):
        if neighbors[i] in votes:
            votes[neighbors[i]] += 1
        else:
            votes[neighbors[i]] = 1
    votes = sorted(votes.items(), key=operator.itemgetter(0), reverse=True)
    return votes[0][0]

def getNeighbors(k, instance, train_set, train_labels):
    distances = []
    instance_length = len(instance)
    for i in range(len(train_set)):
        distance = getEuclideanDistance(instance, train_set[i], instance_length - 1)
        distances.append((distance, train_labels[i]))
    distances.sort(key=operator.itemgetter(0))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][1])
    return neighbors

def getAccuracy(test_labels, predictions):
    matches = 0
    for i in range(len(test_labels)):
        if test_labels[i] == predictions[i]:
            matches += 1
    return (matches / float(len(test_labels))) * 100.0

def getValidationSet(train_set, train_labels):
    validation_set = []
    validation_labels = []
    indexes = []
    train_set_length = len(train_set)
    for i in range (int(0.3 * train_set_length)):
        index = random.randint(0, train_set_length - 1)
        while index in indexes:
            index = random.randint(0, train_set_length - 1)
        validation_set.append(train_set[index])
        validation_labels.append(train_labels[index])
        indexes.append(index)
    train_set = np.delete(train_set, indexes, axis=0)
    train_labels = np.delete(train_labels, indexes, axis=0)
    return train_set, train_labels, np.array(validation_set).astype(np.float), np.array(validation_labels).astype(np.str)

def execute(k, train_set, train_labels, test_set, test_labels):
    predictions = []
    for i in range(len(test_set)):
        neighbors = getNeighbors(k, test_set[i], train_set, train_labels)
        result = getVotes(neighbors)
        predictions.append(result)
    accuracy = Util.getAccuracy(test_labels, predictions)
    return predictions, accuracy

def fitK(train_set, train_labels, iterations, k_max):
    k = 0
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 27, 29, 31, 35, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 87, 89, 95, 97, 101]
    means = []
    for l in range(len(primes)):
        k = primes[l]
        if(k > k_max):
            break;
        accu = 0
        length = int(iterations)
        for i in range(length):
            new_train_set, new_train_labels, validation_set, validation_labels = getValidationSet(train_set, train_labels)
            predictions, accuracy = execute(k, train_set, train_labels, validation_set, validation_labels)
            accu += accuracy
        mean = accu / length
        means.append((k, mean))
        means.sort(key=operator.itemgetter(1), reverse=True)
    k = means[0][0]
    return k

def classify(k, instance, train_set, train_labels):
    neighbors = getNeighbors(k, instance, train_set, train_labels)
    result = getVotes(neighbors)
    return result

# def knn(k, train_set, train_labels, test_set, test_labels, validation, iterations, k_max, norm):
#     if norm:
#         train_set = normalize(train_set)
#         test_set = normalize(test_set)
#     if validation:
#         k = fitK(train_set, train_labels, iterations, k_max)
#     predictions, accuracy = execute(k, train_set, train_labels, test_set, test_labels)
#     return k, predictions, accuracy

# def main():
#     shape_train_set, rgb_train_set, train_labels = readBase('segmentation.test')
#     shape_test_set, rgb_test_set, test_labels = readBase('segmentation.data')
#
#     # print('executing knn for Shape View...')
#     k, predictions, accuracy = knn(
#         k=2,
#         train_set=shape_train_set,
#         train_labels=train_labels,
#         test_set=shape_test_set,
#         test_labels=test_labels,
#         validation=True,
#         iterations=3,
#         k_max=11,
#         norm=False
#     )
#
#     # print('\nexecuting knn for RGB View...')
#     k, predictions, accuracy = knn(
#         k=2,
#         train_set=rgb_train_set,
#         train_labels=train_labels,
#         test_set=rgb_test_set,
#         test_labels=test_labels,
#         validation=True,
#         iterations=3,
#         k_max=11,
#         norm=False
#     )

# if __name__ == "__main__":
#     main()