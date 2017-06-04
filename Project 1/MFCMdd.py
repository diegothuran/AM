import random
import numpy as np
from Util import *

class MFCMdd:


    def __init__(self, datasets=[], k=7, m=1.6, q=3, T=100):
        self.datasets = datasets
        self.k = k
        self.m = m
        self.q = q
        self.T = T
        self.G = []
        self.weights = self.initialize_weights(len(datasets), k)
        self.initialize_G()
        self.C = []
        self.U = self.compute_membership()
        self.J = self.calculate_J()

    def get_subset_by_class(self, classe, dissimilarity_matrix):
        temp = []
        for i in range(dissimilarity_matrix.shape[0]):
            if np.argmax(self.U[i]) == classe:
                temp.append(i)
        return temp

    def get_matricies_by_classes(self):
        return [self.get_subset_by_class(i, self.datasets[0]) for i in range(self.k)]

    def initialize_G(self):
        import random
        samples = random.sample(range(self.datasets[0].shape[0]), 3*self.k)
        for i in range(self.k):
            self.G.append(samples[i*3:3*(i+1)])

    def compute_menbership_by_pattern(self, pattern):
        temp = []
        for k in range(self.k):
            sum = 0.0
            for h in range(self.k):
                num = 0.0
                den = 0.0
                for j in range(len(self.datasets)):
                    num += self.weights[k][j]
                for j in range(len(self.datasets)):
                    den +=self.weights[h][j]

                sum += num * np.sum(self.get_dissimilarities_by_prototypes(k, pattern))/ den * np.sum(self.get_dissimilarities_by_prototypes(h, pattern))
            temp.append(pow(pow(sum, 1/(self.m-1)), -1))
        return temp

    def compute_membership(self):
        U = []
        for i in range(len(self.datasets[0])):
            U.append(self.compute_menbership_by_pattern(i))

        return U



    def calculate_J(self):
        sum = 0.0
        for k in range(self.k):
            sum_u = 0.0
            sum_w = np.sum(self.weights)
            sum_dissmilarities = 0.0
            for i in range(len(self.datasets[0])):
                sum_u += np.sum(self.U[i])
                sum_dissmilarities = np.sum(np.sum([self.get_dissimilarities_by_prototypes(h, i)]) for h in range(self.k))
            sum += sum_u * sum_w * sum_dissmilarities

        return sum

    def get_dissimilarities_by_prototypes(self, cluster, patter):
        dissimilarities = []
        cluster = self.G[cluster]
        for i in range(self.q):
            dissimilarities.append(self.datasets[0][cluster[i]][patter])
            dissimilarities.append(self.datasets[1][cluster[i]][patter])

        return dissimilarities


    def initialize_weights(self, dataset_len=int, k=int):
        """
            Método que inicializa a matriz de pesos, todos eles com o mesmo valor de 1
        :param dataset:
        :param k:
        :return:
        """
        weights = np.ones((k, dataset_len), dtype=float)
        return weights.tolist()

    def initialize_U(self, dataset_len, k=int):
        """
d
            Método responsável por criar a matriz U de pertinência Fuzzy
        :param dataset: matriz de similaridade
        :param k: número de classes
        :return: matriz U com as pertinêcias geradas randomicamente
        """
        u = []
        for i in range(dataset_len):
            temp = np.random.dirichlet(np.ones(k), size=1)
            u.append(temp.tolist())

        return np.array(u)

    def update_G(self):
        result = []
        for h in range(len(self.datasets[0])):
            for i in range(len(self.datasets[0])):
                u = 0.0
                p = 0.0

                for k in range(self.k):
                    u += self.U[i][k]
                for j in range(len(self.datasets)):
                    p += np.sum(self.weights[j]) * self.datasets[j][i][h]
                result.append(u * p)

        matrices = self.get_matricies_by_classes()
        G = [[] for i in range(len(matrices))]
        i = 0
        while len(np.array(G).flatten()) is not 9 and i < len(self.datasets[0]):
            temp = np.argmin(self.U[np.argmin(result)])
            if len(G[temp]) < 3:
                G[temp].append(i)

            result.pop(int(temp))

        return G

    def update_weights(self):
        weights =[]
        for k in range(self.k):
            temp = []
            for p in range(len(self.datasets)):
                u = 0.0
                r = 1.0
                den = 0.0
                for i in range(len(self.datasets[0])):
                    u += self.U[i][k] * np.sum(self.get_dissimilarities_by_prototypes(k, i))
                    den += self.U[i][k] * np.sum(self.get_dissimilarities_by_prototypes(k, i))
                    r *= u
                temp.append([pow(r,1/len(self.datasets))/den])
            weights.append(temp)

        return weights
    def fit(self):
        for i in range(self.T):
            self.G = self.update_G()
            self.weights = self.update_weights()
            self.U = self.compute_membership()
            self.J = self.calculate_J()

        print(self.G)
        print(self.J)
        print(self.weights)



if __name__=='__main__':
    shape_database, rgb_database, labels = readBase("Project 1/segmentation.test")
    shape_dissimilarity = generate_dissimilarity_matrix_scipy(shape_database)
    rgb_dissmilarity = generate_dissimilarity_matrix_scipy(rgb_database)

    mfc = MFCMdd([shape_dissimilarity, rgb_dissmilarity])


    mfc.fit()