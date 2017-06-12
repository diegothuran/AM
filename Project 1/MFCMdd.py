import numpy as np
from Util import *

class MFCMdd:


    def __init__(self, datasets=[], k=7, m=1.6, q=3, T=10):
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
                w = 0.0
                wh = 0.0
                for j in range(len(self.datasets)):
                    w += self.weights[k][j] * np.sum(self.get_dissimilarities_by_prototypes(k, pattern, j))
                for j in range(len(self.datasets)):
                    wh += self.weights[h][j] * np.sum(self.get_dissimilarities_by_prototypes(h, pattern, j))
                sum += pow(w/wh, (1/self.m -1))
            temp.append(pow(sum, -1))

        return temp

    def compute_membership(self):
        U = []
        for i in range(len(self.datasets[0])):
            U.append(self.compute_menbership_by_pattern(i))

        return U



    def calculate_J(self):
        sum = 0.0
        for k in range(self.k):
            u = 0.0
            w = 0.0
            for i in range(len(self.datasets[0])):
                u += self.U[i][k]

            sum += u * np.sum(np.array(self.weights)[k]) * np.sum([self.get_dissimilarities_by_prototypes(k, i, j) for j in range(len(self.datasets))])

        return sum

    def get_dissimilarities_by_prototypes(self, cluster, patter, view=int):
        dissimilarities = []
        cluster = self.G[cluster]

        if len(cluster) > 0:
            for i in range(self.q):
                dissimilarities.append(self.datasets[view][cluster[i]][patter])


        return dissimilarities


    def initialize_weights(self, dataset_len=int, k=int):
        """
            Método que inicializa a matriz de pesos, todos eles com o mesmo valor de 1
        :param dataset:
        :param k:
        :return:
        """
        weights = np.ones((k, dataset_len), dtype=float)/len(self.datasets)
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

        return np.array(u)/len(self.datasets)

    def update_G(self):
        G = [[] for i in range(self.k)]
        usados = []
        for k in range(self.k):
            result = []
            for i in range(len(self.datasets[0])):
                u = np.sum(np.array(self.U).T[k])
                w = 0.0
                for h in range(len(self.datasets[0])):
                    for j in range(len(self.datasets)):
                        w += self.weights[k][j] * self.datasets[j][i][h]
                result.append(u * w)

            while len(G[k]) < 3:
                id_min = np.argmin(result)
                if not usados.__contains__(id_min):
                    usados.append(id_min)
                    G[k].append(id_min)
                result.pop(id_min)
        return G

    def select(self, lst, indices=[]):
        return [lst[i] for i in indices]

    def update_weights(self):
        weights =[]

        for k in range(self.k):
            temp = []
            for j in range(len(self.datasets)):
                num = 1.0
                den = 0.0
                for h in range(len(self.datasets)):
                    u = 0.0
                    for i in range(len(self.datasets[0])):
                        u += self.U[i][k] * np.sum(self.get_dissimilarities_by_prototypes(k, i, h))
                    num *= u
                for i in range(len(self.datasets[0])):
                    den += self.U[i][k] * np.sum(self.get_dissimilarities_by_prototypes(k, i, j))
                temp.append(pow(num, (1/len(self.datasets))) / den)
            weights.append(temp)
        return weights

    def fit(self):
        print(self.J)
        for i in range(self.T):
            self.G = self.update_G()
            self.weights = self.update_weights()
            self.U = self.compute_membership()
            self.J = self.calculate_J()
            print("iteration " + str(i+1) + " " + str(self.J))
        print("G:")
        print(self.G)
        print("grupos")
        matrices = self.get_matricies_by_classes()
        i = 0
        for matrix in matrices:
            print("Grupo: " + str(i))
            print(matrix)
            i+=1
        print("Pesos:")
        print(self.weights)
        print("Fuzzy:")
        print(self.U)




if __name__=='__main__':
    shape_database, rgb_database, labels = readBase("Project 1/segmentation.test")
    shape_dissimilarity = generate_dissimilarity_matrix_scipy(shape_database)
    rgb_dissmilarity = generate_dissimilarity_matrix_scipy(rgb_database)

    mfc = MFCMdd([shape_dissimilarity, rgb_dissmilarity])


    mfc.fit()