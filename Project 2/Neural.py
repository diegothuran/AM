#
# Imports
#
import numpy as np
import Util
from sklearn.model_selection import train_test_split

class TransferFunctions:
    def sgm(x, Derivative=False):
        if not Derivative:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            out = sgm(x)
            return out * (1.0 - out)

    def linear(x, Derivative=False):
        if not Derivative:
            return x
        else:
            return 1.0

    def gaussian(x, Derivative=False):
        if not Derivative:
            return np.exp(-x ** 2)
        else:
            return -2 * x * np.exp(-x ** 2)

    def tanh(x, Derivative=False):
        if not Derivative:
            return np.tanh(x)
        else:
            return 1.0 - np.tanh(x) ** 2

    def truncLinear(x, Derivative=False):
        if not Derivative:
            y = x.copy()
            y[y < 0] = 0
            return y
        else:
            return 1.0

    def relu(x, Derivative=False):
        if not Derivative:
            return x * (x > 0)
        else:
            return 1. * (x > 0)


#
# Classes
#
class BackPropagationNetwork:
    """Redeneural e Backpropagation"""

    #
    # Class methods
    #
    def __init__(self, layerSize, layerFunctions=None):
        """inicailização da rede"""

        self.layerCount = 0
        self.shape = None
        self.weights = []
        self.tFuncs = []

        # Layer info
        self.layerCount = len(layerSize) - 1
        self.shape = layerSize

        if layerFunctions is None:
            lFuncs = []
            for i in range(self.layerCount):
                if i == self.layerCount - 1:
                    lFuncs.append(TransferFunctions.linear)
                else:
                    lFuncs.append(TransferFunctions.sgm)
        else:
            if len(layerSize) != len(layerFunctions):
                raise ValueError("Lista inválida")
            elif layerFunctions[0] is not None:
                raise ValueError("Camada de entrada não pode.")
            else:
                lFuncs = layerFunctions[1:]

        self.tFuncs = lFuncs

        # Dados da última execução
        self._layerInput = []
        self._layerOutput = []
        self._previousWeightDelta = []

        # Array de pesos
        for (l1, l2) in zip(layerSize[:-1], layerSize[1:]):
            self.weights.append(np.random.normal(scale=0.01, size=(l2, l1 + 1)))
            self._previousWeightDelta.append(np.zeros((l2, l1 + 1)))

    #
    # Run
    #
    def Run(self, input):
        """Método que executa a rede"""

        lnCases = input.shape[0]

        #Limpa a iiteração anterior
        self._layerInput = []
        self._layerOutput = []


        for index in range(self.layerCount):
            # Determina qual a camada ta endo usando
            if index == 0:
                layerInput = self.weights[0].dot(np.vstack([input.T, np.ones([1, lnCases])]))
            else:
                layerInput = self.weights[index].dot(np.vstack([self._layerOutput[-1], np.ones([1, lnCases])]))

            self._layerInput.append(layerInput)
            self._layerOutput.append(self.tFuncs[index](layerInput))

        return self._layerOutput[-1].T

    #
    # Trainar uma época
    #
    def TrainEpoch(self, input, target, trainingRate=0.2, momentum=0.5):
        """Este método treina a rede por uma época"""

        delta = []
        lnCases = input.shape[0]

        self.Run(input)

        # Cálculo dos deltas
        for index in reversed(range(self.layerCount)):
            if index == self.layerCount - 1:
                # Compare os valores
                output_delta = self._layerOutput[index] - target.T
                error = np.sum(output_delta ** 2)
                delta.append(output_delta * self.tFuncs[index](self._layerInput[index], True))
            else:
                # Compare com a seguindo rede
                delta_pullback = self.weights[index + 1].T.dot(delta[-1])
                delta.append(delta_pullback[:-1, :] * self.tFuncs[index](self._layerInput[index], True))

        # Calcule os deltas
        for index in range(self.layerCount):
            delta_index = self.layerCount - 1 - index

            if index == 0:
                layerOutput = np.vstack([input.T, np.ones([1, lnCases])])
            else:
                layerOutput = np.vstack(
                    [self._layerOutput[index - 1], np.ones([1, self._layerOutput[index - 1].shape[1]])])

            curWeightDelta = np.sum( \
                layerOutput[None, :, :].transpose(2, 0, 1) * delta[delta_index][None, :, :].transpose(2, 1, 0) \
                , axis=0)

            weightDelta = trainingRate * curWeightDelta + momentum * self._previousWeightDelta[index]

            self.weights[index] -= weightDelta

            self._previousWeightDelta[index] = weightDelta

        return error


if __name__ == "__main__":

    data, labels = Util.read_base('abalone-processed.data')
    data = data.astype(float)
    labels = labels.astype(float)
    data_train, data_test, labels_train, labels_test = train_test_split(data, labels)
    lFuncs = [None, TransferFunctions.relu, TransferFunctions.relu]

    bpn = BackPropagationNetwork((data[0].shape[0], 100, 1), lFuncs)

    lnMax = 200
    lnErr = 1e-6
    for i in range(lnMax + 1):
        err = bpn.TrainEpoch(data_train, labels_train, momentum=0.9)
        if i % 25 == 0 and i > 0:
            print("Iteration {0:6d}K - Error: {1:0.6f}".format(int(i / 1000), err))
        if err <= lnErr:
            print("Desired error reached. Iter: {0}".format(i))
            break

    # Mostra a saída

    lvOutput = bpn.Run(data_test)
    for i in range(data_test.shape[0]):
        print("Input: {0} Output: {1}".format(data_test[i], labels_test[i]))


