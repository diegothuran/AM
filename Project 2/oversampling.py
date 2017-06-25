from imblearn.over_sampling import RandomOverSampler
import Util

def random_oversample(training_samples, training_labels):
    ada = RandomOverSampler()

    training_samples, training_labels = ada.fit_sample(training_samples, training_labels)

    return training_samples, training_labels