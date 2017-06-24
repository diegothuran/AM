from imblearn.over_sampling import RandomOverSampler
import Util

def random_oversample(training_samples, training_labels):
    ada = RandomOverSampler (random_state=42)

    X_res, y_res = ada.fit_sample(training_samples, training_labels)

    return X_res, y_res