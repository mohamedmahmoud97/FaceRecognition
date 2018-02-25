from classifiers.base_classifier import Base_classifier
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
class K_nearest_neighbour(Base_classifier):
    def __init__(self, k=1):
        self.k = k
        self.X_train = None
        self.y_train = None
        self.num_classes = 0
    
    def train(self, X_train, y_train):
        self.y_train = y_train
        self.X_train = X_train
        self.num_classes = len(np.unique(self.y_train))
        
    
    def predict(self, X_test):
        dists = euclidean_distances(X_test, self.X_train)
        sorted_idx = np.argsort(dists, axis=1)
        labeled_idx = self.y_train[sorted_idx]
        first_k = labeled_idx[: ,:self.k]
        frequencies = np.apply_along_axis(np.bincount, 1, first_k, minlength=self.num_classes + 1)
        predictions = np.argmax(frequencies, axis=1)
        return predictions