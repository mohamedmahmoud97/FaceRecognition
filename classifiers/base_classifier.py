import numpy as np
class Base_classifier:
    def train(self, X_train, y_train):
        raise NotImplementedError
    def predict(self, X_test):
        raise NotImplementedError
    def evaluate(self, X_val, y_val):
        raise NotImplementedError

    
    