import numpy as np
from classifiers.pca import * 
from classifiers.lda import *
class FisherFaces:
    def train(self, X, y, C, r=None, load_path=None, verbose=False):
        if r is None:
            r = C - 1
        if r >= C:
            return
        if load_path is not None:
            lda_load = load_path + '_lda.npy'
            pca_load = load_path + '_pca.npy'
        N = X.shape[0]
        
        pca_dim = N - C
        pca, lda = PCA(), LDA()
        if verbose:
            print('calculating pca projection matrix')
        w_pca = pca.train(X, load_path=pca_load, r=pca_dim, verbose=verbose)
        self.pca = pca
        self.lda = lda
        X_pca = X.dot(w_pca)
        if verbose:
            print('calculating lda projection matrix')
        w_lda = lda.train(X_pca, y, C, r=r, load_path=lda_load, verbose=verbose)
        self._proj_mat = w_pca.dot(w_lda) 
        return self._proj_mat
    
    def project(self, X):

        return X.dot(self._proj_mat)
