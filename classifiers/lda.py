import numpy as np
from numpy import linalg as LA

"""
LDA function used for dimensionality reduction
X is the data matrix
y is the label column vector
dims is the number of features in the data matrix
r is the dimensions of the subspace
"""

def LDA(X, y, noOfClasses, dims, r, verbose=False):
    #calculating the mean of the classes
    mean_vectors = []
    for cl in range(1, noOfClasses + 1):
        mean_vectors.append(np.mean(X[y==cl], axis=0, keepdims=True))
            
    #within-class matrix
    S = np.zeros((dims, dims))                                         #initializing the matrix
    for cl, mean in zip(range(1, noOfClasses), mean_vectors):
        class_sc_mat = np.zeros((dims, dims))                          # scatter matrix for every class
        #for row in X[y == cl]:
            #row, mean = row.reshape(dims, 1), mean.reshape(dims, 1)   # make column vectors
            #class_sc_mat += (row-mean).dot((row-mean).T)
        mat = X[y == cl]
        z = mat - mean
        S += z.dot(z.T)                                                # sum class scatter matrices
    
    #between-class matrix
    overall_mean = np.mean(X, axis=0, keepdims=True)

    B = np.zeros((dims, dims))
    for i, mean_vec in enumerate(mean_vectors):  
        ni = X[y==i+1,:].shape[0]
        #mean_vec = mean_vec.reshape(10304, 1)                           # make column vector
        #overall_mean = overall_mean.reshape(10304, 1)                   # make column vector
        B += ni * (mean_vec - overall_mean).T.dot((mean_vec - overall_mean))
    
    #calcualting eigenvalues and eigenvectors
    eig_vals, eig_vecs = LA.eig(np.linalg.inv(S).dot(B))
    index = (-eig_vals).argsort()[:r]                                    #sorting the first r eigenvals 
    sortedEigenvec = eig_vals[index]
    
    X_lda = X.dot(sortedEigenvec)
    assert X_lda.shape == (X.shape[0]/2, r), "The matrix is not 150x2 dimensional."
    
    if verbose:
        print(f'Mean vector class {cl} is {mean_vectors}.')
        print('within-class Scatter Matrix:\n', S)
        print('between-class Scatter Matrix:\n', S)
        print('sorted Eigenvectors\n{sortedEigenvec}')
        
    return X_lda
