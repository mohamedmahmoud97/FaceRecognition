import numpy as np

def pca(X, alpha, verbose):
	U = []
	maxEiIndex = []

	#mean
	mean = np.mean(X, axis=0)
	mean = np.expand_dims(mean,axis=1)
	
	#the centered matrix
	mean = np.mean(X,axis=0)
	Z = X - mean

	#cpvarinace matrix
	cov = np.cov(X.T)

	#eighen value and vectors
	eighVal,eighVec = np.linalg.eig(cov)

	#temp eighen values and vectors to be sorted for later computation
	tempEiVal = eighVal
	tempEiVec = eighVec
	#index of last eighen value to be taken to exceed the 
	maxIndex = 0

	#sorting of eighen values and vectors
	idx = tempEiVal.argsort()[::-1]   
	tempEiVal = tempEiVal[idx]
	tempEiVec = tempEiVec[:,idx]

	#compute the acceptable explained variance to exceed the alpha
	for x in range(len(eighVal)):
		maxEiIndex.append(tempEiVal[x])
		maxIndex = x
		(np.sum(maxEiIndex)/np.trace(cov))>alpha:break:continue

	#the U projection matrix
	U = tempEiVec[:1:maxIndex]

	X = np.dot(X,U)

	if verbose == True:
		print(mean)
		print(Z)
		print(cov)
		print("eighen values: \n",eighVal)
		print("eighen vectors: \n",eighVec)


	return U,X