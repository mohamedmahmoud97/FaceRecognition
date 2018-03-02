from classifiers.base_classifier import Base_classifier
import numpy as np
from scipy import linalg as LA
import os.path
class PCA:
	def __init__(self):
		self._proj_mat = None
	
	def train(self, X, r=None ,alpha=None, load_path=None, verbose=False):
		
		if load_path is not None and os.path.isfile(load_path):
			#compute covariance matrix
			cov = self.computeCov(X)
			
			#load the eigen vectors and values from the saved file
			data = np.load('pca_projection.npz');EiVal = data['name1'];EiVec = data['name2']
			tempEiVal = np.absolute(EiVal)
			tempEiVec = np.absolute(EiVec)

			return self.computeEiFaces(tempEiVal, tempEiVec, cov, r, alpha, verbose)
		
		#covarinace matrix
		cov = self.computeCov(X)

		#eigen values and vectors
		eighVal,eighVec = LA.eig(cov)

		#temp eigen values and vectors to be sorted for later computation
		tempEiVal = eighVal
		tempEiVec = eighVec

		#sorting of eigen values and vectors
		idx = tempEiVal.argsort()[::-1]   
		tempEiVal = tempEiVal[idx]
		tempEiVec = tempEiVec[:,idx]

		slcEiVec = self.computeEiFaces(tempEiVal, tempEiVec, cov, r, alpha, verbose)

		if load_path is None:
			print("Will not save the eigen vectors to a file")
		else:
			with open(load_path,'wb+') as f:
				np.savez(f, name1=tempEiVal, name2=tempEiVec)

		return slcEiVec

	def project(self, X):
		return X.dot(self._proj_mat)

	def computeCov(self, X):
		#mean
		mean = np.mean(X, axis=0)
		mean = np.expand_dims(mean,axis=1)
		
		#the centered matrix
		mean = np.mean(X,axis=0)
		Z = X - mean
		
		#covarinace matrix
		cov = np.cov(Z.T)
		return cov
	
	def computeEiFaces(self, tempEiVal, tempEiVec, cov, r=None, alpha=None, verbose=False):
		#list of max eigen values chosen
		maxEiIndex = []

		if alpha is None and r is None:
			raise ValueError()

		if alpha is not None:
			#index of last eigen value to be taken to exceed the alpha
			maxIndex = 0

			#compute the acceptable explained variance to exceed the self.alpha
			for x in range(len(tempEiVal)):
				maxEiIndex.append(tempEiVal[x])
				maxIndex = x
				if (np.sum(maxEiIndex)/np.trace(cov))>alpha:
					break
				else:
					continue

		if r is not None:
			maxIndex = r

		#the projection matrix
		slcEiVec = tempEiVec[:,0:maxIndex]
		self._proj_mat = np.absolute(slcEiVec)
		
		print(maxIndex)
		print(tempEiVal.shape,slcEiVec.shape)

		if verbose:
			print("eigen values: \n",tempEiVal)
			print("eigen vectors: \n",slcEiVec)
			print(tempEiVec.shape)
		return slcEiVec