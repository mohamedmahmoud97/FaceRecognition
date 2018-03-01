from classifiers.base_classifier import Base_classifier
import numpy as np
from scipy import linalg as LA
import os.path
class PCA:
	def __init__(self):
		self._proj_mat = None
	
	def train(self, X, alpha, load_path=None, verbose=False):
		
		if load_path is not None and os.path.isfile(load_path):
			#mean
			mean = np.mean(X, axis=0)
			mean = np.expand_dims(mean,axis=1)
			
			#the centered matrix
			mean = np.mean(X,axis=0)
			Z = X - mean

			#covarinace matrix
			cov = np.cov(X.T)
			data = np.load('pca_projection.npz');EiVal = data['name1'];EiVec = data['name2']
			tempEiVal = np.absolute(EiVal)
			tempEiVec = np.absolute(EiVec)

			print("shape: = : ", tempEiVec.shape)

			maxEiIndex = []

			#compute the acceptable explained variance to exceed the self.alpha
			for x in range(len(tempEiVal)):
				maxEiIndex.append(tempEiVal[x])
				maxIndex = x
				if (np.sum(maxEiIndex)/np.trace(cov))>alpha:
					break
				else:
					continue

			#the projection matrix
			tempEiVec = tempEiVec[:,0:maxIndex]
			self._proj_mat = np.real(tempEiVec)
			print(maxIndex)
			print(tempEiVal.shape,tempEiVec.shape)

			if verbose:
				print(mean)
				print(Z)
				print(cov)
				print("eighen values: \n",tempEiVal)
				print("eighen vectors: \n",tempEiVec)
				print(tempEiVec.shape)

			return self._proj_mat
		
		maxEiIndex = []

		#mean
		mean = np.mean(X, axis=0)
		mean = np.expand_dims(mean,axis=1)
		
		#the centered matrix
		mean = np.mean(X,axis=0)
		Z = X - mean

		#covarinace matrix
		cov = np.cov(X.T)

		#eighen values and vectors
		eighVal,eighVec = LA.eig(cov)

		#temp eighen values and vectors to be sorted for later computation
		tempEiVal = eighVal
		tempEiVec = eighVec
		#index of last eighen value to be taken to exceed the alpha
		maxIndex = 0

		#sorting of eighen values and vectors
		idx = tempEiVal.argsort()[::-1]   
		tempEiVal = tempEiVal[idx]
		tempEiVec = tempEiVec[:,idx]

		#compute the acceptable explained variance to exceed the self.alpha
		for x in range(len(tempEiVal)):
			maxEiIndex.append(tempEiVal[x])
			maxIndex = x
			if (np.sum(maxEiIndex)/np.trace(cov))>alpha:
				break
			else:
				continue

		#the projection matrix
		slcEiVec = tempEiVec[:,0:maxIndex]
		self._proj_mat = np.absolute(slcEiVec)

		with open(load_path,'wb+') as f:
			np.savez(f, name1=tempEiVal, name2=tempEiVec)

		print('computing projected data')

		if verbose:
			print(mean)
			print(Z)
			print(cov)
			print("eighen values: \n",tempEiVal)
			print("eighen vectors: \n",slcEiVec)
			print(tempEiVec.shape)

		return slcEiVec

	def project(self, X):
		return X.dot(self._proj_mat)