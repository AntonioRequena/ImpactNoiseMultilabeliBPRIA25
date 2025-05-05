import random
import numpy as np
from itertools import combinations
from scipy.spatial import distance
from sklearn.metrics import accuracy_score, pairwise_distances
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris
from skmultilearn.dataset import load_dataset



""" Reduction through Space Partitioning - Version 3 """
class RSP3():

	@staticmethod
	def getFileName(param_dict):
		return 'RSP3'


	@staticmethod
	def getParameterDictionary():
		param_dict = dict()
		return param_dict


	""" Method to get the most distant prototypes in the cluster """
	def getMostDistantPrototypes(self, in_list):
		out1, out2 = -1, -1

		if len(in_list) > 0:
			idx1, idx2 = np.where(self.distances[in_list,:][:,in_list] == np.max(self.distances[in_list,:][:,in_list]))
			out1 = in_list[idx1[0]]
			out2 = in_list[idx2[0]]
		
		return out1, out2


	def divideBIntoSubsets(self, B_indexes, p1, p2):
		B1_indexes = np.array(B_indexes)[np.where(np.array([self.distances[u][p1] <= self.distances[u][p2] for u in B_indexes]) == True)[0]]
		B2_indexes = np.array(sorted(list(set(B_indexes) - set(B1_indexes))))

		if len(B2_indexes) == 0:
			B2_indexes = np.array([B1_indexes[-1]])
			B1_indexes = B1_indexes[:-1]

		return list(B1_indexes), list(B2_indexes)



	def checkClusterCommonLabel(self, in_indexes):
		# Checking whether cluster is homogeneous:
		common_label_vec = set(self.y_init[in_indexes])

		return True if len(common_label_vec) == 1 else False



	""" Generating a prototype based on the obtained cluster """
	def generatePrototype(self, in_indexes):
		# Feature vector as the median of the individual features:
		r_X = np.median(self.X_init[in_indexes], axis = 0)

		# Resulting label as the common one: 
		r_y = list(set(self.y_init[in_indexes]))[0]

		return (r_X, r_y)


	def computeDistances(self):

		self.distances = pairwise_distances(X = self.X_init, n_jobs = -1)

		return


	""" Process reduction parameters """
	def processParameters(self, param_dict):
		return


	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
		# Storing data:
		self.X_init = X
		self.y_init = y

		# Processing parameters:
		self.processParameters(param_dict)

		# Precompute pairwise distances:
		self.computeDistances()

		# Indexes:
		self.indexes = list(range(self.X_init.shape[0]))

		# Starting stack of elements:
		Q = list()
		Q.append(self.indexes)
		CS = list()


		# Processing element at the top of the stack:
		while len(Q) > 0:
			 # Dequeing Q:
			C = Q.pop()

			# Getting most distant elements in C:
			p1, p2 = self.getMostDistantPrototypes(C)

			if len(C) > 2:
				B1, B2 = self.divideBIntoSubsets(C, p1, p2)
			else:
				B1 = [p1]
				B2 = [p2]


			for single_partition in [B1, B2]:
				if len(single_partition) > 0:
					if self.checkClusterCommonLabel(single_partition):
						CS.append(self.generatePrototype(single_partition))
					else:
						Q.append(single_partition)


		# Splitting CS into features and labels:
		X_out = np.array([single_CS[0] for single_CS in CS])
		y_out = np.array([single_CS[1] for single_CS in CS])

		return X_out, y_out



if __name__ == '__main__':
	random.seed(1)
	# Loading dataset:
	X_train, y_train, feature_names, label_names = load_dataset('rcv1subset1', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('rcv1subset1', 'test')


	# Performing reduction:
	params_dict = RSP3.getParameterDictionary()
	X_red, y_red = RSP3().reduceSet(X_train.toarray().copy(), y_train.toarray()[:,0].copy(), params_dict)

	cls = KNeighborsClassifier(n_neighbors = 1)
	cls.fit(X_train.toarray(), y_train.toarray()[:,0])
	y_pred = cls.predict(X_test.toarray())
	acc = accuracy_score(y_true = y_test.toarray()[:,0], y_pred = y_pred)
	print("acc - ALL : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_train.toarray().shape[0]/X_train.toarray().shape[0]))

	cls.fit(X_red, y_red)
	y_pred = cls.predict(X_test.toarray())
	acc = accuracy_score(y_true = y_test.toarray()[:,0], y_pred = y_pred)
	print("acc - RED : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_red.shape[0]/X_train.toarray().shape[0]))