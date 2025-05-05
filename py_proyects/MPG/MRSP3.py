import numpy as np
from itertools import combinations
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset, available_data_sets
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import pairwise_distances
import time

class MRSP3():

	@staticmethod
	def getFileName(param_dict):
		return 'MRSP3'


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


	""" Divide cluster into subsets based on the most-distant prototypes in it """
	def divideBIntoSubsets(self, B_indexes, p1, p2):
		B1_indexes = np.array(B_indexes)[np.where(np.array([self.distances[u][p1] <= self.distances[u][p2] for u in B_indexes]) == True)[0]]
		B2_indexes = np.array(sorted(list(set(B_indexes) - set(B1_indexes))))

		if len(B2_indexes) == 0:
			B2_indexes = np.array([B1_indexes[-1]])
			B1_indexes = B1_indexes[:-1]

		return list(B1_indexes), list(B2_indexes)


	""" Checking cluster homogeneity """
	def checkClusterCommonLabel(self, in_elements):
    	# Checking whether there is a common label in ALL elements in the set:
		common_label_vec = [len(np.nonzero(self.y_init[in_elements, it]==1)[0]) == len(in_elements) for it in range(self.y_init.shape[1])]
		# common_label_vec = [len(np.nonzero(in_elements[:,it]==1)[0]) >= len(in_elements)-5 for it in range(in_elements.shape[1])] # Relaxing the homogeneity

		return True if True in common_label_vec else False


	""" Procedure for generating a new prototype """
	def generatePrototype(self, C):
		r = np.median(self.X_init[C], axis = 0)

		r_labelset = list()

		for it_label in range(self.y_init.shape[1]):
			n = len(np.where(self.y_init[C, it_label] == 1)[0])
			r_labelset.append(1) if n > len(C)//2 else r_labelset.append(0)

		return (r, r_labelset)


	""" Precompute pairwise distances """
	def computeDistances(self):

		self.distances = pairwise_distances(X = self.X_init, n_jobs = -1)

		return


	""" Process reduction parameters """
	def processParameters(self, param_dict):
		return


	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
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
			C = Q.pop() # Dequeing Q

			# Getting most distant elements in C:
			p1, p2 = self.getMostDistantPrototypes(C)

			if len(C) > 2:
				B1, B2 = self.divideBIntoSubsets(C, p1, p2)
			else:
				B1 = [p1]
				B2 = [p2]

			for single_partition in [B1, B2]:
				if len(single_partition) > 0:
					if self.checkClusterCommonLabel(single_partition) or len(single_partition) == 1:
						CS.append(self.generatePrototype(single_partition))
					else:
						# print("B1 Non-homogeneous")
						Q.append(single_partition)


		# Splitting CS into features and labels:
		X_out = np.array([single_CS[0] for single_CS in CS])
		y_out = np.array([single_CS[1] for single_CS in CS])

		return X_out, y_out



if __name__ == '__main__':
	start_time = time.time()
	X_train, y_train, feature_names, label_names = load_dataset('rcv1subset1', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('rcv1subset1', 'test')

	params_dict = MRSP3.getParameterDictionary()

	X_red, y_red = MRSP3().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
	print("Done!")
	print("--- %s seconds ---" % (time.time() - start_time))