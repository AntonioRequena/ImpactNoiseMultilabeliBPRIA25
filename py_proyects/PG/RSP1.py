import numpy as np
from skmultilearn.dataset import load_dataset
from sklearn.metrics import pairwise_distances
import random
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris

import time

class RSP1():

	@staticmethod
	def getFileName(param_dict):
		return 'RSP1_Red-{}'.format(param_dict['red'])


	@staticmethod
	def getParameterDictionary():
		param_dict = dict()
		param_dict['red'] = 50
		return param_dict


	def getMostDistantPrototypes(self, in_list):
		out1, out2 = -1, -1

		if len(in_list) > 0:
			idx1, idx2 = np.where(self.distances_dict[in_list,:][:,in_list] == np.max(self.distances_dict[in_list,:][:,in_list]))
			out1 = in_list[idx1[0]]
			out2 = in_list[idx2[0]]
		
		return out1, out2
		



	def divideBIntoSubsets(self, B, p1, p2):
		B1_indexes = [B[idx] for idx in np.where(np.array([self.distances_dict[min(u, p1)][max(u, p1)] <= self.distances_dict[min(u, p2)][max(u, p2)] for u in B]) == True)[0]]
		B2_indexes = list(np.array(sorted(list(set(B) - set(B1_indexes)))))
		
		return B1_indexes, B2_indexes


	def setContainSeveralClasses(self, in_set):
		return True if np.unique(self.y[in_set], axis = 0).shape[0] > 1 else False



	def generatePrototypes(self, indexes):
		X = self.X[indexes]
		y = self.y[indexes]

		X_out = list()
		y_out = list()

		# Retrieving different classes:
		unique_classes = np.unique(y, axis = 0)

		for single_unique_class in unique_classes:
			r = np.median(X[np.where(y == single_unique_class)[0]], axis = 0)
			r_class = single_unique_class

			X_out.append(r)
			y_out.append(r_class)

		return (X_out, y_out)



	def computePairwiseDistances(self):

		self.distances_dict = pairwise_distances(X = self.X, n_jobs = -1)
		return


	""" Process reduction parameters """
	def processParameters(self, param_dict):
		self.red = param_dict['red']
		return


	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
		self.X = X
		self.y = y

		# Processing parameters:
		self.processParameters(param_dict)

		# Number of out elements:
		n_out = int(self.red * self.X.shape[0]/100)

		# Precompute pairwise distances:
		self.computePairwiseDistances()

		# Out elements:
		C = [list(range(self.X.shape[0]))]

		# Aux lists:
		most_distant_prototypes = list()
		most_distant_prototypes_distances = list()

		# Step 2:
		bc = 0
		Qchosen = 0
		several_classes_list = list()
		most_distant_prototypes.append(self.getMostDistantPrototypes(C[0]))
		most_distant_prototypes_distances.append(self.distances_dict[most_distant_prototypes[0]])
		several_classes_list.append(self.setContainSeveralClasses(C[0]))


		for _ in range(n_out - 1):

			# Step 3:
			B_list = C[Qchosen]

			# Step 4:
			p1, p2 = most_distant_prototypes[Qchosen]

			# Step 5:
			B1_indexes, B2_indexes = self.divideBIntoSubsets(B_list, p1, p2)
			B1 = B1_indexes
			B2 = B2_indexes

			# Step 6:
			i = Qchosen
			bc += 1
			C[i] = B1
			C.append(B2)
			most_distant_prototypes[i] = self.getMostDistantPrototypes(C[i])
			most_distant_prototypes_distances[i] = self.distances_dict[most_distant_prototypes[i]]
			most_distant_prototypes.append(self.getMostDistantPrototypes(C[bc]))
			most_distant_prototypes_distances.append(self.distances_dict[most_distant_prototypes[bc]])
			several_classes_list[i] = self.setContainSeveralClasses(C[i])
			several_classes_list.append(self.setContainSeveralClasses(C[bc]))

			# Step 7:
			selected_case = True if True in np.array(several_classes_list) else False
			selected_indexes = np.where(np.array(several_classes_list) == selected_case)[0]
			Qchosen = selected_indexes[np.where(np.array(most_distant_prototypes_distances)[selected_indexes] == max(np.array(most_distant_prototypes_distances)[selected_indexes]))[0][0]]

		X_out = list()
		y_out = list()
		for single_cluster in C:
			if len(single_cluster) > 0:
				prot, labels = self.generatePrototypes(single_cluster)
				X_out.extend(prot)
				y_out.extend(labels)

		return np.array(X_out), np.array(y_out)



if __name__ == '__main__':
	start_time = time.time()
	random.seed(1)
	# Loading dataset:
	digits = load_digits()
	# Number of elements:
	n_elements = list(range(digits.data.shape[0]))

	# Partitions:
	random.shuffle(n_elements)
	### - Train:
	X_train, y_train = digits.data[n_elements[:int(len(n_elements)*.6)]], digits.target[n_elements[:int(len(n_elements)*.6)]]
	### - Test:
	X_test, y_test = digits.data[n_elements[int(len(n_elements)*.6):]], digits.target[n_elements[int(len(n_elements)*.6):]]

	# Performing reduction:
	params_dict = RSP1.getParameterDictionary()
	params_dict['red'] = 100
	X_red, y_red = RSP1().reduceSet(X_train.copy(), y_train.copy(), params_dict)


	cls = KNeighborsClassifier(n_neighbors = 1)
	cls.fit(X_train, y_train)
	y_pred = cls.predict(X_test)
	acc = accuracy_score(y_true = y_test, y_pred = y_pred)
	print("acc - ALL : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_train.shape[0]/X_train.shape[0]))

	cls.fit(X_red, y_red)
	y_pred = cls.predict(X_test)
	acc = accuracy_score(y_true = y_test, y_pred = y_pred)
	print("acc - RED : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_red.shape[0]/X_train.shape[0]))
	print("--- %s seconds ---" % (time.time() - start_time))