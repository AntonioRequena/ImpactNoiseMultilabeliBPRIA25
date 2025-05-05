import random
import numpy as np
from skmultilearn.dataset import load_dataset
from sklearn.metrics import pairwise_distances
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris
import time

class RSP2():

	@staticmethod
	def getFileName(param_dict):
		return 'RSP2_Red-{}'.format(param_dict['red'])


	@staticmethod
	def getParameterDictionary():
		param_dict = dict()
		param_dict['red'] = 50
		return param_dict


	def getMostDistantPrototypes(self, in_list):

		sorted_in_list = sorted(in_list)

		if len(sorted_in_list) > 1:
			max_dist = float('-inf')
			most_distant_duple = ()
			for it_src_element in range(len(sorted_in_list)):
				for it_dst_element in range(it_src_element, len(sorted_in_list)):
					curr_dist = self.distances_dict[sorted_in_list[it_src_element]][sorted_in_list[it_dst_element]]
					if curr_dist > max_dist:
						most_distant_duple = (sorted_in_list[it_src_element], sorted_in_list[it_dst_element])
						max_dist = curr_dist
		else:
			most_distant_duple = (sorted_in_list[0], sorted_in_list[0])
			max_dist = 0

		return most_distant_duple[0], most_distant_duple[1]



	def divideBIntoSubsets(self, B, p1, p2):
		B1_indexes = [B[idx] for idx in np.where(np.array([self.distances_dict[min(u, p1)][max(u, p1)] <= self.distances_dict[min(u, p2)][max(u, p2)] for u in B]) == True)[0]]
		B2_indexes = list(sorted(list(set(B) - set(B1_indexes))))

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



	def getOverlappingDegree(self, indexes):

		d_equal = list()
		d_different = list()

		for src_index in range(len(indexes)):
			for dst_index in range(src_index+1, len(indexes)):
				if self.y[src_index] == self.y[dst_index]:
					d_equal.append(self.distances_dict[src_index][dst_index])
				else:
					d_different.append(self.distances_dict[src_index][dst_index])

		ov = np.nan_to_num(np.average(d_different)/np.average(d_equal))

		return ov


	""" Process reduction parameters """
	def processParameters(self, param_dict):
		self.red = param_dict['red']
		return


	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
		random.seed(10)

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
		C_len = [len(C[0]) > 0]

		# Dictionary for the overlapping degrees:
		OV_Degree = [self.getOverlappingDegree(C[0])]

		# Step 2:
		bc = 0
		Qchosen = 0
		prototypeIndexesQchosen = [self.getMostDistantPrototypes(C[0])]

		for _ in range(n_out - 1):

			# Step 3:
			B = C[Qchosen]
				
			# Step 4:
			p1, p2 = prototypeIndexesQchosen[Qchosen]

			# Step 5:
			B1_indexes, B2_indexes = self.divideBIntoSubsets(B, p1, p2)
			B1 = B1_indexes
			B2 = B2_indexes

			# Step 6:
			i = Qchosen
			bc += 1

			### Adding new partitions:
			C[i] = B1
			C.append(B2)
			OV_Degree[i] = self.getOverlappingDegree(B1)
			OV_Degree.append(self.getOverlappingDegree(B2))
			C_len[i] = len(C[i]) > 0
			C_len.append(len(C[bc]) > 0)


			prototypeIndexesQchosen[i] = self.getMostDistantPrototypes(B1) if len(B1) > 0 else ()
			prototypeIndexesQchosen.append(self.getMostDistantPrototypes(B2) if len(B2) > 0 else ())

			# Selecting partition with the maximum overlap degree:
			if np.max(OV_Degree) > 0.001: Qchosen = np.where(np.array(OV_Degree) == np.max(OV_Degree))[0][0]
			else: Qchosen = random.choice(np.where(np.array(C_len) == True)[0])

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
	params_dict = RSP2.getParameterDictionary()
	params_dict['red'] = 100
	X_red, y_red = RSP2().reduceSet(X_train.copy(), y_train.copy(), params_dict)


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