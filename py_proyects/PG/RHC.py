import random
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits, load_iris




""" Reduction through Homogeneous Clustering """
class RHC():

	@staticmethod
	def getFileName(param_dict):
		return 'RHC'

	@staticmethod
	def getParameterDictionary():
		param_dict = dict()
		return param_dict

	""" Checking whether cluster is homogeneous """
	def checkClusterCommonLabel(self, in_elements):
		common_label_vec = set(in_elements)

		return True if len(common_label_vec) == 1 else False

	""" Process reduction parameters """
	def processParameters(self, param_dict):
		return

	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
		self.X_init = X
		self.y_init = y

		# Processing parameters:
		self.processParameters(param_dict)

		# Starting stack of elements:
		Q = list()
		Q.append((self.X_init, self.y_init))
		CS = list()

		# Processing element at the top of the stack:
		while len(Q) > 0:
			 # Dequeing Q:
			C = Q.pop()


			if self.checkClusterCommonLabel(C[1]) or C[0].shape[0] == 1:
				r_X = np.median(C[0], axis = 0)

				r_y = list(set(C[1]))[0]

				CS.append((r_X, r_y))
			else:
				M = list() # Initializing set of label-centroids

				# Obtaining set of label-centroids:
				for it_label in list(set(C[1])): # range(C[1].shape[1]):
					label_indexes = np.where(C[1] == it_label)[0]
					if len(label_indexes) > 0:
						M.append(np.median(C[0][label_indexes,:], axis = 0))
				M = np.array(M) # label X n_features
				M = np.unique(M, axis = 0) #####

				resulting_labels = list(range(C[0].shape[0]))
				if C[0].shape[0] > M.shape[0] and M.shape[0] > 1:
					# Kmeans with M as initial centroids:
					kmeans = KMeans(n_clusters = M.shape[0], init = M)
					kmeans.fit(np.array(C[0] + 0.001, dtype = 'double'))
					resulting_labels = kmeans.labels_
    				
				
				# Create new groups and enqueue them:
				for cluster_index in np.unique(resulting_labels):
					indexes = np.where(resulting_labels == cluster_index)[0]
					Q.append((C[0][indexes], C[1][indexes]))


		self.X_out = np.array([CS[u][0] for u in range(len(CS))])
		self.y_out = np.array([CS[u][1] for u in range(len(CS))])

		return self.X_out, self.y_out



if __name__ == '__main__':

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
	params_dict = RHC.getParameterDictionary
	X_red, y_red = RHC().reduceSet(X_train.copy(), y_train.copy(), params_dict)


	cls = KNeighborsClassifier(n_neighbors = 1)
	cls.fit(X_train, y_train)
	y_pred = cls.predict(X_test)
	acc = accuracy_score(y_true = y_test, y_pred = y_pred)
	print("acc - ALL : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_train.shape[0]/X_train.shape[0]))

	cls.fit(X_red, y_red)
	y_pred = cls.predict(X_test)
	acc = accuracy_score(y_true = y_test, y_pred = y_pred)
	print("acc - RED : {:.2f}% - Size : {:.2f}%".format(100*acc, 100*X_red.shape[0]/X_train.shape[0]))