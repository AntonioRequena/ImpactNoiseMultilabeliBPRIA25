import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier





class MRHC():

	@staticmethod
	def getFileName(param_dict):
		return 'MRHC'

	@staticmethod
	def getParameterDictionary():
		param_dict = dict()
		return param_dict


	def checkClusterCommonLabel(self, in_elements):
		# Checking whether there is a common label in ALL elements in the set:
		common_label_vec = [len(np.nonzero(in_elements[:,it]==1)[0]) == len(in_elements) for it in range(in_elements.shape[1])]

		return True if True in common_label_vec else False


	""" Process reduction parameters """
	def processParameters(self, param_dict):
		return


	""" Method for performing the reduction """
	def reduceSet(self, X, y, param_dict):
		self.X_init = np.array(X)
		self.y_init = np.array(y)

		# Processing parameters:
		self.processParameters(param_dict)

		Q = list()
		Q.append((self.X_init, self.y_init))
		CS = list()

		while len(Q) > 0:
			#print(len(Q)) #Comentado por Antonio Requena.
			C = Q.pop() # Dequeing Q
			if self.checkClusterCommonLabel(C[1]) or C[0].shape[0] == 1:
				r = np.median(C[0], axis = 0)

				r_labelset = list()

				for it_label in range(C[1].shape[1]):
					n = len(np.where(C[1][:, it_label] == 1)[0])
					r_labelset.append(1) if n > C[1].shape[0]//2 else r_labelset.append(0)

				CS.append((r, r_labelset))
			else:
				M = list() # Initializing set of label-centroids

				# Alternative cluster assignment to the k-means process (when not adequately working)
				resulting_labels_aux = list(range(C[0].shape[0]))
				aux_id_label = 0

				# Obtaining set of label-centroids:
				for it_label in range(C[1].shape[1]):
					label_indexes = np.where(C[1][:,it_label] == 1)[0]
					if len(label_indexes) > 0:
						M.append(np.median(C[0][label_indexes,:], axis = 0))
						
						# Alternative cluster assignment:
						for u in label_indexes: resulting_labels_aux[u] = aux_id_label
						aux_id_label += 1
				M = np.array(M) # label X n_features

				resulting_labels = list(range(C[0].shape[0]))
				if C[0].shape[0] > M.shape[0]  and M.shape[0] > 1:
					# Kmeans with M as initial centroids:
					kmeans = KMeans(n_clusters = M.shape[0], init = M)
					kmeans.fit(np.array(C[0] + 0.001, dtype = 'double'))
					resulting_labels = kmeans.labels_
    				

				# If k-means only retrieves a single cluster -> not working properly
				### Using alternative cluster assignment:
				if (len(set(resulting_labels))) == 1:
					resulting_labels = np.array(resulting_labels_aux, dtype=np.int32)
				
				# Create new groups and enqueue them:
				for cluster_index in np.unique(resulting_labels):
					indexes = np.where(resulting_labels == cluster_index)[0]
					Q.append((C[0][indexes], C[1][indexes]))


		self.X_out = np.array([CS[u][0] for u in range(len(CS))])
		self.y_out = np.array([CS[u][1] for u in range(len(CS))])

		return self.X_out, self.y_out



if __name__ == '__main__':

	X_train, y_train, feature_names, label_names = load_dataset('rcv1subset1', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('rcv1subset1', 'test')

	params_dict = MRHC.getParameterDictionary()
	X_red, y_red = MRHC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)


	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
