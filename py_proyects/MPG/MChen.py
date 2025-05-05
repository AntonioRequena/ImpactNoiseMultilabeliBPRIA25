import numpy as np
from itertools import combinations
from scipy.spatial import distance
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.metrics import pairwise_distances
import time


class MChen():

	@staticmethod
	def getFileName(param_dict):
		return 'MChen_Red-{}'.format(param_dict['red'])


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
		B2_indexes = list(sorted(list(set(B) - set(B1_indexes))))

		return B1_indexes, B2_indexes


	def setContainSeveralClasses(self, in_set):
		return True if np.unique(self.y[in_set], axis = 0).shape[0] > 1 else False


	def generatePrototype(self, indexes):
		r = np.median(self.X[indexes], axis = 0)

		r_labelset = list()

		for it_label in range(self.y.shape[1]):
			n = len(np.where(self.y[indexes, it_label] == 1)[0])
			r_labelset.append(1) if n > len(indexes)//2 else r_labelset.append(0)

		return (r, r_labelset)


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

		# Step 2:
		most_distant_prototypes_distances = list()

		bc = 0
		Qchosen = 0
		several_classes_list = list()
		prototypeIndexesQchosen = [self.getMostDistantPrototypes(C[0])]
		several_classes_list.append(self.setContainSeveralClasses(C[0]))
		most_distant_prototypes_distances.append(self.distances_dict[prototypeIndexesQchosen[0]])

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
			C[i] = B1
			C.append(B2)
			
			prototypeIndexesQchosen[i] = self.getMostDistantPrototypes(C[i])
			most_distant_prototypes_distances[i] = self.distances_dict[prototypeIndexesQchosen[i]]
			prototypeIndexesQchosen.append(self.getMostDistantPrototypes(C[bc]))
			most_distant_prototypes_distances.append(self.distances_dict[prototypeIndexesQchosen[bc]])
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
				prot, labels = self.generatePrototype(single_cluster)
				X_out.append(prot)
				y_out.append(labels)

		return np.array(X_out), np.array(y_out)



if __name__ == '__main__':
	start_time = time.time()
	X_train, y_train, feature_names, label_names = load_dataset('rcv1subset1', 'train')
	X_test, y_test, feature_names, label_names = load_dataset('rcv1subset1', 'test')

	params_dict = MChen.getParameterDictionary()
	params_dict['red'] = 10

	X_red, y_red = MChen().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


	cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
	cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

	y_pred_ori = cls_ori.predict(X_test)
	y_pred_red = cls_red.predict(X_test)

	print("Results:")
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
	print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
	print("Done!")
	print("--- %s seconds ---" % (time.time() - start_time))