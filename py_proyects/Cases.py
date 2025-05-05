import os
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import LabelEncoder
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import hamming_loss, f1_score, roc_auc_score



""" Compute metrics """
def compute_metrics(y_test, y_pred, red_size, init_size):
	
	# Init dict:
	results_dict = dict()
	
	# Preparing data into np array (if necessary):
	if type(y_test) != type(y_pred):
		y_pred = y_pred.toarray() 

	# HL:
	results_dict['HL'] = 100*hamming_loss(y_true = y_test, y_pred = y_pred)
	
	# F1:
	results_dict['F1-m'] = 100*f1_score(y_true = y_test, y_pred = y_pred,\
		average = 'micro')
	results_dict['F1-M'] = 100*f1_score(y_true = y_test, y_pred = y_pred,\
		average = 'macro')
	results_dict['F1-s'] = 100*f1_score(y_true = y_test, y_pred = y_pred,\
		average = 'samples')

	# AUC:
	results_dict['AUC-m'] = 100*roc_auc_score(y_true = y_test, y_score = y_pred,\
		average = 'micro')
	try:
		results_dict['AUC-m'] = 100*roc_auc_score(y_true = y_test, y_score = y_pred,\
			average = 'micro')
	except:
		results_dict['AUC-m'] = -1


	try:
		results_dict['AUC-M'] = 100*roc_auc_score(y_true = y_test, y_score = y_pred,\
			average = 'macro')
	except:
		results_dict['AUC-M'] = -1
	
	try:
		results_dict['AUC-s'] = 100*roc_auc_score(y_true = y_test, y_score = y_pred,\
			average = 'samples')
	except:
		results_dict['AUC-s'] = -1

	# Size:
	results_dict['size'] = 100*red_size/init_size


	return results_dict




""" Base PR case """
def base_PR_case(*args):

	# Extracting parameters:
	X_train, y_train, X_test, y_test, k_value, red_method, corpus_name,\
		Reduction_root_path = args

	# Reduction path:
	reduction_path = os.path.join(Reduction_root_path, 'PR-Case',\
		corpus_name, red_method.getFileName())
	if not os.path.exists(reduction_path):
		os.makedirs(reduction_path)

	# Performing reduction:
	X_dst_file = os.path.join(reduction_path, 'X.csv.gz')
	y_dst_file = os.path.join(reduction_path, 'y.csv.gz')

	if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
		X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
		y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

	else:
		X_red, y_red = red_method.reduceSet(X_train.copy(), y_train.copy(), params = 1)

		pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
		pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None, compression='gzip')


	# Binary Relevance kNN:
	k = min(k_value, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
	cls = BRkNNaClassifier(k = k)
	cls.fit(X_red, y_red)
	y_pred = cls.predict(X_test)
	
	# Figures of merit:
	return compute_metrics(y_test = y_test, y_pred = y_pred, red_size = X_red.shape[0],\
		init_size = X_train.shape[0])


""" Retrieving closest elements """ 
def closestElement(Xref, Yref, Xred, Yred):

	neigh = NearestNeighbors(n_neighbors=1)
	neigh.fit(Xref)
	
	_, indices = neigh.kneighbors(Xred)

	return Xref[indices[:,0]], Yref[indices] 



""" Study I """
def StudyI(*args):

	# Extracting parameters:
	X_train, y_train, X_test, y_test, k_value, red_method, corpus_name,\
		Reduction_root_path = args

	# Reduction path:
	reduction_path = os.path.join(Reduction_root_path, 'StudioI',\
		corpus_name, red_method.getFileName())
	if not os.path.exists(reduction_path):
		os.makedirs(reduction_path)

	# Result array (initialized to zero):
	y_result = np.zeros(shape = y_test.shape)

	avg_size = list()
	for it_class in range(y_train.shape[1]):

		X_dst_file = os.path.join(reduction_path, 'X_label' + str(it_class) +  '.csv.gz')
		y_dst_file = os.path.join(reduction_path, 'y_label' + str(it_class) +  '.csv.gz')

		if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
			X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
			y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

		else:
			X_red, y_red = red_method.reduceSet(X_train.copy(), y_train[:,it_class].copy(),\
				params = 1)

			pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
			pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None, compression='gzip')

		# Multiclass classifier:
		k = min(k_value, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
		cls = KNeighborsClassifier(n_neighbors = k)

		# Fitting classifier:
		cls.fit(X_red, y_red)
		
		# Predicting with the classifier:
		y_pred = cls.predict(X_test)

		# Adding results to the output structure:
		y_result[:, it_class] = y_pred

		# Adding size as a metric:
		avg_size.append(X_red.shape[0])

	# Figures of merit:
	return compute_metrics(y_test = y_test, y_pred = y_result,\
		red_size = np.average(avg_size), init_size = X_train.shape[0])


""" Study I Selection """
def StudyISelection(*args):

	# Extracting parameters:
	X_train, y_train, X_test, y_test, k_value, red_method, corpus_name,\
		Reduction_root_path = args

	# Reduction path:
	reduction_path = os.path.join(Reduction_root_path, 'StudioI',\
		corpus_name, red_method.getFileName())
	if not os.path.exists(reduction_path):
		os.makedirs(reduction_path)

	# Result array (initialized to zero):
	y_result = np.zeros(shape = y_test.shape)

	avg_size = list()
	for it_class in range(y_train.shape[1]):

		X_dst_file = os.path.join(reduction_path, 'X_label' + str(it_class) +  '.csv.gz')
		y_dst_file = os.path.join(reduction_path, 'y_label' + str(it_class) +  '.csv.gz')

		if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
			X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
			y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

		else:
			X_red, y_red = red_method.reduceSet(X_train.copy(), y_train[:,it_class].copy(),\
				params = 1)

			pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
			pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None, compression='gzip')

		# Multiclass classifier:
		k = min(k_value, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
		cls = KNeighborsClassifier(n_neighbors = k)
		
		# From Generation to Selection:
		X_red, y_red = closestElement(X_train, y_train[:, it_class], X_red, y_red)

		# Fitting classifier:
		cls.fit(X_red, y_red)
		
		# Predicting with the classifier:
		y_pred = cls.predict(X_test)

		# Adding results to the output structure:
		y_result[:, it_class] = y_pred

		# Adding size as a metric:
		avg_size.append(X_red.shape[0])

	# Figures of merit:
	return compute_metrics(y_test = y_test, y_pred = y_result,\
		red_size = np.average(avg_size), init_size = X_train.shape[0])


""" Study II """
def StudyII(*args):

	# Extracting parameters:
	X_train, y_train, X_test, y_test, k_value, red_method, corpus_name,\
		Reduction_root_path = args

	# Reduction path:
	reduction_path = os.path.join(Reduction_root_path, 'StudioII-III',\
		corpus_name, red_method.getFileName())
	if not os.path.exists(reduction_path):
		os.makedirs(reduction_path)

	# Encoding labels using Label Powerset approach:
	le = LabelEncoder()
	le.fit([str(y) for y in y_train])
	y_train_lp = le.transform([str(y) for y in y_train])

	# Performing reduction:
	X_dst_file = os.path.join(reduction_path, 'X.csv.gz')
	y_dst_file = os.path.join(reduction_path, 'y.csv.gz')

	if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
		X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
		y_red_lp = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

	else:
		X_red, y_red_lp = red_method.reduceSet(X_train.copy(), y_train_lp.copy(), params = 1)

		pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
		pd.DataFrame(y_red_lp).to_csv(y_dst_file, header=None, index=None, compression='gzip')

	# Multiclass classifier:
	k = min(k_value, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
	cls = KNeighborsClassifier(n_neighbors = k).fit(X_red, y_red_lp)

	# Inference:
	y_pred_lp = cls.predict(X_test)

	# Undo label encoding for the predictions:
	y_pred = [eval(le.inverse_transform(y_pred_lp)[it].replace(" ",",")) for it in range(y_pred_lp.shape[0])]
	
	# Figures of merit:
	return compute_metrics(y_test = y_test, y_pred = np.array(y_pred),\
		red_size = X_red.shape[0], init_size = X_train.shape[0])



""" Study III """
def StudyIII(*args):

	# Extracting parameters:
	X_train, y_train, X_test, y_test, k_value, red_method, corpus_name,\
		Reduction_root_path = args

	# Reduction path:
	reduction_path = os.path.join(Reduction_root_path, 'StudioII-III',\
		corpus_name, red_method.getFileName())
	if not os.path.exists(reduction_path):
		os.makedirs(reduction_path)

	# Encoding labels using Label Powerset approach:
	le = LabelEncoder()
	le.fit([str(y) for y in y_train])
	y_train_lp = le.transform([str(y) for y in y_train])

	# Performing reduction:
	X_dst_file = os.path.join(reduction_path, 'X.csv.gz')
	y_dst_file = os.path.join(reduction_path, 'y.csv.gz')

	if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
		X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
		y_red_lp = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

	else:
		X_red, y_red_lp = red_method.reduceSet(X_train.copy(), y_train_lp.copy(), params = 1)

		pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None, compression='gzip')
		pd.DataFrame(y_red_lp).to_csv(y_dst_file, header=None, index=None, compression='gzip')

	# Converting LP to BR:
	y_red = np.array([eval(le.inverse_transform(y_red_lp)[it].replace(" ",",")) for it in range(y_red_lp.shape[0])])

	# Binary Relevance kNN:
	k = min(k_value, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
	cls = BRkNNaClassifier(k = k).fit(X_red, y_red)

	# Inference:
	y_pred = cls.predict(X_test)
		
	# Figures of merit:
	results_dict = dict()
	results_dict['HL'] = 100*hamming_loss(y_true = y_test, y_pred = y_pred)
	results_dict['F1-m'] = 100*f1_score(y_true = y_test, y_pred = y_pred,\
		average = 'micro')
	results_dict['size'] = 100*X_red.shape[0]/X_train.shape[0]
	
	return compute_metrics(y_test = y_test, y_pred = y_pred, red_size = X_red.shape[0],\
		init_size = X_train.shape[0])