import os
import numpy as np
import pandas as pd
from Metrics import compute_metrics
from skmultilearn.dataset import load_dataset

# Multilabel classifiers:
from skmultilearn.adapt import BRkNNaClassifier, MLkNN
from skmultilearn.problem_transform import LabelPowerset

# Multiclass classifiers:
from sklearn.neighbors import KNeighborsClassifier


# Params dict:
red_algos_param = {
    'ALL' : [1],
    'RHC' : [1],
    'MRHC' : [1],
    'RSP3' : [1],
    'MRSP3' : [1],
    'RSP1' : [10, 30, 50, 70, 90],
    'MRSP1' : [10, 30, 50, 70, 90],
    'RSP2' : [10, 30, 50, 70, 90],
    'MRSP2' : [10, 30, 50, 70, 90],
    'Chen' : [10, 30, 50, 70, 90],
    'MChen' : [10, 30, 50, 70, 90],
}


""" Function to load a given dataset """
def load_corpus(corpus_name):
    # Loading corpus:
    X_train, y_train, feature_names, label_names = load_dataset(corpus_name, 'train')
    X_test, y_test, feature_names, label_names = load_dataset(corpus_name, 'test')

    # Corpus to standard array:
    X_train = X_train.toarray().copy()
    y_train = y_train.toarray().copy()
    X_test = X_test.toarray().copy()
    y_test = y_test.toarray().copy()

    return X_train, y_train, X_test, y_test


""" Function for obtaining the reduced set using the specified method """
def obtainReducedSet(dst_path, reductionMethod, X_train, y_train, params_dict, res_dict):

        #ARequena
    tipo_ruido = res_dict['noise']
    porcentaje='0'
    porcentaje = res_dict['percen']
    if tipo_ruido != 'swap':
        probabilidad = res_dict['prob']
    else:
        probabilidad = 'none'
    

    
    if not tipo_ruido:
        tipo_ruido = 'none'
        
    nombre_X = f"X_{tipo_ruido}_{porcentaje}_{probabilidad}.csv.gz"
    nombre_Y = f"Y_{tipo_ruido}_{porcentaje}_{probabilidad}.csv.gz"
    
    
    X_dst_file = os.path.join(dst_path, nombre_X)
    y_dst_file = os.path.join(dst_path, nombre_Y)

    #ARequena

    if os.path.isfile(X_dst_file) and os.path.isfile(y_dst_file):
        X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None,\
            compression='gzip'))
        y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None,\
            compression='gzip'))

    else:
        #input('llego hasta aqui:')
        X_red, y_red = reductionMethod().reduceSet(X_train, y_train, params_dict)
        #input('Despues de reduction method:')
        pd.DataFrame(X_red).to_csv(X_dst_file, header=None, index=None,\
            compression='gzip')
        pd.DataFrame(y_red).to_csv(y_dst_file, header=None, index=None,\
            compression='gzip')

    return X_red, y_red


""" Function for performing the ML classification process """
def multilabelClassification(cls, k, X_red, y_red, X_test):

    # Instantiating classifier:
    if cls == 'LabelPowerset': # LP-kNN
        kNN = KNeighborsClassifier(n_neighbors = k)
        cls = LabelPowerset(classifier = kNN, require_dense=[False, False])
    else: # BRkNN, ML-kNN
        cls = eval(cls + '(k=' + str(k) + ')')

    # Fitting classifier:
    cls.fit(X_red, np.array(y_red))
    
    # Inference stage
    y_pred = cls.predict(X_test)
    y_pred = y_pred.toarray()

    return y_pred


""" Function for obtaining the performance metrics as well as the 
size-related ones, and storing the results in the corresponding file """
def obtainResults(res_dict, y_true, y_pred, X_red, X_train, dst_results_file):
    # Computing metrics:
    res_dict.update(compute_metrics(y_true = y_true, y_pred = y_pred))
    res_dict['Size'] = 100*X_red.shape[0]/X_train.shape[0]

    # Writing results:
    if os.path.isfile(dst_results_file):
        out_file = pd.read_csv(dst_results_file)
        out_file.loc[len(out_file)] = res_dict
    else:
        out_file = pd.DataFrame([res_dict])
    out_file.to_csv(dst_results_file, index=False)

    return