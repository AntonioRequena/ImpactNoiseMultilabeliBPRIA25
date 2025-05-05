import sys
import numpy as np
import pandas as pd
from PG.RHC import RHC
from PG.RSP3 import RSP3
from skmultilearn.dataset import load_dataset
from sklearn.neighbors import KNeighborsClassifier

sys.path.append('./')



""" Splitting the multilabel problem into single binary classification tasks """
class Plain1():

    @staticmethod
    def getFileName(param_dict):
        return 'Plain1_PG-{}'.format(param_dict['PG'].getFileName(param_dict['PG_param']))


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['PG'] = RHC
        param_dict['PG_param'] = param_dict['PG'].getParameterDictionary()
        
        return param_dict


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.PG = param_dict['PG']()
        self.PG_param = param_dict['PG_param']

        return


    """ Reduction process for a single label """
    def reduceSetSingleLabel(self, X, y, param_dict):
        # Storing the data:
        self.X_single = X
        self.y_single = y

        # Processing the parameters of the PG method:
        self.processParameters(param_dict)

        # Performing the reduction process:
        X_red, y_red = self.PG.reduceSet(self.X_single, self.y_single, self.PG_param)
        
        return X_red, y_red


    """ Reduction process for a multilabel set """
    def reduceSet(self, X, y, param_dict):
        # Storing the data:
        self.X = X
        self.y = y

        # Processing the parameters of the PG method:
        self.processParameters(param_dict)

        # Iterate through the different labels (binary cases):
        X_red_list = list()
        y_red_list = list()
        for it_class in range(self.y.shape[1]):
            # Single label reduction:
            X_red, y_red = self.reduceSetSingleLabel(self.X, self.y[:, it_class], param_dict)
            X_red_list.append(X_red)
            y_red_list.append(y_red)

        return X_red_list, y_red_list


    """ Classification process """
    def classifyAllLabels(self, X_test, X_file, y_file, n_labels, n_neigh):
        
        # Initiliazing output structre:
        y_pred = np.zeros(shape = (X_test.shape[0], n_labels))

        # Size:
        X_size = list()

        # Binary classifier per label:
        for it_label in range(n_labels):
            # Reading files with the reduced sets:
            X_dst_file = X_file.replace("{}", str(it_label))
            y_dst_file = y_file.replace("{}", str(it_label))
            X_red = np.array(pd.read_csv(X_dst_file, sep=',', header=None, compression='gzip'))
            y_red = np.array(pd.read_csv(y_dst_file, sep=',', header=None, compression='gzip'))

            # Sized of the reduced set per label:
            X_size.append(X_red.shape[0])

            # Suitable k value (just in case there are not enough samples):
            k = min(n_neigh, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
            
            # Setting up the k values for the kNN classifier:
            cls = KNeighborsClassifier(n_neighbors = k)

            # Fitting classifier:
            cls.fit(X_red, y_red)
            
            # Predicting with the classifier:
            y_pred_single_label = cls.predict(X_test)

            # Adding results to the output structure:
            y_pred[:, it_label] = y_pred_single_label

        return y_pred, np.average(X_size)




if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = Plain1.getParameterDictionary()
    params_dict['PG'] = RSP3
    params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
    params_dict['PG_param']['red'] = 1

    X_red, y_red = Plain1().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    print("Done!")