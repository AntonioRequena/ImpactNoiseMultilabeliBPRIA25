import sys
import numpy as np
import pandas as pd
from PG.RHC import RHC
from PG.RSP3 import RSP3
from Common import multilabelClassification
from skmultilearn.dataset import load_dataset

sys.path.append('./')




""" Splitting the multilabel problem into single binary classification tasks """
class Plain2():

    @staticmethod
    def getFileName(param_dict):
        return 'Plain2_PG-{}'.format(param_dict['PG'].getFileName(param_dict['PG_param']))


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


        # param_dict = dict()
        # param_dict['PG'] = RHC
        # param_dict['PG_param'] = param_dict['PG'].getParameterDictionary()

    """ Reduction process for a multilabel set """
    def reduceSet(self, X_train, y_train, param_dict, le):
        # Storing the data:
        self.X = X_train
        self.y = y_train

        # Processing the parameters of the PG method:
        self.processParameters(param_dict)

        # Encoding labels using Label Powerset approach:
        y_train_lp = le.transform([str(y) for y in self.y])

        X_red, y_red_lp = self.PG.reduceSet(self.X.copy(), y_train_lp.copy(),\
            param_dict = self.PG_param)

        return X_red, y_red_lp


    """ Classification process """
    def classify(self, X_test, X_file, y_file, classifier, n_neigh, le):
        
        # Reading files with the reduced sets:
        #X_red = np.array(pd.read_csv(X_file, sep=',', header=None, compression='gzip'))
        #y_red = np.array(pd.read_csv(y_file, sep=',', header=None, compression='gzip'))

        # Reading files with the reduced sets:
        X_red = np.array(pd.read_csv(X_file, sep=',', header=None, compression='gzip'))
        y_red = np.array(pd.read_csv(y_file, sep=',', header=None, compression='gzip')).ravel()  # Aquí se aplica ravel()


        # Suitable k value (just in case there are not enough samples):
        k = min(n_neigh, X_red.shape[0]//2) if X_red.shape[0]//2 >= 1 else 1
        
        # Inference:
        # Inference:
        if classifier != 'LabelPowerset':
            y_red_lp = y_red.copy()
            y_red = [eval(le.inverse_transform(y_red_lp)[it].replace(" ",",")) for it in range(y_red_lp.shape[0])]

        y_pred = multilabelClassification(classifier, k, X_red, y_red, X_test)

        if classifier == 'LabelPowerset':
            # Undo label encoding for the predictions:
            y_pred_lp = y_pred.copy()
            y_pred = [eval(le.inverse_transform(y_pred_lp)[it].replace(" ",",")) for it in range(y_pred_lp.shape[0])]

        return y_pred, X_red.shape[0]


if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = Plain2.getParameterDictionary()
    params_dict['PG'] = RSP3
    params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
    params_dict['PG_param']['red'] = 1

    X_red, y_red = Plain2().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)

    print("Done!")