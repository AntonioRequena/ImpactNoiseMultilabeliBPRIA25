import sys
sys.path.append('./')

import numpy as np
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier
from PG.RHC import RHC
from PG.RSP3 import RSP3
from skmultilearn.problem_transform import LabelPowerset
from sklearn.neighbors import NearestNeighbors


""" Using a Label Powerset approach to convert the multilabel dataset into a multiclass one """
class Metamethod2():

    @staticmethod
    def getFileName(param_dict):
        return 'Metamethod2_PG-{}'.format(param_dict['PG'].getFileName(param_dict['PG_param']))


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['PG'] = RSP3
        param_dict['PG_param'] = param_dict['PG'].getParameterDictionary()
        
        return param_dict


    """ Retrieving closest elements """ 
    def closestElement(self, Xref, Yref, Xred, Yred):

        vectorIndices = np.zeros(shape = Xred.shape[0], dtype = np.int)

        neigh = NearestNeighbors(n_neighbors=1)
        for single_class in sorted(list(set(Yred))):
            # Selecting elements from the sets with that class:
            ### Reference set:
            idx_reference_elements = np.where(Yref == single_class)[0]
            ### Reduced set:
            idx_reduced_elements = np.where(Yred == single_class)[0]

            # Training kNN with reference samples of current class:
            neigh.fit(Xref[idx_reference_elements])

            # Locating closest elements of the generated elements with the same class:
            _, indices = neigh.kneighbors(Xred[idx_reduced_elements])
            
            # Storing indexes:
            vectorIndices[idx_reduced_elements] = idx_reference_elements[indices[:,0]]
        

        return Xref[vectorIndices], Yref[vectorIndices]


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.PG = param_dict['PG']()
        self.PG_param = param_dict['PG_param']
        return


    """ Reduction process """
    def reduceSet(self, X, y, param_dict):
        self.X = X
        self.y = y
        
        # Processing parameters:
        self.processParameters(param_dict)

        # Performing Label Powerset transformation:
        lp = LabelPowerset()
        y_lp = lp.transform(self.y)

        # Reduction process:
        X_out, y_out_lp = self.PG.reduceSet(self.X, y_lp, self.PG_param)

        # X_out, y_out_lp = self.closestElement(X, y_lp, X_out, y_out_lp)

        # In case the reduction process eliminates all samples, we restore the initial collection:
        if X_out.shape[0] == 0:
            X_out = self.X
            y_out_lp = y_lp


        return X_out, lp.inverse_transform(y_out_lp).toarray()





if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = Metamethod2.getParameterDictionary()
    X_red, y_red = Metamethod2().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red = cls_red.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")