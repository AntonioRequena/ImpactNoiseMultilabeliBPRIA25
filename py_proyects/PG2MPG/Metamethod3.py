import sys
sys.path.append('./')

import random
import numpy as np
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier
from PG.RHC import RHC
from PG.RSP3 import RSP3
from skmultilearn.problem_transform import LabelPowerset
from sklearn.neighbors import NearestNeighbors

"""  """
"""  """
class Metamethod3():

    @staticmethod
    def getFileName(param_dict):
        return 'Metamethod3_PG-{}'.format(param_dict['PG'].getFileName(param_dict['PG_param']))


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['PG'] = RSP3
        param_dict['PG_param'] = param_dict['PG'].getParameterDictionary()
        
        param_dict['NN'] = 5
        param_dict['gamma'] = 0.5
        param_dict['p'] = 0.75
        param_dict['m'] = 3

        return param_dict


    def _createClassGroups(self):
        class_indexes = list(range(self.y.shape[1]))

        class_groups = list()
        while len(class_indexes) > 0:
            if len(class_indexes) > self.m:
                class_groups.append(sorted(random.sample(class_indexes, self.m)))
            else:
                class_groups.append(class_indexes)
            
            class_indexes = sorted(list(set(class_indexes) - set(class_groups[-1])))
        
        return class_groups


    """ Retrieving closest elements """ 
    def closestElement(self, Xref, Yref, Xred, Yred):
        vectorIndices = np.zeros(shape=Xred.shape[0], dtype=np.int64)

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

        return vectorIndices


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.PG = param_dict['PG']()
        self.PG_param = param_dict['PG_param']

        self.NN = param_dict['NN']
        self.gamma = param_dict['gamma']
        self.p = param_dict['p']
        self.m = param_dict['m']

        return


    """ Reduction process """
    def reduceSet(self, X, y, param_dict):
        self.X = X
        self.y = y
        
        
        self.processParameters(param_dict)

        class_groups = self._createClassGroups()

        # Performing Label Powerset transformation:
        votes = np.zeros(shape=self.y.shape[0], dtype=np.int64)
        for single_class_group in class_groups:

            # Obtaining current label annotations:
            y_current = self.y[:, single_class_group]

            # Performing Label Powerset for the label subset considered:
            lp = LabelPowerset()
            y_lp = lp.transform(y_current)

            # Using a PG method:
            Xred, yred = self.PG.reduceSet(self.X, y_lp, self.PG_param)
            
            # 
            selection = self.closestElement(self.X, y_lp, Xred, yred)

            # Accumulating votes:
            votes[selection] += 1


        # Votes' threshold:
        ### Random subset T:
        randomSubsetIndexes = random.sample(list(range(self.X.shape[0])), int(self.p*self.X.shape[0]))
        X_T = self.X[randomSubsetIndexes]
        y_T = self.y[randomSubsetIndexes]

        ### Iterative process:
        bestI = float('inf')
        v = 0
        for u in range(y_T.shape[1]-1):
            # Instances from X with votes > u:
            X_U = self.X[votes > u]
            y_U = self.y[votes > u]

            # Training classifier over U and assessing error on subset T
            error_u = hamming_loss(y_T, np.zeros(shape=y_T.shape))
            if X_U.shape[0] > 0:
                # Train ML-based kNN classifier with U:
                clsBRkNN = BRkNNaClassifier(k = min(self.NN, X_U.shape[0])).fit(X_U, y_U)
                y_pred = clsBRkNN.predict(X_T)

                # Error:
                error_u = hamming_loss(y_T, y_pred)
                
            # Computing reduction rate:
            m_u = X_U.shape[0] / self.X.shape[0]

            # Combined figure of merit:
            I_u = self.gamma * error_u + (1-self.gamma) * m_u
            # print("u: {} - error_u: {} - m_u: {} - I_u: {}".format(u, error_u, m_u, I_u))

            # Minimizing combined figure of merit:
            if I_u < bestI and m_u > 0:
                v = u
                bestI = I_u
           
        
        # Reduced set:
        X_out = self.X[votes > v].copy()
        y_out = self.y[votes > v].copy()

        if X_out.shape[0] == 0:
            X_out = self.X
            y_out = self.y

        return X_out, y_out




if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')


    params_dict = Metamethod3.getParameterDictionary()
    X_red, y_red = Metamethod3().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)


    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red = cls_red.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")