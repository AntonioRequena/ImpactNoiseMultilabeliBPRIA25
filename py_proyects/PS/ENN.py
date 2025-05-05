import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


""" Wilson's ENN """
"""  """
class ENN():

    @staticmethod
    def getFileName(param_dict):
        return 'ENN_NN-{}'.format(param_dict['NN'])


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['NN'] = 5
        return param_dict


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.NN = param_dict['NN']
        return


    """ Reduction process """
    def reduceSet(self, X, y, param_dict):
        self.X, self.y = shuffle(X, y, random_state = 0)

        # Processing parameters:
        self.processParameters(param_dict)

        # kNN classifier:
        kNN = KNeighborsClassifier(n_neighbors = self.NN)

        # Iterating through the initial set:
        vector_delete = list()
        for it_sample in range(self.X.shape[0]):
            # "Training" kNN classifier:
            kNN.fit(np.delete(self.X, it_sample, axis = 0), np.delete(self.y, it_sample, axis = 0))

            # Predicting sample:
            y_pred = kNN.predict(self.X[[it_sample]])

            # If prediction mismatches GT -> delete sample:
            if y_pred[0] != self.y[it_sample]: vector_delete.append(it_sample)

        # Deleting samples:
        X_out = np.delete(self.X, vector_delete, axis = 0)
        y_out = np.delete(self.y, vector_delete, axis = 0)

        # Indexes of selected samples:
        selected_indexes = sorted(list(set(list(range(X.shape[0]))) - set(vector_delete)))

        return X_out, y_out, selected_indexes



if __name__ == '__main__':
    X, y = load_iris(return_X_y = True)
    
    params_dict = ENN.getParameterDictionary()
    X_red, y_red, _ = ENN().reduceSet(X, y, params_dict)


    cls_ori = KNeighborsClassifier(n_neighbors = 1).fit(X, y)
    cls_red = KNeighborsClassifier(n_neighbors = 1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X)
    y_pred_red = cls_red.predict(X_red)

    print("Results:")
    # print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    # print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")