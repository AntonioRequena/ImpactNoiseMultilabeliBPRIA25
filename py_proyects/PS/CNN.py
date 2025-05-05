import random
from tkinter import N
import numpy as np
from sklearn.utils import shuffle
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier


""" CNN """
"""  """
class CNN():

    @staticmethod
    def getFileName(param_dict):
        return 'CNN_NN-{}'.format(param_dict['NN'])


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

        selected_indexes = list()

        kNN = KNeighborsClassifier(n_neighbors=1)

        # Selecting one random sample per class:
        list_classes = np.unique(self.y)
        for single_class in list_classes:
            selected_indexes.append(random.choice(np.where(self.y == single_class)[0]))

        # Iterating through the rest of the elements to check whether they should be included:
        initial_elements = sorted(list(set(list(range(X.shape[0]))) - set(selected_indexes)))
        for idx in initial_elements:
            kNN.fit(self.X[selected_indexes], self.y[selected_indexes])
            y_pred = kNN.predict(self.X[[idx]])

            if y_pred[0] != self.y[idx]:
                selected_indexes.append(idx)


        return self.X[selected_indexes], self.y[selected_indexes], selected_indexes



if __name__ == '__main__':
    X, y = load_iris(return_X_y = True)
    
    params_dict = CNN.getParameterDictionary()
    X_red, y_red, _ = CNN().reduceSet(X, y, params_dict)


    cls_ori = KNeighborsClassifier(n_neighbors = 1).fit(X, y)
    cls_red = KNeighborsClassifier(n_neighbors = 1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X)
    y_pred_red = cls_red.predict(X_red)

    print("Results:")
    # print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    # print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")