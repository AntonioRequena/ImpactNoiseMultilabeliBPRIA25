import numpy as np
from sklearn.metrics import hamming_loss
from skmultilearn.dataset import load_dataset
from skmultilearn.adapt import BRkNNaClassifier
from sklearn.neighbors import NearestNeighbors

""" Edited Multilabel Classification """
""" Editing training data for multi-label classification with the k-nearest neighbor rule (Kanj et al., 2016) """
class EMLC():


    @staticmethod
    def getFileName(param_dict):
        return 'EMLC_NN-{}_l-{}_thresholdHL-{}'.format(param_dict['NN'], param_dict['l'], param_dict['thresholdHL'])


    @staticmethod
    def getParameterDictionary():
        param_dict = dict()
        param_dict['NN'] = 5
        param_dict['l'] = 50
        param_dict['thresholdHL'] = 0.05
        return param_dict


    """ Process reduction parameters """
    def processParameters(self, param_dict):
        self.NN = param_dict['NN']
        self.l = param_dict['l']
        self.thresholdHL = param_dict['thresholdHL']
        return


    """ Reduction process """
    def reduceSet(self, X, y, param_dict):
        self.X_out = X.copy()
        self.y_out = y.copy()


        # Processing parameters:
        self.processParameters(param_dict)

        meanHL = np.float('inf')
        # it = 0
        while self.thresholdHL < meanHL:
            # print("Iteration #{} - meanHL : {}".format(it, meanHL))

            # Estimating HL per sample:
            HLoss = list()
            for it_sample in range(self.X_out.shape[0]):
                X_temp = np.delete(self.X_out, it_sample, axis = 0)
                y_temp = np.delete(self.y_out, it_sample, axis = 0)

                # ML-based kNN classifier:
                clsBRkNN = BRkNNaClassifier(k = self.NN).fit(X_temp, y_temp)

                # Prediction:
                y_pred = clsBRkNN.predict(np.expand_dims(self.X_out[it_sample], 0))

                # i-th sample HL loss:
                HLoss.append(hamming_loss(self.y_out[[it_sample]],y_pred))
                
            # Mean HL:
            meanHL = np.mean(HLoss)

            # Assessing whether mean HL is higher than threshold:
            if meanHL > self.thresholdHL:
                indices = list(range(self.X_out.shape[0]))
                _, indices = zip(*sorted(zip(HLoss, indices), reverse = True))

                # Removing 'most erroneous' self.l samples:
                self.X_out = np.delete(self.X_out, indices[:self.l], axis = 0)
                self.y_out = np.delete(self.y_out, indices[:self.l], axis = 0)

                # it += 1


        return self.X_out, self.y_out




if __name__ == '__main__':
    X_train, y_train, feature_names, label_names = load_dataset('yeast', 'train')
    X_test, y_test, feature_names, label_names = load_dataset('yeast', 'test')

    params_dict = EMLC.getParameterDictionary()
    X_red, y_red = EMLC().reduceSet(X_train.toarray().copy(), y_train.toarray().copy(), params_dict)




    cls_ori = BRkNNaClassifier(k=1).fit(X_train, y_train)
    cls_red = BRkNNaClassifier(k=1).fit(X_red, y_red)

    y_pred_ori = cls_ori.predict(X_test)
    y_pred_red = cls_red.predict(X_test)

    print("Results:")
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_ori), 100*X_train.shape[0]/X_train.shape[0]))
    print("\t - Hamming Loss: {:.3f} - Size: {:.1f}%".format(hamming_loss(y_test, y_pred_red), 100*X_red.shape[0]/X_train.shape[0]))
    print("Done!")



