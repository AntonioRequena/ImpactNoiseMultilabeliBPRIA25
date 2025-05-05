import numpy as np
from sklearn.metrics import hamming_loss, accuracy_score, f1_score,\
    jaccard_score, label_ranking_loss


""" Accuracy for the multilabel case """
def accuracy_ml(y_true, y_pred):
    n_elements = y_true.shape[0]

    y_true_array = y_true
    y_pred_array = y_pred

    acc = 0
    for it_element in range(n_elements):
        true_labels = set(np.where(y_true_array[it_element]==1)[0])
        estimated_labels = set(np.where(y_pred_array[it_element]==1)[0])

        try:
            acc += len(true_labels.intersection(estimated_labels)) / len(true_labels.union(estimated_labels))
        except:
            acc += 0

    return acc/n_elements


""" F1 multilabel case (in-house) """
def F1_inhouse(y_true, y_pred):
    n_elements = y_true.shape[0]

    y_true_array = y_true
    y_pred_array = y_pred

    out = 0
    for it_element in range(n_elements):
        true_labels = set(np.where(y_true_array[it_element]==1)[0])
        estimated_labels = set(np.where(y_pred_array[it_element]==1)[0])

        try:
            out += 2*len(true_labels.intersection(estimated_labels)) / (len(true_labels)\
                + len(estimated_labels))
        except:
            out += 0

    return out/n_elements


""" Computing all performance metrics given GT and estimation vectors """
def compute_metrics(y_true, y_pred):
    res = dict()

    res['HL'] = hamming_loss(y_true = y_true, y_pred = y_pred)
    res['EMR'] = accuracy_score(y_true = y_true, y_pred = y_pred)
    res['acc'] = accuracy_ml(y_true = y_true, y_pred = y_pred)
    res['jaccard-m'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'micro')
    res['F1-m'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'micro')
    res['jaccard-M'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'macro')
    res['F1-M'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'macro')
    res['jaccard-s'] = jaccard_score(y_true = y_true, y_pred = y_pred, average = 'samples')
    res['F1-s'] = f1_score(y_true = y_true, y_pred = y_pred, average = 'samples')
    res['F1_inhouse'] = F1_inhouse(y_true = y_true, y_pred = y_pred)
    res['RL'] = label_ranking_loss(y_true = y_true, y_score = y_pred)


    return res



if __name__ == '__main__':
    y_true = np.random.randint(2, size=(5,10))
    y_pred = np.random.randint(2, size=(5,10))

    res = compute_metrics(y_true = y_true, y_pred = y_pred)

    pass