import os
import pandas as pd
from Metrics import compute_metrics

# PG methods:
from MPG.ALL import ALL
from PG.RHC import RHC
from PG.Chen import Chen
from PG.RSP1 import RSP1
from PG.RSP2 import RSP2
from PG.RSP3 import RSP3

from PG2MPG.Decomposition1 import Decomposition1
from Common import load_corpus, red_algos_param


def experimentDecomposition1(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)

    # Results file:
    results_path = os.path.join(Classification_root_path, 'Decomposition1')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_{}.csv".format(corpus_name, classifier))

    for red_method in [ALL, RHC, Chen, RSP1, RSP2, RSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:

            params_dict = Decomposition1.getParameterDictionary()
            params_dict['PG'] = red_method
            params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
            params_dict['PG_param']['red'] = red_parameter

            dst_path = os.path.join(Reduction_root_path, 'Decomposition1', corpus_name,\
                params_dict['PG'].getFileName(params_dict['PG_param']))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter
            X_dst_file = os.path.join(dst_path, 'X_{}.npz')
            y_dst_file = os.path.join(dst_path, 'y_{}.npz')

            # Reduction stage:
            for it_label in range(y_train.shape[1]):
                X_dst_file_current_label = X_dst_file.replace("{}", str(it_label))
                y_dst_file_current_label = y_dst_file.replace("{}", str(it_label))

                if (not os.path.isfile(X_dst_file_current_label)) or \
                    (not os.path.isfile(y_dst_file_current_label)):
                    X_red, y_red = Decomposition1().reduceSetSingleLabel(X_train,\
                        y_train[:, it_label], params_dict)

                    pd.DataFrame(X_red).to_csv(X_dst_file_current_label, header=None,\
                        index=None, compression='gzip')
                    pd.DataFrame(y_red).to_csv(y_dst_file_current_label, header=None,\
                        index=None, compression='gzip')

            # Iterating through k values:
            for single_k in k_values:
                res_dict['k'] = single_k

                # Performing classification:
                y_pred, size_temp = Decomposition1().classifyAllLabels(X_test = X_test,\
                    X_file = X_dst_file,y_file = y_dst_file,\
                    n_labels = y_train.shape[1], n_neigh = single_k)

                # Computing metrics:
                res_dict.update(compute_metrics(y_true = y_test, y_pred = y_pred))
                res_dict['Size'] = 100*size_temp/X_train.shape[0]

                # Writing results:
                if os.path.isfile(dst_results_file):
                    out_file = pd.read_csv(dst_results_file)
                    out_file.loc[len(out_file)] = res_dict
                else:
                    out_file = pd.DataFrame([res_dict])
                out_file.to_csv(dst_results_file,index=False)

            pass
        pass
    pass
    return