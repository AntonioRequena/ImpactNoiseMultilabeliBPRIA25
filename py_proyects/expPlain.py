import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from MPG.ALL import ALL
from PG.RHC import RHC
from PG.Chen import Chen
from PG.RSP1 import RSP1
from PG.RSP2 import RSP2
from PG.RSP3 import RSP3

from PG2MPG.Plain1 import Plain1
from PG2MPG.Plain1Selection import Plain1Selection
from PG2MPG.Plain2 import Plain2

from Metrics import compute_metrics
from Common import load_corpus, red_algos_param


""" Plain 1 """
def experimentPlain1(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)

    # Results file:
    results_path = os.path.join(Classification_root_path, res_dict['Case'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_kNN.csv".format(corpus_name))

    
    #for red_method in [ALL, RHC, Chen, RSP1, RSP2, RSP3]:
    for red_method in [RSP2, RSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:

            params_dict = Plain1.getParameterDictionary()
            params_dict['PG'] = red_method
            params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
            params_dict['PG_param']['red'] = red_parameter


            dst_path = os.path.join(Reduction_root_path, 'Plain1', corpus_name,\
                params_dict['PG'].getFileName(params_dict['PG_param']))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter
            res_dict['Classifier'] = classifier
            X_dst_file = os.path.join(dst_path, 'X_{}.npz')
            y_dst_file = os.path.join(dst_path, 'y_{}.npz')

            # Reduction stage:
            for it_label in range(y_train.shape[1]):
                X_dst_file_current_label = X_dst_file.replace("{}", str(it_label))
                y_dst_file_current_label = y_dst_file.replace("{}", str(it_label))

                if (not os.path.isfile(X_dst_file_current_label)) or \
                    (not os.path.isfile(y_dst_file_current_label)):
                    X_red, y_red = Plain1().reduceSetSingleLabel(X_train,\
                        y_train[:, it_label], params_dict)

                    pd.DataFrame(X_red).to_csv(X_dst_file_current_label, header=None,\
                        index=None, compression='gzip')
                    pd.DataFrame(y_red).to_csv(y_dst_file_current_label, header=None,\
                        index=None, compression='gzip')

            # Iterating through k values:
            for single_k in k_values:
                res_dict['k'] = single_k

                # Performing classification:
                y_pred, size_temp = Plain1().classifyAllLabels(X_test = X_test,\
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


""" Plain 1 Selection """
def experimentPlain1Selection(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)

    # Results file:
    results_path = os.path.join(Classification_root_path, res_dict['Case'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_kNN.csv".format(corpus_name))

    for red_method in [ALL, RHC, Chen, RSP1, RSP2, RSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:

            params_dict = Plain1Selection.getParameterDictionary()
            params_dict['PG'] = red_method
            params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
            params_dict['PG_param']['red'] = red_parameter


            dst_path = os.path.join(Reduction_root_path, 'Plain1Selection', corpus_name,\
                params_dict['PG'].getFileName(params_dict['PG_param']))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter
            res_dict['Classifier'] = classifier
            X_dst_file = os.path.join(dst_path, 'X_{}.npz')
            y_dst_file = os.path.join(dst_path, 'y_{}.npz')

            # Reduction stage:
            for it_label in range(y_train.shape[1]):
                X_dst_file_current_label = X_dst_file.replace("{}", str(it_label))
                y_dst_file_current_label = y_dst_file.replace("{}", str(it_label))

                if (not os.path.isfile(X_dst_file_current_label)) or \
                    (not os.path.isfile(y_dst_file_current_label)):
                    X_red, y_red = Plain1Selection().reduceSetSingleLabel(X_train,\
                        y_train[:, it_label], params_dict)

                    pd.DataFrame(X_red).to_csv(X_dst_file_current_label, header=None,\
                        index=None, compression='gzip')
                    pd.DataFrame(y_red).to_csv(y_dst_file_current_label, header=None,\
                        index=None, compression='gzip')

            # Iterating through k values:
            for single_k in k_values:
                res_dict['k'] = single_k

                # Performing classification:
                y_pred, size_temp = Plain1Selection().classifyAllLabels(X_test = X_test,\
                    X_file = X_dst_file,y_file = y_dst_file,\
                     X_train = X_train, y_train = y_train,\
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



""" Plain 2 """
def experimentPlain2(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)

    # Results file:
    results_path = os.path.join(Classification_root_path, res_dict['Case'])
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_{}.csv".format(corpus_name, classifier))

    for red_method in [ALL, RHC, Chen, RSP1, RSP2, RSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:

            params_dict = Plain2.getParameterDictionary()
            params_dict['PG'] = red_method
            params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
            params_dict['PG_param']['red'] = red_parameter


            dst_path = os.path.join(Reduction_root_path, 'Plain2', corpus_name,\
                params_dict['PG'].getFileName(params_dict['PG_param']))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter
            res_dict['Classifier'] = classifier
            X_dst_file = os.path.join(dst_path, 'X.npz')
            y_dst_file = os.path.join(dst_path, 'y.npz')

            # Label encoding:
            le = LabelEncoder()
            le.fit([str(y) for y in y_train])

            if (not os.path.isfile(X_dst_file)) or \
                (not os.path.isfile(y_dst_file)):
                X_red, y_red_lp = Plain2().reduceSet(X_train,\
                    y_train, params_dict, le)

                pd.DataFrame(X_red).to_csv(X_dst_file, header=None,\
                    index=None, compression='gzip')
                pd.DataFrame(y_red_lp).to_csv(y_dst_file, header=None,\
                    index=None, compression='gzip')
            
            # Iterating through k values:
            for single_k in k_values:
                res_dict['k'] = single_k

                #xxa=input("Llegue hasta aqui -1")
                #print("X_test:", X_test)
                #xxa=input("Llegue hasta aqui -2")
                
                
                y_pred, red_size = Plain2().classify(X_test, X_dst_file, y_dst_file, classifier, single_k, le)
                #xxa=input("Llegue hasta aqui -3")
               
                
                # Computing metrics:
                res_dict.update(compute_metrics(y_true = y_test, y_pred = y_pred))
                res_dict['Size'] = 100*red_size/X_train.shape[0]

                # Writing results:
                if os.path.isfile(dst_results_file):
                    out_file = pd.read_csv(dst_results_file)
                    out_file.loc[len(out_file)] = res_dict
                else:
                    out_file = pd.DataFrame([res_dict])
                out_file.to_csv(dst_results_file,index=False)