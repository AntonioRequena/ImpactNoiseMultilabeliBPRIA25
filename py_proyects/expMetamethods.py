import os

from MPG.ALL import ALL
from PG.RHC import RHC
from PG.Chen import Chen
from PG.RSP1 import RSP1
from PG.RSP2 import RSP2
from PG.RSP3 import RSP3

from PG2MPG.Metamethod1 import Metamethod1
from PG2MPG.Metamethod2 import Metamethod2
from PG2MPG.Metamethod3 import Metamethod3

from Common import load_corpus, obtainReducedSet, obtainResults,\
    multilabelClassification, red_algos_param

""" Function for the metamethod adaptations """
def experimentMetamethod(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)

    
    

    # Selected metamethod:
    metamethod = eval(res_dict['Case'])

    # Results file:
    results_path = os.path.join(Classification_root_path, metamethod.__name__)
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_{}.csv".format(corpus_name, classifier))

    for red_method in [ALL, RHC, Chen, RSP1, RSP2, RSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:

            params_dict = metamethod.getParameterDictionary()
            params_dict['PG'] = red_method
            params_dict['PG_param'] = params_dict['PG'].getParameterDictionary()
            params_dict['PG_param']['red'] = red_parameter

            dst_path = os.path.join(Reduction_root_path, metamethod.__name__,\
                corpus_name,params_dict['PG'].getFileName(params_dict['PG_param']))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter

            # Performing the reduction:
            X_red, y_red = obtainReducedSet(dst_path, metamethod,\
                X_train, y_train, params_dict)
        
            # Iterating through k values:
            res_dict['Classifier'] = classifier
            for single_k in k_values:
                res_dict['k'] = single_k
                
                # Performing the classification using the specified
                # multilabel classifier:
                y_pred = multilabelClassification(classifier, min(single_k, X_red.shape[0]),\
                    X_red, y_red, X_test)
                # Compute results:
                obtainResults(res_dict, y_test, y_pred,\
                    X_red, X_train, dst_results_file)

    return