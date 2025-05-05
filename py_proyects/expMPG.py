import os
from Common import load_corpus, obtainReducedSet, obtainResults,\
    multilabelClassification, red_algos_param

# MPG methods:
from MPG.ALL import ALL
from MPG.MRHC import MRHC
from MPG.MChen import MChen
from MPG.MRSP1 import MRSP1
from MPG.MRSP2 import MRSP2
from MPG.MRSP3 import MRSP3



from noise import noise_swap, noise_add, noise_PUMN, noise_sub, noise_add_sub, noise_DAAS
from scipy.sparse import issparse
import numpy as np



""" Function that considers the multilabel PG schemes """
def experimentMPG(res_dict, classifier, k_values, Reduction_root_path, Classification_root_path):

    # Loading current corpus:
    corpus_name = res_dict['DDBB']
    X_train, y_train, X_test, y_test = load_corpus(corpus_name)
    
   
    #print("Primeros datos de X_train:", X_train[:20])
    #print("Primeras etiquetas de y_train:", y_train[:20])
    # codigo para añadir ruido.
    
    tipo_ruido = ""
    tipo_ruido = res_dict['noise']
    porcentaje = int(res_dict['percen'])
    probabilidad = round(float(res_dict['prob']), 2)
    
    
    if tipo_ruido == "swap":
        y_out = noise_swap(y_train, porcentaje)
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
        y_train = y_out.copy()
    if tipo_ruido == "add":
        y_out = noise_add(y_train, porcentaje, probabilidad)
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
        y_train = y_out.copy()
    if tipo_ruido == "PUMN":
        y_out = noise_PUMN(y_train, porcentaje, probabilidad)
        y_train = y_out.copy()
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
    if tipo_ruido == "sub":
        y_out = noise_sub(y_train, porcentaje, probabilidad)
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
        y_train = y_out.copy()
    if tipo_ruido == "add-sub":
        y_out = noise_add_sub(y_train, porcentaje, probabilidad)
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
        y_train = y_out.copy()
    if tipo_ruido == "DAAS":
        y_out = noise_DAAS(y_train, porcentaje, probabilidad)
                    #y_out que es el conjunto de entrenamiento con ruido, lo asigno a y_train y sigo el algoritmo de igual manera.
        y_train = y_out.copy()
        
        
    # Results file:
    results_path = os.path.join(Classification_root_path, 'MPG')
    if not os.path.exists(results_path):
        os.makedirs(results_path)
    dst_results_file = os.path.join(results_path, "Results_{}_{}.csv".format(corpus_name, classifier))

    for red_method in [ALL, MRHC, MChen, MRSP1, MRSP2, MRSP3]:
        for red_parameter in red_algos_param[red_method.__name__]:
            
            params_dict = red_method.getParameterDictionary()
            params_dict['red'] = red_parameter
            dst_path = os.path.join(Reduction_root_path, 'MPG', corpus_name,\
                red_method.getFileName(params_dict))
            if not os.path.exists(dst_path):
                os.makedirs(dst_path)

            res_dict['PG_method'] = red_method.__name__
            res_dict['Reduction_parameter'] = red_parameter


            # Performing the reduction:
            X_red, y_red = obtainReducedSet(dst_path, red_method,\
                X_train, y_train, params_dict, res_dict)                       #modificado por Antonio Requena
            

            res_dict['Classifier'] = classifier
            # Iterating through k values:
            for single_k in k_values:
                res_dict['k'] = single_k
                
                # Performing the classification using the specified
                # multilabel classifier:
                y_pred = multilabelClassification(classifier, single_k,\
                    X_red, y_red, X_test)

                #print("Resultados antes de obtainResults: ", res_dict)

                # Compute results:
                obtainResults(res_dict, y_test, y_pred,\
                    X_red, X_train, dst_results_file)
            
            pass
        pass
    pass

    return