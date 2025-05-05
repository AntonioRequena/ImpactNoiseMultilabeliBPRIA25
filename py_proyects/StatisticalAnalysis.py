import os
import Orange 
import numpy as np
import pandas as pd
import scikit_posthocs as sp
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare


# Params dict:
red_algos_param = {
    'RHC' : [1],
    'RSP3' : [1],
    'RSP1' : [10, 30, 50, 70, 90],
    'RSP2' : [10, 30, 50, 70, 90],
    'Chen' : [10, 30, 50, 70, 90],
}



def statistical_analyses(stat_values, names, data_dict, dst_nemenyi, dst_Bonferroni):

    st, pvalue = friedmanchisquare(stat_values[0,:], stat_values[1,:], stat_values[2,:], stat_values[3,:]) #, stat_values[4,:])

    print("Friedman test: {}".format(pvalue))

    df = pd.DataFrame(stat_values.T)
    avg_ranks = df.rank(axis=1).mean(axis=0).values

    # NEMENYI:
    cd = Orange.evaluation.scoring.compute_CD(avg_ranks, n=stat_values.T.shape[0], alpha="0.05", test="nemenyi")
    Orange.evaluation.graph_ranks(
        avranks=avg_ranks,
        names=names,
        cd=cd,
        width=6,
        textspace=1.5,
        filename = os.path.join(dst_nemenyi, '{}-{}.jpg'.format(data_dict['Method'],data_dict['Red']))
    )

    # Bonferroni-Dunn:
    reference_element_position = np.where(np.array(names) == 'MPG')[0][0]
    cd = Orange.evaluation.scoring.compute_CD(avg_ranks, n=stat_values.T.shape[0], alpha="0.05", test="bonferroni-dunn")
    res = Orange.evaluation.graph_ranks(
        avranks=avg_ranks,
        names=names,
        cd=cd,
        width=6,
        textspace=1.5,
        cdmethod = reference_element_position, #### REFERENCE (Bonferroni-Dunn)
        filename = os.path.join(dst_Bonferroni, '{}-{}.jpg'.format(data_dict['Method'],data_dict['Red']))
    )

    return pvalue


def getTestSamples(src_results:str, PG_method:str = 'Chen', red_parameter:int = 1, metric:str = 'F1-M'):

    out_values = list()
    names = list()

    df = pd.read_csv(os.path.join(src_results, 'Results_Summary.csv'))

    df_excerpt = df.loc[(df['PG_method'] == PG_method) | (df['PG_method'] == 'M'+PG_method)][df['Reduction_parameter'] == red_parameter][['DDBB','Case',metric]]

    Cases = sorted(df_excerpt['Case'].unique())
    # try:
    #     Cases.pop(Cases.index('Decomposition1')) ################
    # except:
    #     pass

    for single_case in Cases:
        out_values.append(df_excerpt.loc[df_excerpt['Case'] == single_case][metric].values)
        names.append(single_case)
    pass

    return np.array(out_values), names







def extractND(df):
    minima = [min(df['Size'].values[:u]) for u in range(1, len(df['Size'].values))]
    minima.append(df['Size'].values[-1])
    mask = (np.array(minima[1:]) < np.array(minima[:-1])).tolist()
    mask.insert(0, True)



    return mask


def getAverageSamples():

    df = pd.read_csv('./Results/Results_Summary.csv')
    df_excerpt = df[['DDBB','HL','Size','Case','PG_method','Reduction_parameter']]
    # mask = df_excerpt['PG_method'].isin(['ALL','MRSP1','MRSP2'])
    # mask = df_excerpt['PG_method'].isin(['MRSP1','MRSP2'])
    # df_excerpt = df_excerpt[~mask]


    df_excerpt = df_excerpt.groupby(['Case','PG_method','Reduction_parameter']).mean().sort_values(by=['HL', 'Size'], ascending=[True,True])
    df_excerpt.reset_index()

    mask_ND = extractND(df_excerpt)

    df_excerpt_ND = df_excerpt[mask_ND]
    return






if __name__ == '__main__':

    classifier = 'BRkNN'
    base_results = 'Results'

    dst_CD = os.path.join(base_results, 'CD_' + classifier)
    if not os.path.exists(dst_CD):
        os.makedirs(dst_CD)
    
    dst_nemenyi = os.path.join(dst_CD, 'nemenyi')
    if not os.path.exists(dst_nemenyi):
        os.makedirs(dst_nemenyi)

    dst_Bonferroni = os.path.join(dst_CD, 'Bonferroni')
    if not os.path.exists(dst_Bonferroni):
        os.makedirs(dst_Bonferroni)

    results = list()
    for single_method in list(red_algos_param.keys()):
        res_dict = dict()
        res_dict['Method'] = single_method
        for single_parameter in red_algos_param[single_method]:
            res_dict['Red'] = single_parameter

            src_results = os.path.join(base_results, 'Classification_' + classifier)

            print("Method: {} - Param: {}".format(single_method, single_parameter))
            
            stat_values, names = getTestSamples(src_results = src_results, PG_method = single_method,\
                red_parameter = single_parameter, metric = 'F1-M')
            pvalue = statistical_analyses(stat_values, names, res_dict, dst_nemenyi, dst_Bonferroni)

            res_dict['pvalue'] = pvalue
            results.append(res_dict.copy())
        pass
    pass

    df_pvalue = pd.DataFrame(results)
    df_pvalue.to_csv(os.path.join(dst_CD, 'Res_pvalues.csv'), index = False, header = True)
    pass

    getAverageSamples()
    # pass
