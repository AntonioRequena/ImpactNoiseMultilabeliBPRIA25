import os
import argparse

# Importing considered scenearios:
### Base MPG case:
from expMPG import experimentMPG
### Metamethods_
from expMetamethods import experimentMetamethod
### Decomposition:
from expDecomposition import experimentDecomposition1
### Plain:
from expPlain import experimentPlain1, experimentPlain1Selection,\
    experimentPlain2

""" Argument parsing """
def parse_arguments():
    parser = argparse.ArgumentParser(description="Supervised training arguments.")
    parser.add_argument("--bbdd", type=str, default="emotions",\
        choices = ['bibtex', 'birds', 'Corel5k', 'emotions',\
        'genbase', 'medical', 'rcv1subset1', 'rcv1subset2', 'rcv1subset3',\
        'rcv1subset4', 'scene', 'yeast'], help="Corpus to process")
    
    parser.add_argument("--case", type=str, default="MPG",\
        choices = ['MPG', 'Metamethod1', 'Metamethod2', 'Metamethod3',\
            'Decomposition1', 'Plain1', 'Plain1Selection', 'Plain2',\
            'Plain3'],\
        help="Cases to process")

    parser.add_argument("--cls", type=str, default="MLkNN",\
        choices = ['BRkNNaClassifier', 'LabelPowerset', 'MLkNN'],\
        help="Classifier to use")

    parser.add_argument("--k", type=str, default="1,3,5,7",\
        help="Parameter k of the classifier")    
    
    #modificado por ARequena
    parser.add_argument("--noise", type=str, default="None", help="Noise type")

    parser.add_argument("--percen", type=str, default="0", \
        help="noise percen")

    parser.add_argument("--prob", type=str, default="0.5", \
        help="probability r")

    args = parser.parse_args()      
    args.k = [int(u) for u in args.k.split(",")]

    return args


""" Main """
if __name__ == '__main__':

    # Parsing arguments:
    args = parse_arguments()

    # Initializing results dictionary:
    res_dict = dict()

    # Creating base directories:
    Results_root_path = 'Results'
    Reduction_root_path = os.path.join(Results_root_path, 'Reduction')
    if not os.path.exists(Reduction_root_path):
        os.makedirs(Reduction_root_path)
    Classification_root_path = os.path.join(Results_root_path, 'Classification')
    if not os.path.exists(Classification_root_path):
        os.makedirs(Classification_root_path)

    # Registering DDBB:
    res_dict['DDBB'] = args.bbdd
    print("Current corpus: {}".format(args.bbdd))

    # Registering reduction case:
    res_dict['Case'] = args.case
    
    # Registering noise percen
    res_dict['noise'] = args.noise
    res_dict['percen'] = args.percen      # modificado por ARequena
    res_dict['prob'] = args.prob
    

    # Performing the reduction process:
    if args.case.startswith('Metamethod'):
        experimentMetamethod(res_dict, args.cls, args.k, Reduction_root_path, Classification_root_path)
    if args.case.startswith('Plain'):
        if args.case == 'Plain1':
            experimentPlain1(res_dict, 'kNN', args.k, Reduction_root_path, Classification_root_path)
        elif args.case == 'Plain1Selection':
            experimentPlain1Selection(res_dict, 'kNN', args.k, Reduction_root_path, Classification_root_path)
        elif args.case == 'Plain2':
            #experimentPlain2(res_dict, args.cls, int(args.k), Reduction_root_path, Classification_root_path)
            experimentPlain2(res_dict, args.cls, args.k, Reduction_root_path, Classification_root_path)
    elif args.case == 'MPG':
        experimentMPG(res_dict, args.cls, args.k, Reduction_root_path, Classification_root_path)
        #experimentMPG(res_dict, args.cls, int(args.k), Reduction_root_path, Classification_root_path)

    elif args.case == 'Decomposition1':
        experimentDecomposition1(res_dict, args.cls, args.k, Reduction_root_path, Classification_root_path)