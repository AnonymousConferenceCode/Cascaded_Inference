'''
This script reads the results of the gradual networks
script runs for different networks.

It can perform aggregation of the results using
 a) variable deltas (confidence thresholds)
 b) variable epsilon (accuracy compromise)

#######################################################
###                                                 ###
###     Be An Accurate Researcher !                 ###
###                                                 ###
###     Before running this script on the results,  ###
###     make sure that you checked *ALL* of the     ###
###     "PHASE I - Configuration" section           ###
###                                                 ###
#######################################################

By Anonymous
'''
from Utils.plot import plotListOfScatters, plotListOfPlots
from Utils.LogDrivenUtils.multiple_delta import aggregate_hierarchical_model_performance
from Utils.LogDrivenUtils.cst import aggregate_hierarchical_model_performance_cst
from Utils.LogDrivenUtils.ccst import aggregate_hierarchical_model_performance_ccst
import numpy as np
import logging
import argparse
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug



if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Reads the softmax dumps and finds confidence'
                    'threshold triplets (delta0,delta1,delta2) that correspond '
                    'to a wide range of epsilons. Each of these triplets is saved in a '
                    'Logs directory in a text file "log_resnet50_epsilon<e>.txt".')
    parser.add_argument('-dataset_path', type=str,
                        default='/data/datasets/Imagenet',
                        help='a path to an Imagenet directory, where a "tf_records" '
                             'directory must reside.')
    parser.add_argument('-softmax_path', type=str,
                        default='../TF_Models/official/resnet/output_softmax',
                        help='The softmax_dir, where clf0, clf1, clf2'
                             'are softmax dumps for train and test must be located.')
    parser.add_argument('--top5', action="store_true",
                        help='If set, then the log that will be created will contain'
                             ' the deltas computed for top-5 cascaded inference')
    parser.add_argument('-output_dir', type=str,
                        default='output_log_processing',
                        help='Where all the figures are about to be created')

    args = parser.parse_args()


    nDeltas             = 1000
    nClassifiers        = 3
    input_dir           = 'output_softmax'
    output_dir          = args.output_dir
    dataset_dir         = args.dataset_path

    #### ICML`19 Imagenet Resnet-50-v2 with CST (Classifier-Specific-Thresholds with full datasets) #####

    show_top5_accuracy = args.top5
    useCachedResults = False

    epsilon_lst = [0,1,1.6,1.7,1.8,1.9,2,2.5,2.6,2.7,2.8,3,4,5.1,5.2,5.3,5.4,6,7] if show_top5_accuracy else range(21)
    dataset_name        = 'Imagenet'
    input_dir           = args.softmax_path

    log_dir             = "Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top5" if show_top5_accuracy else "Logs/IMAGENET_Resnet50_Logs_with_cst_deltas_top1"
    imagenet_resnet50_num_gigamacs = 4.037883817

    num_ccst_models = 0
    num_ccst_models_f =0
    num_cst_models = len(epsilon_lst)
    num_cst_models_f = 0
    num_global_threshold_models = 1
    num_models = num_ccst_models + num_cst_models + num_global_threshold_models + num_ccst_models_f + num_cst_models_f

    cst_modelnames  = ["resnet50_epsilon{:04d}".format(int(eps*100)) for eps in epsilon_lst]
    primary_modelname_lst = cst_modelnames
    dict_modelname_lst = ["resnet50"]*len(primary_modelname_lst)

    top5_appendix = "_top5" if show_top5_accuracy else ""
    representing_name_for_plots = dataset_name + "_ResNet_CST_ICML19" + top5_appendix
    title_plot = ""
    legend_list = ["Our",r"Bolukbasi $et$ $al.$ $2017$"] if show_top5_accuracy else None

    GFLOP_per_inference_lst = [imagenet_resnet50_num_gigamacs]*(num_models-num_global_threshold_models)
    runType_lst= ['classifier_and_class_specific_thresholds_f']*num_ccst_models_f+ \
                 ['classifier_specific_thresholds_f']*num_cst_models_f+ \
                 ['classifier_and_class_specific_thresholds'] * num_ccst_models + \
                 ['classifier_specific_thresholds'] * num_cst_models + \
                 ['global_threshold']*num_global_threshold_models
    fontsize=13
    # Initialize accumulators for plot X-Y axis. These axis will be used as X and Y vectors
    # in the further plot generation.
    mac_lst_of_lists = []
    acc_lst_of_lists = []
    mac_cst = []
    acc_cst = []
    mac_ccst = []
    acc_ccst = []
    mac_cst_f = []
    acc_cst_f = []
    mac_ccst_f = []
    acc_ccst_f = []
    acc_other_authors = [0.9321,0.9221,0.9121,0.8821] if show_top5_accuracy else []             # Reported by Bolukbasi etal in "Adaptve Neural Networks for Efficient Inference"
    mac_other_authors = [imagenet_resnet50_num_gigamacs,                                        # Reported by Bolukbasi etal in "Adaptve Neural Networks for Efficient Inference"
                         imagenet_resnet50_num_gigamacs/1.08,                                   # Reported by Bolukbasi etal in "Adaptve Neural Networks for Efficient Inference"
                         imagenet_resnet50_num_gigamacs/1.18,                                   # Reported by Bolukbasi etal in "Adaptve Neural Networks for Efficient Inference"
                         imagenet_resnet50_num_gigamacs/1.22] if show_top5_accuracy else []     # Reported by Bolukbasi etal in "Adaptve Neural Networks for Efficient Inference"


    # Read the recorded softmax outputs of all the models and obtain the X,Y vectors of
    # 1) MAC count (will be further used as a Y axis)
    # 2) accuracy (will be further used as an X axis)
    for primary_modelname, dict_modelname, baselineGFLOPsPerInference, runType in zip(primary_modelname_lst, dict_modelname_lst, GFLOP_per_inference_lst, runType_lst):
        if 'classifier_and_class_specific_thresholds' in runType:
            mac_tt, acc_tt = aggregate_hierarchical_model_performance_ccst(nClassifiers,
                                                                primary_modelname,
                                                                dict_modelname,
                                                                input_dir,
                                                                output_dir,
                                                                dataset_name,
                                                                dataset_dir,
                                                                log_dir,
                                                                baselineGFLOPsPerInference,
                                                                useCachedResults=useCachedResults)
            if "_f" in runType:
                mac_ccst_f += mac_tt
                acc_ccst_f += acc_tt
            else:
                mac_ccst += mac_tt
                acc_ccst += acc_tt

        elif 'classifier_specific_thresholds' in runType:
            mac_tt, acc_tt = aggregate_hierarchical_model_performance_cst(nClassifiers,
                                                                          primary_modelname,
                                                                          dict_modelname,
                                                                          input_dir,
                                                                          output_dir,
                                                                          dataset_name,
                                                                          dataset_dir,
                                                                          log_dir,
                                                                          baselineGFLOPsPerInference,
                                                                          useCachedResults=useCachedResults,
                                                                          return_topK=show_top5_accuracy,
                                                                          produce_figures=False)
            if "_f" in runType:
                mac_cst_f += mac_tt
                acc_cst_f += acc_tt
            else:
                mac_cst += mac_tt
                acc_cst += acc_tt

        elif runType == 'global_threshold':
            mac, acc = aggregate_hierarchical_model_performance(nDeltas,
                                                                nClassifiers,
                                                                primary_modelname,
                                                                dict_modelname,
                                                                input_dir,
                                                                output_dir,
                                                                log_dir,
                                                                dataset_name,
                                                                dataset_dir,
                                                                baselineGFLOPsPerInference,
                                                                useCachedResults=useCachedResults)
            mac_lst_of_lists.append(mac)
            acc_lst_of_lists.append(acc)
        else:
            ld("No")

    # Stack the X,Y vectors of different models
    if len(acc_cst)>0 and len(mac_cst)>0:
            mac_lst_of_lists = [mac_cst] + mac_lst_of_lists
            acc_lst_of_lists = [acc_cst] + acc_lst_of_lists
    if len(acc_ccst)>0 and len(mac_ccst)>0:
            mac_lst_of_lists = [mac_ccst] + mac_lst_of_lists
            acc_lst_of_lists = [acc_ccst] + acc_lst_of_lists
    if len(acc_cst_f)>0 and len(mac_cst_f)>0:
            mac_lst_of_lists = [mac_cst_f] + mac_lst_of_lists
            acc_lst_of_lists = [acc_cst_f] + acc_lst_of_lists
    if len(acc_ccst_f)>0 and len(mac_ccst_f)>0:
            mac_lst_of_lists = [mac_ccst_f] + mac_lst_of_lists
            acc_lst_of_lists = [acc_ccst_f] + acc_lst_of_lists
    if len(acc_other_authors) > 0 and len(mac_other_authors) > 0:
        mac_lst_of_lists = mac_lst_of_lists + [mac_other_authors]
        acc_lst_of_lists = acc_lst_of_lists + [acc_other_authors]

    # Generate the plot of all the stacked models
    plotListOfPlots(mac_lst_of_lists,
                       acc_lst_of_lists,
                       legend_list,
                       "Giga-MACs per inference",
                       "Accuracy",
                       title_plot,
                       output_dir,
                       representing_name_for_plots + '_Acc_Mac.png',
                        fontsize=fontsize,
                        showGrid=True)
