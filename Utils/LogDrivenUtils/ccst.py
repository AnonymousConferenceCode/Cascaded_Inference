'''
This function reads the results of the gradual networks
script runs for different networks. Next, it
simulates a hierarchical inference using:
 " Classifier and Class specific thrsholds (CCST) "

over the models specified in a modelnames_list list below.
Namely, when for an input x, the i^th classifier in the cascade
predicts that the class is j, then this prediction will have
to surpass the threshold Delta(i,j) in order to quit
further computation. Where Delta is a matrix of thresholds,
obtained from the BT-C2ST training algorithm.

Anonymized Source Code
'''
import sys
import os
from math import ceil, log
# from datasets import dataset_utils
import tensorflow as tf
import numpy as np
import itertools
import pickle
import logging
import json
from utils import *
import itertools
import ast

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug


def aggregate_hierarchical_model_performance_ccst(nClassifiers,
                                                 primary_modelname,
                                                 dict_modelname,
                                                 input_dir,
                                                 output_dir,
                                                 dataset_name,
                                                 dataset_dir,
                                                 log_dir,
                                                 baselineGFLOPsPerInference,
                                                 useCachedResults=False):
    '''
    TODO: refine the folllowing documentation:
    Reads the separate test-results from files:
        <input_dir>/<model_name>_clf*_top5_classes.npy
        <input_dir>/<model_name>_clf*_all_probabilities.npy

    for every sub-classifier (* stands for classifier id in the hierarchical model)
    Then the function calculates the required number of MAC loperations and accuracy
    as a function of various confidence_thresholds (deltas).

    Generates plots&stats.txt file and places them into the directory
        <output_dir>/<model_name>/*

    :param nDeltas: number of different confidence-thresholds checked for the plots
    :param nClassifiers: number of different classifiers checked in the
                         network {clf0,clf1,...clf(nClassifiers-1)}
    :param primary_modelname: the primary name of the hierarchical network (excluding the "_clfi" postfix)
    :param dict_modelname: the name of the model to be read from the model_dict for the sake of
                           the #MACs
    :param input_dir: the directory where the classes and the probabilities files will be read from
    :param output_dir: the directory where a "modelname" subdirectory will be created, for
                    containing the results of this script
    :param dataset_name: "SVHN", "CIFAR10", "CIFAR100"
    :param dataset_dir: a path to the directory where the dataset directory can be found
    :param log_dir: a path to where the log files can be found
    :param baselineGFLOPsPerInference: number of GIGA-MACs required for a single input inference,
                                        when using a non hierarchical model, evaluated through
                                        all the CONV layers up to clf_(nClassifiers-1).


    :return: two lists which correspond to each other and are of the length of nDeltas:
                1) number of MACs required by the hierarchical model for each delta
                2) top-1 accuracies achieved by the hierarchical model for each delta

    '''

    ###############################################################################
    ###################### PHASE I - Configuration ################################
    ###############################################################################

    topK = 5
    epsilon_lst = map(lambda x: x / float(100),
                            range(0, (100 + 1), 1))  # A.K.A. epsilons to be plotted as the X-axis
    output_filename = 'Results.json'  # Path to the output file with all the numerical result of this script
    labels_filename = 'test_labels_non_onehot.npy'  # Name of the labels file, containing a 1-D numpy array with the ground
    modelnames_list = [primary_modelname + "_clf" + str(i) for i in range(nClassifiers)]
    dict_modelname_list = [dict_modelname + "_clf" + str(i) for i in range(nClassifiers)]
    classifier_legends = [r"$M_{}$".format(i) for i in range(nClassifiers)]
    reReadResult = not useCachedResults  # if true then the results from the input folder will be re-run prior to plots generation
    # if false then the Result.json will be loaded for plot generation

    # For backtrack-doubling network baseline printout:
    showBaseline = True  # if true then two plots will be shown  - one of them is the baseline performance of the non-GII version
    perCLFaccuracy = []
    perCLFperCLSthreshold = None
    for modelname, clf_id in zip(modelnames_list, range(nClassifiers)):
        with open(os.path.join(log_dir, "log_" + modelname + ".txt"), "r") as f:
            lineList = f.readlines()
            precentage_str = lineList[-1].split(" ")[-1]  # This gets you the "87.7%" string
            perCLFaccuracy.append(float(precentage_str.split("%")[0]) / 100.0)

            clf_thresh_str = lineList[-2].split(":")[-1]  # This gets you the "[0.904487, 0.2, ...,0.65]" string
            clf_thresh_lst = ast.literal_eval(clf_thresh_str)
            clf_thresh_vec = np.array(clf_thresh_lst, dtype=np.float32)

            if perCLFperCLSthreshold is None:
                perCLFperCLSthreshold = np.zeros((nClassifiers,clf_thresh_vec.shape[0]))
            perCLFperCLSthreshold[clf_id, :] = clf_thresh_vec

            accuracy_compromise_str = lineList[-3].split(":")[-1]  # This gets you the "0.904487" string
            accuracy_compromise = float(accuracy_compromise_str)

    baselineTop1Accuracy = perCLFaccuracy[-1]

    # Create a private directory for the model being analyzed
    output_dir = os.path.join(output_dir, primary_modelname)
    if (not os.path.exists(output_dir)):
        ld("Creating temporary directory for log driven analysis products: " + output_dir)
        os.makedirs(output_dir)

    # For backtrack-thin network baseline printout:
    # showBaseline    = True # if true then two plots will be shown  - one of them is the baseline performance of the non-GII version
    # baselineGFLOPsPerInference = 0.119708415
    # baselineTop1Accuracy = 0.633

    checkpoints_dir = '../Hierarchical/checkpoints'
    models_dict_filename = 'Model_dict.json'
    path_to_json = os.path.join(checkpoints_dir, models_dict_filename)
    if os.path.exists(path_to_json):
        with open(path_to_json, "r") as f:
            json_data = f.read()
            model_dict = json.loads(json_data)
    else:
        ld("Error, no model dictionary found at " + path_to_json)

    ###############################################################################
    ####### PHASE II.a - parse logs and generate statistics in a dictionary########
    ###############################################################################

    Results_dict = {}

    if reReadResult:

        Probabilities_dict = {}
        Topclasses_dict = {}

        LABELS_ORIGINAL = readNPY(os.path.join(dataset_dir, dataset_name, labels_filename))
        nExaminedImages = len(LABELS_ORIGINAL)

        Results_dict['nExaminedImages'] = nExaminedImages
        ld("Evaluating {prim} model using dictionary model {dct} for epsilon={eps}.".format(prim=primary_modelname,
                                                                                            dct=dict_modelname,
                                                                                            eps=accuracy_compromise))
        Epsilon_dict = {}

        # this is cool. it stores the indices mapping from the initially run samples'
        # indices to the most recently surviived ones
        indices_mapping_composition_lst = [range(len(LABELS_ORIGINAL))]


        for modelname, clf_id in zip(modelnames_list, range(nClassifiers)):
            with open(os.path.join(input_dir, modelname + "_all_probabilities.npy"), "rb") as probFile:
                Probabilities_dict[modelname] = pickle.load(probFile)
            with open(os.path.join(input_dir, modelname + "_top" + str(topK) + "_classes.npy"), "rb") as classFile:
                Topclasses_dict[modelname] = pickle.load(classFile)

            Model_dict = {}

            # Take the predictions from the pre-loaded log files.
            if (not (modelname in Topclasses_dict)) or (not (modelname in Probabilities_dict)):
                ld("Error, the model named " + modelname + " was not found in the output files)")
                os.exit(-1)
            classes = Topclasses_dict[modelname]
            probabilities = Probabilities_dict[modelname]

            ld("{} - Using confidence threshold list: {}.".format(modelname, perCLFperCLSthreshold[clf_id,:].tolist()))

            # Perform multiple index re-mappings until you reach the mapping
            # implied by this hierarchy level.
            # It is needed to simulate the input samples that "survived"
            # until the current hierarchy-inference level because they
            # didn't satisfy the confidence-threshold (delta) during the
            # previous hierarchy levels.
            # debug=0
            LABELS = np.copy(LABELS_ORIGINAL)
            # ld("mapping list is of the length " + `len(indices_mapping_composition_lst)`)
            for index_mapping in indices_mapping_composition_lst:
                # ld("index_mapping " + `debug` + " len is:")
                # ld(len(index_mapping))
                # debug += 1
                # ld("index mapping is are of the shape:")
                # ld(np.shape(index_mapping))
                LABELS = LABELS[index_mapping]
                # ld("LABELS became of the shape:")
                # ld(np.shape(LABELS))
                classes = classes[index_mapping, :]
                probabilities = probabilities[index_mapping, :]

            nSamples = len(LABELS)
            ld("\t" + modelname + " - running " + str(nSamples) + " samples")
            assert (np.shape(classes)[0] == np.shape(probabilities)[0] == nSamples)

            # "indicators" are lists of "nSamples"-long boolean-vectors.
            correct_indicators = [(classes[:, i] == LABELS) for i in range(topK)]
            if modelname == modelnames_list[-1]:
                # stupid tautology in order to effectively reduce the threshold to zero!
                confident_indicators = [np.ones(nSamples, dtype=np.bool) for i in range(topK)]
            else:
                # Conident-indicators list contains vectors, a totak of K vectors:
                # - vector number i corresponds to the i^th prediction of the classifier (i in {0,...,nClasses-1})
                #   for example the first vector in the confident_indicators contains "1" at every sample which was
                #   with the probability higher than the threshold for the predicted class.
                confident_indicators = [(np.sort(probabilities)[:, -i - 1] >= perCLFperCLSthreshold[clf_id,:].take(classes[:,i])) for i in range(topK)]
            correct_and_confident_indicators = [np.logical_and(confident_indicators[i], correct_indicators[i]) for i in range(topK)]

            nCorrect_top1 = np.sum(correct_indicators[0])
            nConfident_top1 = np.sum(confident_indicators[0])
            nCorrect_and_confident_top1 = np.sum(correct_and_confident_indicators[0])
            accuracy_top1 = 100 * nCorrect_top1 / float(nSamples) if nSamples > 0 else 0
            confident_accuracy_top1 = 100 * nCorrect_and_confident_top1 / float(
                nConfident_top1) if nConfident_top1 > 0 else 0

            correct_topK = np.zeros(nSamples, dtype=bool)
            correct_and_confident_topK = np.zeros(nSamples, dtype=bool)
            for correct_vector, correct_and_confident_vector in zip(correct_indicators,
                                                                    correct_and_confident_indicators):
                correct_topK = np.logical_or(correct_topK, correct_vector)
                correct_and_confident_topK = np.logical_or(correct_and_confident_topK, correct_and_confident_vector)
            nCorrect_topK = np.sum(correct_topK)
            nConfident_topK = nConfident_top1  # if the top probability won't satisfy the threshold, no other probability will
            nCorrect_and_confident_topK = np.sum(correct_and_confident_topK)
            accuracy_topK = 100 * nCorrect_topK / float(nSamples) if nSamples > 0 else 0
            confident_accuracy_topK = 100 * nCorrect_and_confident_topK / float(
                nConfident_topK) if nConfident_topK > 0 else 0

            # Statistics
            Model_dict["n_correct_top1_answers"] = nCorrect_top1
            Model_dict["n_correct_topK_answers"] = nCorrect_topK
            Model_dict["n_correct_and_confident_top1"] = nCorrect_and_confident_top1
            Model_dict["n_correct_and_confident_topK"] = nCorrect_and_confident_topK
            Model_dict["n_queried_images"] = nSamples
            Model_dict["n_confident_images"] = nConfident_top1
            Model_dict["confidence_threshold"] = perCLFperCLSthreshold[clf_id,:].tolist()
            Model_dict["accuracy_compromise"] = accuracy_compromise

            if modelname != modelnames_list[-1]:
                # Prepare for next layer. but if this was the last level - no need to prepare.
                indices_of_samples_that_must_run_in_next_level = \
                    np.where(probabilities.max(axis=1) < perCLFperCLSthreshold[clf_id,:].take(classes[:,0]))[0]

                if nSamples > 0:  # special case, where there was no samples left in the first place,
                    # no need to construct the index_mapping further
                    # ld("adding to comp list len " + `len(indices_mapping_composition_lst)`)
                    # ld(indices_of_samples_that_must_run_in_next_level)
                    indices_mapping_composition_lst += [indices_of_samples_that_must_run_in_next_level]
                    # ld("now list is len " + `len(indices_mapping_composition_lst)`)
                nUnsatisfied = len(indices_of_samples_that_must_run_in_next_level)
                nSatisfied = nSamples - nUnsatisfied
                ld("\t" + modelname + " - " + str(nUnsatisfied) + " unsatisfied samples found")
                assert (nSatisfied == nConfident_topK == nConfident_top1)

                # satisfied sample doesnt mean correct!!!
                # it only means that there is a class with a
                # confidence measure above the defined threshold
                nSamples_passed_to_next_level = str(nUnsatisfied)

            Results_dict[modelname] = Model_dict


        with open(os.path.join(output_dir, output_filename), "w") as f:
            json.dump(Results_dict, f)

    ###############################################################################
    ######## PHASE II.b - LOAD processed logs from json file#######################
    ###############################################################################
    with open(os.path.join(output_dir, output_filename), "r") as f:
        json_data = f.read()
        Results_dict = json.loads(json_data)
        nExaminedImages = Results_dict['nExaminedImages']

    ###############################################################################
    ###################### PHASE III - generate plots #############################
    ###############################################################################
    ylst_accmac_top_1 = []
    ylst_accmac_top_K = []
    ylst_acc_top_1 = []
    ylst_acc_top_K = []
    ylst_mac = []
    y_queried_images_list = [[] for _ in range(len(modelnames_list))]

    gflops_coeff = 1

    correct_and_confident_top1 = 0
    correct_and_confident_topK = 0
    confident_top1 = 0
    gflop = 0.0
    nTotal_Samples_Queried_using_this_epsilon = 0
    cummul_images_queried_for_this_epsilon = 0

    for netname, netname_dict, netid in zip(modelnames_list, dict_modelname_list, range(len(modelnames_list))):
        images_queried = Results_dict[netname]["n_queried_images"]
        correct_and_confident_top1 += Results_dict[netname]['n_correct_and_confident_top1']
        correct_and_confident_topK += Results_dict[netname]['n_correct_and_confident_topK']
        confident_top1 += Results_dict[netname]["n_confident_images"]
        confident_topK = confident_top1
        gflop += images_queried * model_dict[netname_dict][2]
        nTotal_Samples_Queried_using_this_epsilon += Results_dict[netname]["n_queried_images"]
        cummul_images_queried_for_this_epsilon += images_queried
        y_queried_images_list[netid].append(cummul_images_queried_for_this_epsilon)

    gflop_per_inference = gflop * gflops_coeff / float(nExaminedImages)

    # ld("For delta=" + `delta` + ", " + `nExaminedImages` + " images required " + `nTotal_Samples_Queried_using_this_epsilon` + " queries, the total GFLOPS=" + `gflop` + ". Per-inference-GFLOPS=" + `gflop_per_inference`)
    ylst_accmac_top_1 += [float(correct_and_confident_top1) / float(confident_top1) / float(gflop_per_inference)]
    ylst_accmac_top_K += [float(correct_and_confident_topK) / float(confident_topK) / float(gflop_per_inference)]
    ylst_acc_top_1 += [float(correct_and_confident_top1) / float(confident_top1)]
    ylst_acc_top_K += [float(correct_and_confident_topK) / float(confident_topK)]
    ylst_mac += [gflop_per_inference]

    accuracy_compromise_str = "Accuracy compromise " + r"$(\epsilon)$"
    if showBaseline:
        plotListOfScatters([[accuracy_compromise], epsilon_lst], [ylst_mac, len(epsilon_lst) * [baselineGFLOPsPerInference]],
                           ['GII', 'non-GII'], accuracy_compromise_str, "Giga-FLOP*" + str(gflops_coeff),
                           "Number of floating point operaions as a function of accuracy compromise",
                           output_dir, "FLOP_vs_threshold.jpg")
        # plotTwoLines(ylst_mac, len(ylst_mac) * [baselineGFLOPsPerInference], epsilon_lst,
        #              accuracy_compromise_str, "Giga-FLOP*" + `gflops_coeff`,
        #              "Number of floating point operaions as a function of accuracy compromise", output_dir,
        #              "FLOP_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
        plotListOfScatters([[accuracy_compromise], epsilon_lst], [ylst_accmac_top_1, len(epsilon_lst) * [baselineTop1Accuracy / baselineGFLOPsPerInference]],
                           ['GII', 'non-GII'], accuracy_compromise_str, "accuracy/GFLOP*" + str(gflops_coeff),
                           "Top 1 combined performance as a function of accuracy compromise",
                           output_dir, "TOP1_AccFlop_vs_threshold.jpg")
        # plotTwoLines(ylst_accmac_top_1, len(ylst_accmac_top_1) * [baselineTop1Accuracy / baselineGFLOPsPerInference],
        #              epsilon_lst, accuracy_compromise_str, "accuracy/GFLOP*" + `gflops_coeff`,
        #              "Top 1 combined performance as a function of accuracy compromise", output_dir,
        #              "TOP1_AccFlop_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
        plotListOfScatters([[accuracy_compromise], epsilon_lst], [ylst_acc_top_1, len(epsilon_lst) * [baselineTop1Accuracy / baselineGFLOPsPerInference]],
                           ['GII', 'non-GII'], accuracy_compromise_str, "Accuracy",
                           "Top 1 confident accuracy as a function of accuracy compromise",
                           output_dir, "TOP1_Acc_vs_threshold.jpg")
        # plotTwoLines(ylst_acc_top_1, len(ylst_acc_top_1) * [baselineTop1Accuracy], epsilon_lst,
        #              accuracy_compromise_str, "Accuracy",
        #              "Top 1 confident accuracy as a function of accuracy compromise", output_dir,
        #              "TOP1_Acc_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
    else:
        plotListOfScatters([[accuracy_compromise]], [ylst_mac],
                           ['GII'], accuracy_compromise_str, "Giga-FLOP*" + str(gflops_coeff),
                           "Number of floating point operaions as a function of accuracy compromise",
                           output_dir, "TOP1_Acc_vs_threshold.jpg")
        # plotOneLine(ylst_mac, epsilon_lst, accuracy_compromise_str, "Giga-FLOP*" + `gflops_coeff`,
        #             "Number of floating point operaions as a function of accuracy compromise", output_dir,
        #             "FLOP_vs_threshold.jpg")
        plotListOfScatters([[accuracy_compromise]], [ylst_accmac_top_1],
                           ['GII'], accuracy_compromise_str, "accuracy/GFLOP*" + str(gflops_coeff),
                           "Top 1 combined performance as a function of accuracy compromise",
                           output_dir, "TOP1_AccFlop_vs_threshold.jpg")
        # plotOneLine(ylst_accmac_top_1, epsilon_lst, accuracy_compromise_str, "Accuracy/GFLOP*" + `gflops_coeff`,
        #             "Top 1 combined performance as a function of accuracy compromise", output_dir,
        #             "TOP1_AccFlop_vs_threshold.jpg")
        plotListOfScatters([[accuracy_compromise]], [ylst_acc_top_1],
                           ['GII'], accuracy_compromise_str, "Accuracy",
                           "Top 1 confident accuracy as a function of accuracy compromise",
                           output_dir, "TOP1_Acc_vs_threshold.jpg")
        # plotOneLine(ylst_acc_top_1, epsilon_lst, accuracy_compromise_str, "Accuracy",
        #             "Top 1 confident accuracy as a function of accuracy compromise", output_dir,
        #             "TOP1_Acc_vs_threshold.jpg")

    plotBetweens(y_queried_images_list, [accuracy_compromise], accuracy_compromise_str, "Number of processed inputs",
                 "Breakdown of inputs processed at each classifier", classifier_legends, output_dir,
                 "TOP1_Processed_Inputs_Breakdown.jpg")
    plotOneLine(ylst_accmac_top_K, [accuracy_compromise], accuracy_compromise_str, "Accuracy/GFLOP*" + str(gflops_coeff),
                "Top " + str(topK) + " combined performance as a function of accuracy compromise", output_dir,
                "TOP" + str(topK) + "_AccFlop_vs_threshold.jpg")
    plotOneLine(ylst_acc_top_K, [accuracy_compromise], accuracy_compromise_str, "Accuracy",
                "Top " + str(topK) + " confident accuracy as a function of accuracy compromise", output_dir,
                "TOP" + str(topK) + "_Acc_vs_threshold.jpg")

    # Write additional statistics to a text file
    with open(os.path.join(output_dir, "Stats.txt"), "w") as f:
        f.write("--- Per classifier accuracies and thresholds: ---\n")
        for modelname, acc, i in zip(modelnames_list, perCLFaccuracy, range(nClassifiers)):
            f.write("{} {:4.2f}%, Delta({})={}\n".format(modelname, acc * 100, i , perCLFperCLSthreshold[i,:].tolist()))

        f.write("Compared GII against Baseline(Top1-accuracy of {:4.2f}%".format(baselineTop1Accuracy * 100))
        f.write(", baseline runtime of {:10.10f} GFLOP)\n".format(baselineGFLOPsPerInference))
        f.write("For epsilon={:3.2f} accuracy is {:4.2f}% is speedup is {:6.3f}\n".format(accuracy_compromise, ylst_acc_top_1[0] * 100, baselineGFLOPsPerInference/ylst_mac[0]))

    return ylst_mac, ylst_acc_top_1
