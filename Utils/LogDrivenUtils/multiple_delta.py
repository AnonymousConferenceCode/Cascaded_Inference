'''
This function reads the results of the gradual networks
script runs for different networks. Next, it
simulates a hierarchical inference over the models
specified in a modelnames_list list below.

It tries a hierarchical inference using different
delta values, and for every delta measures how many
classifiers were required in order to reach the delta
confidence. The ground-truth labels are loaded from the

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

logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

def aggregate_hierarchical_model_performance(nDeltas,
                                             nClassifiers,
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
    Reads the separate test-results from files:
        <input_dir>/<model_name>_clf*_top5_classes.npy
        <input_dir>/<model_name>_clf*_all_probabilities.npy

    for every sub-classifier (* stands for classifier id in the hierarchical model)
    Then the function calculates the required number of MAC loperations and accuracy
    as a function of various confidence_thresholds (deltas).

    Generates plots&stats.txt file and places them into the directory
        <output_dir>/<model_name>/*

    :param nDeltas: can be one of the following:
                    (i) int - number of different confidence-thresholds checked for the plots
                              in this case the scanned deltas will be range(0,nDeltas)
                    (ii) [int1,int2, int3] - the lower and the upper limits of the deltas. in this case
                              the range will be from int1/(int3) to int2/(int3)
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

    ############################################################################### #  ###### ### ######  #
    ###################### PHASE I - Configuration ################################ ##  ##### ### #####  ##
    ############################################################################### ###  #### ### ####  ###

    # nClasses            = 10
    topK = 5
    if type(nDeltas)==list:
        confidence_th_lst = map(lambda x: x / float(nDeltas[2]),
                                range(nDeltas[0], (nDeltas[1] + 1), 1))  # A.K.A. deltas to be plotted as the X-axis
    else:
        confidence_th_lst = map(lambda x: x / float(nDeltas),
                                range(0, (nDeltas + 1), 1))  # A.K.A. deltas to be plotted as the X-axis
    # input_dir           = 'output_softmax'            # Path where the softmax values per test example are stored (input of the script)
    # output_dir          = 'output_log_processing'       # Path where the products of this script are written to
    output_filename = 'Results.json'  # Path to the output file with all the numerical result of this script
    # dataset_dir         = '/data/datasets'              # Path to the main directory of the datasets
    # dataset_name        = 'SVHN'                     # Name of the dataset (also the nameof the sub-directory in the datasets dir)
    labels_filename = 'test_labels_non_onehot.npy'  # Name of the labels file, containing a 1-D numpy array with the ground
    # nClassifiers       = 3
    # modelnames_list     = ['svhn_backtrack_doubling_tiny_clf0', 'svhn_backtrack_doubling_tiny_clf1', 'svhn_backtrack_doubling_tiny_clf2']
    modelnames_list = [primary_modelname + "_clf" + str(i) for i in range(nClassifiers)]
    # dict_modelname_list = ['backtrack_doubling_tiny_clf0', 'backtrack_doubling_tiny_clf1', 'backtrack_doubling_tiny_clf2']# The names of the corresponding models from the model_dict.json
    dict_modelname_list = [dict_modelname + "_clf" + str(i) for i in range(nClassifiers)]
    classifier_legends = [r"$M_{}$".format(i) for i in range(nClassifiers)]
    reReadResult = not useCachedResults  # if true then the results from the input folder will be re-run prior to plots generation
    # if false then the Result.json will be loaded for plot generation

    # For backtrack-doubling network baseline printout:
    showBaseline = True  # if true then two plots will be shown  - one of them is the baseline performance of the non-GII version
    # baselineGFLOPsPerInference = 0.003820710
    getbaselineTop1AccuracyFromLogs = True
    baselineTop1Accuracy = 1.0  # ignored if previous option is True. Edit this line only if you want to implant the baseline accuracy
    if getbaselineTop1AccuracyFromLogs:
        perCLFaccuracy = []
        perCLFthreshold = {}
        for modelname in modelnames_list:
            with open(os.path.join(log_dir, "log_" + modelname + ".txt"), "r") as f:
                lineList = f.readlines()
                precentage_str = lineList[-1].split(" ")[-1]  # This gets you the "87.7%" string
                perCLFaccuracy.append(float(precentage_str.split("%")[0]) / 100.0)
        baselineTop1Accuracy = perCLFaccuracy[-1]

    # Create a private directory for the model being analyzed
    output_dir = os.path.join(output_dir, primary_modelname)
    if (not os.path.exists(output_dir)):
        print("Creating temporary directory for log driven analysis products: " + output_dir)
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
        print("Error, no model dictionary found at " + path_to_json)

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

        for modelname in modelnames_list:
            with open(os.path.join(input_dir, modelname + "_all_probabilities.npy"), "rb") as probFile:
                Probabilities_dict[modelname] = pickle.load(probFile)
            with open(os.path.join(input_dir, modelname + "_top" + str(topK) + "_classes.npy"), "rb") as classFile:
                Topclasses_dict[modelname] = pickle.load(classFile)

        for confidence_threshold in confidence_th_lst:

            Delta_dict = {}
            LABELS = np.copy(LABELS_ORIGINAL)
            logging.debug("Using confidence threshold {0:.3f}.".format(confidence_threshold))
            indices_mapping_composition_lst = [range(len(
                LABELS))]  # this is cool. it stores the indices mapping from the initially run samples' indices to the most recently surviived ones

            for modelname in modelnames_list:

                # ld("Beginning model " + modelname)

                Model_dict = {}

                # Take the predictions from the pre-loaded log files.
                if (not (modelname in Topclasses_dict)) or (not (modelname in Probabilities_dict)):
                    ld("Error, the model named " + modelname + " was not found in the output files)")
                    os.exit(-1)
                classes = Topclasses_dict[modelname]
                probabilities = Probabilities_dict[modelname]

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
                    confident_indicators = [np.ones(nSamples, dtype=np.bool) for i in range(
                        topK)]  # stupid tautology in order to effectively reduce the threshold to zero!
                else:
                    confident_indicators = [(np.sort(probabilities)[:, -i - 1] >= confidence_threshold) for i in range(
                        topK)]  # each line will contain TRUE indicators where a confidence was reached
                correct_and_confident_indicators = [np.logical_and(confident_indicators[i], correct_indicators[i]) for i
                                                    in range(topK)]

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

                if modelname != modelnames_list[-1]:
                    # Prepare for next layer. but if this was the last level - no need to prepare.
                    indices_of_samples_that_must_run_in_next_level = \
                    np.where(probabilities.max(axis=1) < confidence_threshold)[0]

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

                Delta_dict[modelname] = Model_dict

            Results_dict[confidence_threshold] = Delta_dict

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

    for delta in confidence_th_lst:
        correct_and_confident_top1 = 0
        correct_and_confident_topK = 0
        confident_top1 = 0
        gflop = 0.0
        nTotal_Samples_Queried_using_this_delta = 0
        cummul_images_queried_for_this_delta = 0

        for netname, netname_dict, netid in zip(modelnames_list, dict_modelname_list, range(len(modelnames_list))):
            images_queried = Results_dict[str(delta)][netname]["n_queried_images"]
            correct_and_confident_top1 += Results_dict[str(delta)][netname]['n_correct_and_confident_top1']
            correct_and_confident_topK += Results_dict[str(delta)][netname]['n_correct_and_confident_topK']
            confident_top1 += Results_dict[str(delta)][netname]["n_confident_images"]
            confident_topK = confident_top1
            gflop += images_queried * model_dict[netname_dict][2]
            nTotal_Samples_Queried_using_this_delta += Results_dict[str(delta)][netname]["n_queried_images"]
            cummul_images_queried_for_this_delta += images_queried
            y_queried_images_list[netid].append(cummul_images_queried_for_this_delta)

        gflop_per_inference = gflop * gflops_coeff / float(nExaminedImages)

        # ld("For delta=" + `delta` + ", " + `nExaminedImages` + " images required " + `nTotal_Samples_Queried_using_this_delta` + " queries, the total GFLOPS=" + `gflop` + ". Per-inference-GFLOPS=" + `gflop_per_inference`)
        ylst_accmac_top_1 += [float(correct_and_confident_top1) / float(confident_top1) / float(gflop_per_inference)]
        ylst_accmac_top_K += [float(correct_and_confident_topK) / float(confident_topK) / float(gflop_per_inference)]
        ylst_acc_top_1 += [float(correct_and_confident_top1) / float(confident_top1)]
        ylst_acc_top_K += [float(correct_and_confident_topK) / float(confident_topK)]
        ylst_mac += [gflop_per_inference]

    confidence_threshold_str = "Confidence threshold " + r"$(\delta)$"
    if showBaseline:
        plotTwoLines(ylst_mac, len(ylst_mac) * [baselineGFLOPsPerInference], confidence_th_lst,
                     confidence_threshold_str, "Giga-FLOP*" + str(gflops_coeff),
                     "Number of floating point operaions as a function of confidence threshold", output_dir,
                     "FLOP_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
        plotTwoLines(ylst_accmac_top_1, len(ylst_accmac_top_1) * [baselineTop1Accuracy / baselineGFLOPsPerInference],
                     confidence_th_lst, confidence_threshold_str, "accuracy/GFLOP*" + str(gflops_coeff),
                     "Top 1 combined performance as a function of confidence threshold", output_dir,
                     "TOP1_AccFlop_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
        plotTwoLines(ylst_acc_top_1, len(ylst_acc_top_1) * [baselineTop1Accuracy], confidence_th_lst,
                     confidence_threshold_str, "accuracy",
                     "Top 1 confident accuracy as a function of confidence threshold", output_dir,
                     "TOP1_Acc_vs_threshold.jpg", isAnnotated=False, trainStr='GII', validStr='non-GII')
    else:
        plotOneLine(ylst_mac, confidence_th_lst, confidence_threshold_str, "Giga-FLOP*" + str(gflops_coeff),
                    "Number of floating point operaions as a function of confidence threshold", output_dir,
                    "FLOP_vs_threshold.jpg")
        plotOneLine(ylst_accmac_top_1, confidence_th_lst, confidence_threshold_str, "Accuracy/GFLOP*" + str(gflops_coeff),
                    "Top 1 combined performance as a function of confidence threshold", output_dir,
                    "TOP1_AccFlop_vs_threshold.jpg")
        plotOneLine(ylst_acc_top_1, confidence_th_lst, confidence_threshold_str, "Accuracy",
                    "Top 1 confident accuracy as a function of confidence threshold", output_dir,
                    "TOP1_Acc_vs_threshold.jpg")

    plotBetweens(y_queried_images_list, confidence_th_lst, confidence_threshold_str, "Number of processed inputs",
                 "Breakdown of inputs processed at each classifier", classifier_legends, output_dir,
                 "TOP1_Processed_Inputs_Breakdown.jpg")
    plotOneLine(ylst_accmac_top_K, confidence_th_lst, confidence_threshold_str, "Accuracy/GFLOP*" + str(gflops_coeff),
                "Top " + str(topK) + " combined performance as a function of confidence threshold", output_dir,
                "TOP" + str(topK) + "_AccFlop_vs_threshold.jpg")
    plotOneLine(ylst_acc_top_K, confidence_th_lst, confidence_threshold_str, "Accuracy",
                "Top " + str(topK) + " confident accuracy as a function of confidence threshold", output_dir,
                "TOP" + str(topK) + "_Acc_vs_threshold.jpg")

    # Write additional statistics to a text file
    with open(os.path.join(output_dir, "Stats.txt"), "w") as f:
        if getbaselineTop1AccuracyFromLogs:
            f.write("--- Per classifier accuracies: ---\n")
            for modelname, acc in zip(modelnames_list, perCLFaccuracy):
                f.write(modelname + " {:4.2f}%\n".format(acc * 100))

        f.write("-------- Other Statistics --------\n")
        f.write("Comparing GII against Baseline(Top1-accuracy of {:4.2f}%".format(baselineTop1Accuracy * 100))
        f.write(", baseline runtime of {:10.10f} GFLOP)\n".format(baselineGFLOPsPerInference))

        GIIaccs = np.array(ylst_acc_top_1)
        GIIdeltas = np.array(confidence_th_lst)
        GIImacs = np.array(ylst_mac)
        GII_best_acc = GIIaccs.max() * 100
        GII_best_delta = GIIdeltas[GIIaccs.argmax()]
        GII_best_mac = GIImacs[GIIaccs.argmax()]
        speedup = baselineGFLOPsPerInference / GII_best_mac
        finalString = "GII best  accuracy is {:4.2f}% at delta={:4.3f}".format(GII_best_acc, GII_best_delta)
        finalString += " speedup is {:6.3f}\n".format(speedup)

        # Filter the accuracies and stay only with accuracies which are at leas as good as the baseline
        for i in [8, 4, 2, 1, 0]:
            percent = i / 100.0
            inds = np.where(GIIaccs >= baselineTop1Accuracy - percent)[0]
            ministring = "(eq) " if i == 0 else "(-" + str(i) + "%)"
            if len(inds > 0):
                GIIaccs = GIIaccs[inds]
                GIIdeltas = GIIdeltas[inds]
                GIImacs = GIImacs[inds]
                GII_best_acc = GIIaccs[GIImacs.argmin()] * 100
                GII_best_delta = GIIdeltas[GIImacs.argmin()]
                GII_best_mac = GIImacs.min()
                speedup = baselineGFLOPsPerInference / GII_best_mac
                f.write("GII_" + ministring + " accuracy")
                f.write(" is {:4.2f}% at delta={:4.3f}".format(GII_best_acc, GII_best_delta))
                f.write(" speedup is {:6.3f} \n".format(speedup))
            else:
                f.write("GII_" + ministring + " accuracy was not achieved by the GII model\n")

        f.write(finalString)

    return ylst_mac, ylst_acc_top_1
