'''
This script reads the softmax outputs of the imagenet
trained cascaded resnet models. The cascade model
consists of three classifiers clf0, clf1, clf2. The
softmax outputs of ech classifier are composed into 3 chunks.
and are of the shape num_samples*num_classes. There are
softmax outputs for the validation set, which consist of
only 50k samples and therefore fit into a single chunk.

The script finds deltas for a wide variety of epsilons,
and then stores these deltas into the Log files of the
models that will be called log_<model_name>_epsilonXX.txt
The fact that these log files are created allows to run
the "gradual_log_driven.py" to produce useful plots of
acc-mac.

#######################################################
###                                                 ###
###     Be An Accurate Researcher !                 ###
###                                                 ###
###     Before running this script on the results,  ###
###     make sure that you checked *ALL* of the     ###
###     "PHASE I - Configuration" section           ###
###                                                 ###
#######################################################

Anonymized Source Code
'''
import logging
import os
import argparse
import numpy as np
import pickle
from analyze_imagenet_labels import get_train_test_accuracy_of_classifier
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

def get_topK_indicators(topk, ground_truth_classes, predicted_classes):
    """
    Returns an array of integer indicators where the predicted classes
    :param topk: 1 for top-1 , 5 for top-5
    :param ground_truth_classes: 1D array of ground truth labels
    :param predicted_classes: possibly 2D array of (samples, predicted classes)
    :return: 1D array of integer indicators I where I[i]=1 means that the sample i
             was correctly predicted among the topk predictions. I[i]=0 otherwise.
    """
    assert(ground_truth_classes.shape[0] == predicted_classes.shape[0])

    if topk==1:
        assert len(predicted_classes.shape) == 1
        return (ground_truth_classes == predicted_classes).astype(int)
    else:
        assert len(predicted_classes.shape) == 2
        assert predicted_classes.shape[1] == topk
        labels_replicated = np.tile(ground_truth_classes[np.newaxis].T, topk)
        indicators_topk = (predicted_classes == labels_replicated).astype(np.float32).sum(axis=1)
        assert(indicators_topk.min() >= 0 and indicators_topk.max() <= 1)
        return indicators_topk

def find_cst_deltas(epsilon, tr_softmaxes, tr_labels, deltas_for_top5=False):
    # type: (float, list, np.ndarray, bool) -> list
    """
    For a given epsilon, compute the classifier specific
    thresholds (CST).
    :param epsilon:
    :param tr_softmaxes:
    :param tr_labels:
    :param deltas_for_top5: boolean that if set to True, makes the whole
                            way of finding deltas suitable for top5 accuracy,
                            if set to False, then the way of setting the deltas
                            corresponds to a regular top-1 accuracy.
    :return: a list of floats that correspond to the
                classifier specific thresholds.
    """
    num_classifiers = len(tr_softmaxes)
    cst_deltas = []
    for classifier_id in range(num_classifiers):

        num_top_predictions = 5 if deltas_for_top5 else 1

        # Step 1 - acquire the indices order for sorting the predictions according to an increasing confidence
        train_preds = tr_softmaxes[classifier_id]
        num_train_inputs = train_preds.shape[0]
        train_classes = (-train_preds).argsort()
        train_classes_top1 = train_classes[:, 0]
        train_classes_top5 = train_classes[:, :5]
        train_preds_top = np.sort(np.partition(train_preds, -num_top_predictions, axis=1)[:, -num_top_predictions:], axis=1).sum(axis=1) # this gets the sum of the top K confidence levels
        sort_indices = train_preds_top.argsort()


        # Step 2 - sort the predictions in an increasing confidence order
        deltas_sorted = train_preds_top[sort_indices]
        predicted_classes_sorted = train_classes_top5[sort_indices,:] if deltas_for_top5 else train_classes_top1[sort_indices]
        label_classes_sorted = tr_labels[sort_indices]

        # Step 3 - find cumulitive accuracies over the training set as a function of different thresholds
        #          a residual accuracy of a threshold delta for classifier i is denoted \alpha_i(delta)
        #          and it determines the accuracy of the classifier i predictions over all the training
        #          samples that achieved confidence >= delta.
        correctly_predicted_indicators_sorted = get_topK_indicators(num_top_predictions,label_classes_sorted,predicted_classes_sorted)  # In the threshold setting section of the article, this array is denoted \c_i (\delta)
        residual_num_correct_predictions = correctly_predicted_indicators_sorted[::-1].cumsum()[::-1]
        residual_num_images = np.arange(1, (num_train_inputs + 1), 1).astype(float)[::-1]
        residual_accuracy = np.divide(residual_num_correct_predictions,
                                      residual_num_images)  # In the BT-CSTT algorithm, this array is denoted \alpha_i (\delta)

        # Step 4 - given (epsilon) - the desired number of percents to compromise from the top performance
        #          find the minimal threshold delta^i that satisfies alpha_i(delta^i) >= maximal_accuracy_of_clf_i - epsilon
        max_residual_accuracy = residual_accuracy.max()
        residual_accuracy_is_good_indicators = np.greater_equal(residual_accuracy,
                                                                max_residual_accuracy - epsilon).astype(int)  # True everywhere that the confidence level is above max-eplsilon)
        residual_accuracy_non_zero_only_if_good = np.multiply(residual_accuracy, residual_accuracy_is_good_indicators)
        best_delta_id = residual_accuracy_non_zero_only_if_good.nonzero()[0][0]  # Find fist non-zero element --> it will correspond to the smallest delta which satisfies the max-epsilon accuracy over the training set
        best_delta = deltas_sorted[best_delta_id]
        cst_deltas.append(best_delta)
    return cst_deltas


def evaluate_cascade(cst_deltas, te_softmaxes, te_labels):
    """

    :param cst_deltas:
    :param te_softmaxes:
    :param te_labels:
    :return:
    """
    pass



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
    parser.add_argument('-epsilon_low', type=float,
                        default=0,
                        help='The first epsilon (integer number of percents) to be'
                             'scanned for deltas')
    parser.add_argument('-epsilon_high', type=float,
                        default=20,
                        help='The last epsilon (integer number of percents) to be'
                             'scanned for deltas')
    parser.add_argument('-num_epsilons', type=int,
                        default=21,
                        help='number of epsilons to sample within the (low,high) range.')
    parser.add_argument('-output_dir', type=str,
                        default="Logs",
                        help='The log directory where all the log files will be created '
                             '(each will contain the confidence thresholds - deltas - '
                             'that were found)')

    args = parser.parse_args()

    num_train_samples = 1281167
    num_test_samples = 50000
    num_classes = 1001
    num_classifiers = 3
    classifier_id_list = range(num_classifiers)
    epsilon_list = list(np.linspace(args.epsilon_low,
                                    args.epsilon_high,
                                    args.num_epsilons)) #list(np.linspace(0,0.20,41))
    deltas_for_top5 = args.top5
    dataset_path = args.dataset_path
    softmax_path = args.softmax_path

    # 1) Read all the softmax outputs (train,test) into
    #    memory (takes many GBs) and the ground truth labels
    #    of the training and the test sets.
    #
    #
    classifier_test_accuracies = []
    classifier_test_accuracies_top5 = []
    tr_clf0_softmax = np.zeros((num_train_samples, num_classes), dtype=np.float32)
    tr_clf1_softmax = tr_clf0_softmax.copy()
    tr_clf2_softmax = tr_clf0_softmax.copy()
    tr_softmaxes = [tr_clf0_softmax, tr_clf1_softmax, tr_clf2_softmax]


    te_clf0_softmax = np.zeros((num_test_samples, num_classes), dtype=np.float32)
    te_clf1_softmax = te_clf0_softmax.copy()
    te_clf2_softmax = te_clf0_softmax.copy()
    te_softmaxes = [te_clf0_softmax, te_clf1_softmax, te_clf2_softmax]


    # load ground truth labels
    with open(os.path.join(dataset_path, "train_non_onehot.npy"), "rb") as openfile:
        tr_labels = pickle.load(openfile)
    with open(os.path.join(dataset_path, "validation_non_onehot.npy"), "rb") as openfile:
        te_labels = pickle.load(openfile)

    for classifier_id in classifier_id_list:
        # load test softmax results
        with open(os.path.join(softmax_path, "clf{}_all_probabilities.npy".format(classifier_id)), "rb") as openfile:
            te_softmaxes[classifier_id] = pickle.load(openfile)
        # load train softmax results
        with open(os.path.join(softmax_path, "clf{}_train_1of3_all_probabilities.npy".format(classifier_id)), "rb") as openfile:
            tr_softmaxes[classifier_id][:400000, :] = pickle.load(openfile)
        with open(os.path.join(softmax_path, "clf{}_train_2of3_all_probabilities.npy".format(classifier_id)), "rb") as openfile:
            tr_softmaxes[classifier_id][400000:800000, :] = pickle.load(openfile)
        with open(os.path.join(softmax_path, "clf{}_train_3of3_all_probabilities.npy".format(classifier_id)), "rb") as openfile:
            tr_softmaxes[classifier_id][800000:, :] = pickle.load(openfile)

        [_, _, clf_test_accuracy_top1, clf_test_accuracy_top5] = get_train_test_accuracy_of_classifier(classifier_id, softmax_path, None,
                                                                             os.path.join(dataset_path, "validation_non_onehot.npy"))
        classifier_test_accuracies.append(clf_test_accuracy_top1)
        classifier_test_accuracies_top5.append(clf_test_accuracy_top5)

    ld("This run will find deltas (confidence thresholds) for the following list of epsilons:")
    ld(epsilon_list)
    for epsilon in epsilon_list:
        ld('Finding deltas for epsilon {} under "top-{}" accuracy metric'.format(epsilon, 5 if deltas_for_top5 else 1))
        cst_deltas = find_cst_deltas(epsilon, tr_softmaxes, tr_labels, deltas_for_top5)
        ld("--> found deltas {}".format(cst_deltas))
        #print to log file#
        # Classifier_accuracy_compromise:0.20
        # Classifier_confidence_threshold:0.0
        # Classifier 2 Test accuracy: 70.52%

        for classifier_id, best_delta, test_accuracy, test_accuracy_top5 in zip(range(num_classifiers), cst_deltas, classifier_test_accuracies, classifier_test_accuracies_top5):
            int_epsilon = int(epsilon*10000)
            logFile = open(os.path.join("Logs", "log_resnet50_epsilon{:04d}_clf{}.txt".format(int_epsilon, classifier_id)), "w")
            cstStr = "Classifier_accuracy_compromise:{:3.5f}\n".format(epsilon)
            logFile.write(cstStr)
            cstStr = "Classifier_confidence_threshold:{}\n".format(best_delta)
            logFile.write(cstStr)
            testAccStr = "Classifier {} Test accuracy-top5: {:2.2f}% accuracy-top1: {:2.2f}%".format(classifier_id, test_accuracy_top5*100, test_accuracy*100)
            #ld(testAccStr)
            logFile.write(testAccStr)
            logFile.close()
