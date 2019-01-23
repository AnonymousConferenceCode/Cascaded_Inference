import argparse
import numpy as np
import os
import tensorflow as tf
from Utils.IO import writeNPY, readNPY
from os import listdir
from os.path import isfile, join
import pickle
import pdb



def get_train_test_accuracy_of_classifier(classifier_id, softmax_dir,
                                          groundtruth_train_label_file_path=None,
                                          groundtruth_test_label_file_path=None):
    '''
    Reads the output classes of classifier_id, located in the softmax_dir.
    Compares the outputs against the groundtruth train and test labels and
    returns the top-1 and top-5 accuracy metrics of the train and test sets.
    :param classifier_id:
    :param softmax_dir:
    :param groundtruth_train_label_file_path:
    :param groundtruth_test_label_file_path:
    :return: train-top1 accuracy, train top-5 accuracy, test-top1 accuracy, test-top5 accuracy
    '''
    train_acc_top1, train_acc_top5, test_acc_top1, testacc_top5 = 0,0,0,0

    if groundtruth_train_label_file_path is not None:
        with open(os.path.join(softmax_dir, "clf{}_train_top5_classes.npy".format(classifier_id)), "rb") as openfile:
            top_5_output_train_classes = pickle.load(openfile)
        with open(groundtruth_train_label_file_path, "rb") as openfile:
            groundtruth_train_classes = pickle.load(openfile)
        train_acc_top1 = (top_5_output_train_classes[:, 0] == groundtruth_train_classes).sum().astype(np.float32) / \
                         float(np.shape(top_5_output_train_classes)[0])
        train_groundtruth_replicated = np.tile(groundtruth_train_classes[np.newaxis].T, 5)
        train_acc_top5 = (top_5_output_train_classes==train_groundtruth_replicated).sum().astype(np.float32) / \
                         float(np.shape(top_5_output_train_classes)[0])
    if groundtruth_test_label_file_path is not None:
        with open(os.path.join(softmax_dir, "clf{}_top5_classes.npy".format(classifier_id)), "rb") as openfile:
            top_5_output_test_classes = pickle.load(openfile)
        with open(groundtruth_test_label_file_path, "rb") as openfile:
            groundtruth_test_classes = pickle.load(openfile)
        test_acc_top1  = (top_5_output_test_classes[:, 0]==groundtruth_test_classes).sum().astype(np.float32) / \
                         float(np.shape(top_5_output_test_classes)[0])
        test_groundtruth_replicated = np.tile(groundtruth_test_classes[np.newaxis].T, 5)

        testacc_top5 = (top_5_output_test_classes == test_groundtruth_replicated).sum().astype(np.float32) / \
                       float(np.shape(top_5_output_test_classes)[0])

    return [train_acc_top1, train_acc_top5, test_acc_top1, testacc_top5]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Analyzes the train/test accuracies of the three classifiers by comparing '
                    'their reported softmax dumps against train_labels_non_onehot.npy and '
                    'validation_labels_non_onehot.npy in the imagenet directory.')
    parser.add_argument('-dataset_path', type=str,
                        default='/data/datasets/Imagenet',
                        help='a path to an Imagenet directory, where a "tf_records" directory must reside.')
    parser.add_argument('-softmax_dir_to_validate_against', type=str,
                        default='../TF_Models/official/resnet/output_softmax',
                        help='If set, the existing npy files are compared against the softmax_dir, where clf0, clf1, clf2'
                             'are softmax dumps for train and test must be located. The output of the comparison is '
                             'presented in terms of accuracy. This operation should be applied in order to verify'
                             'the corect order of the inputs fed into labels and the into the neural net which '
                             'produced the softmax dumps.')

    args = parser.parse_args()

    train_label_filepath = os.path.join(args.dataset_path, "train_non_onehot.npy")
    test_label_filepath = os.path.join(args.dataset_path, "validation_non_onehot.npy")

    for classifier_id in range(3):
        r = get_train_test_accuracy_of_classifier(classifier_id=classifier_id,
                                              softmax_dir=args.softmax_dir_to_validate_against,
                                              groundtruth_train_label_file_path=train_label_filepath,
                                              groundtruth_test_label_file_path=test_label_filepath)
        print("Classifier {}:".format(classifier_id))
        print(" Train top-1:{:.4}".format(r[0]))
        print(" Train top-5:{:.4}".format(r[1]))
        print(" Test top-1:{:.4}".format(r[2]))
        print(" Test top-5:{:.4}".format(r[3]))


