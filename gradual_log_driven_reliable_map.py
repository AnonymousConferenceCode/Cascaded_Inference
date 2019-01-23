'''

Anonymized Source Code
'''
from utils import *
import pickle
from scipy.signal import savgol_filter
import argparse
logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
ld = logging.debug

###############################################################################
###################### PHASE I - Configuration ################################
###############################################################################
if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='Reads the softmax dumps and evaluates the accuracy '
                    'of the three classifiers in a cascade, as a function of confidnce'
                    'saves a figure containing bars and lines in the same drawing.')
    parser.add_argument('-dataset_path', type=str,
                        default='/data/datasets/Imagenet',
                        help='a path to an Imagenet directory, where a "tf_records" '
                             'directory must reside.')
    parser.add_argument('-softmax_path', type=str,
                        default='../TF_Models/official/resnet/output_softmax',
                        help='The softmax_dir, where clf0, clf1, clf2'
                             'are softmax dumps for train and test must be located.')
    parser.add_argument('-output_dir', type=str,
                        default='output_log_processing',
                        help='Where all the figures are about to be created')

    args = parser.parse_args()
    nClasses_list   = [10,100,10]
    B               = 200 #50 # Number of bins for quantization of confidence levels
    topK            = 5
    nClassifiers    = 3
    epsilon         = 0.01
    DO_IMAGENET     = True
    ANALYZE_TESTSET_INSTEAD_OF_TRAINSET = True

    #confidence_threshold_lst = map(lambda(x):x/100.0, range(0,101,5))
    # input_dir       = 'output_softmax'

    plot_out_fnames = ['_rk.png','_rk.eps']
    output_dir = 'output_log_processing'
    if DO_IMAGENET:
        input_dir = args.softmax_path #'../TF_Models/official/resnet/output_softmax'
        global_dataset_dir     = [args.dataset_path] #['/data/datasets/Imagenet']
        labels_filename = 'validation_non_onehot.npy'
        model_basename_list = ['resnet50']
    else:
        global_dataset_dir     = ['/data/datasets/CIFAR10',
                                    '/data/datasets/CIFAR100',
                                    '/data/datasets/SVHN']
        labels_filename = 'test_labels_non_onehot.npy'
        model_basename_list = ['resnetec_n_18', 'c100_resnetec_n_18', 'svhn_resnetec_n_18']


    global_modelnames_list = [["{}_clf{}".format(model_basename, clf_id) for clf_id in range(nClassifiers)] for model_basename in model_basename_list]

    #legend_list = [r"$M_{}$".format(clf_id) for clf_id in range(nClassifiers)]
    legend_list = [r"$M_0$", r"$M_0\rightarrow M_1$", r"$M_0\rightarrow M_1\rightarrow M_2$"]
    #inference_gflops_list = [15.3, 19.6]
    #reReadResult    = True # if true then the results from the input folder will be re-run prior to plots generation
    #                       # if false then the Result.json will be loaded for plot generation

    #checkpoints_dir = 'checkpoints'
    #models_dict_filename = 'Model_dict.json'
    #path_to_json = os.path.join(checkpoints_dir, models_dict_filename)
    #if os.path.exists(path_to_json):
    #    with open(path_to_json,"r") as f:
    #        json_data = f.read()
    #        model_dict = json.loads(json_data)
    #else:
    #    print("Error, no model dictionary found at " + path_to_json)

    ###############################################################################
    ####### PHASE II.a - parse logs and generate statistics in a dictionary########
    ###############################################################################

    #Results_dict = {}

    #if reReadResult:



    for dataset_dir, modelnames_list, nClasses, model_basename in zip(global_dataset_dir, global_modelnames_list, nClasses_list, model_basename_list):

        Probabilities_dict = {}
        Topclasses_dict = {}

        LABELS = readNPY(os.path.join(dataset_dir, labels_filename))
        #nExaminedImages = len(LABELS)
        nSamples = len(LABELS)
        #Results_dict['nExaminedImages'] = nExaminedImages

        for modelname in modelnames_list:
            te_tr_str = "" if ANALYZE_TESTSET_INSTEAD_OF_TRAINSET else "train"
            modelneame_brief = modelname.replace("resnet50_", "") if DO_IMAGENET else modelname
            with open(os.path.join(input_dir, modelneame_brief+te_tr_str+"_all_probabilities.npy"),"rb") as probFile:
                Probabilities_dict[modelname] = pickle.load(probFile)
            with open(os.path.join(input_dir, modelneame_brief+te_tr_str+"_top" + str(topK) + "_classes.npy"), "rb") as classFile:
                Topclasses_dict[modelname] = pickle.load(classFile)

        rk_y_axis_list = []
        alpha_rk_y_axis_list = []
        rk_x_axis_list = []
        sumS_gs_in_binS_list = []

        for modelname in modelnames_list:

            #ld("Beginning model " + modelname)

            #Model_dict = {}

            # Take the Top-1 predictions from the pre-loaded log files.
            # (these predictions are 2D array  with one row per sample
            # and 1000 classes sorted from the most probable to least probable

            classes = Topclasses_dict[modelname][:,0]

            # take all the probabilities, note that the probabilities are not sorted,
            # instead, they are arranged according to the classes. Namely probabilities[0,0]
            # will contain the probability outputted by the network for the 0th image
            # and the 0th class.
            probabilities = Probabilities_dict[modelname]


            #ld("\t" + modelname + " - running " + `nSamples` + " samples")
            assert(np.shape(classes)[0] == np.shape(probabilities)[0] == nSamples)

            # c is the boolean array of correct prediction indications
            #(for every prediction sample - 0 indicates incorrect)
            c = classes == LABELS
            #ld("Model " + modelname + " corrects:")
            #ld(c)
            # g is the array with the largest probability of every prediction sample
            g = np.max(probabilities, axis=1)
            #ld("Model " + modelname + " gs:")
            #ld(g)
            max_g = np.max(g)

            B_w = float(max_g) / float(B)
            rk_x_axis = np.zeros(B,dtype=np.float32)
            rk_y_axis = np.zeros(B,dtype=np.float32)
            alpha_rk_y_axis = np.zeros(B,dtype=np.float32)
            sumS_gs_in_binS = np.zeros(B,dtype=np.float32)

            #ld("Using B=" + `B` + " and B_w=" + `B_w`)
            for k in range(B):

                # This is a mask with "1" at indices that belong to the current bucket
                vector_of_gs_in_bin = np.logical_and(    k*B_w < g  ,   g <= (k+1)*B_w  )
                vector_of_correct_gs_in_bin = np.logical_and(   c   ,   vector_of_gs_in_bin  )
                sum_gs_in_bin = vector_of_gs_in_bin.sum()
                sum_correct_gs_in_bin = vector_of_correct_gs_in_bin.sum()

                # This is the Alpha based accuracy (namely for every delta, the alpha accuracy is the accuracy over all the
                alpha_vector_of_gs_in_bin = (k * B_w < g).astype(np.float32)
                alpha_vector_of_correct_gs_in_bin = np.logical_and(c, alpha_vector_of_gs_in_bin)
                alpha_sum_gs_in_bin = alpha_vector_of_gs_in_bin.sum()
                alpha_sum_correct_gs_in_bin = alpha_vector_of_correct_gs_in_bin.sum()

                rk_x_axis[k] = k*B_w
                rk_y_axis[k] = float(sum_correct_gs_in_bin) / float(sum_gs_in_bin) if sum_gs_in_bin > 0 and sum_correct_gs_in_bin>0 else epsilon
                alpha_rk_y_axis[k] = float(alpha_sum_correct_gs_in_bin) / float(alpha_sum_gs_in_bin) if alpha_sum_gs_in_bin > 0 and alpha_sum_correct_gs_in_bin>0 else epsilon
                sumS_gs_in_binS[k] = float(sum_gs_in_bin)



            # Normalize the incidence plot
            sumS_gs_in_binS = sumS_gs_in_binS/float(sum(sumS_gs_in_binS))

            # Append the plot-vectors into a list for further plotting (outside this loop)
            rk_y_axis_list.append(rk_y_axis)
            alpha_rk_y_axis_list.append(alpha_rk_y_axis)
            rk_x_axis_list.append(rk_x_axis)
            sumS_gs_in_binS_list.append(sumS_gs_in_binS)


        if not os.path.exists(os.path.join(output_dir, model_basename)):
            os.mkdir(os.path.join(output_dir, model_basename))

        plotListOfPlots(rk_x_axis_list,
                        rk_y_axis_list,
                        legend_list,
                        r"Confidence ($\delta$)",
                        "Reliable Accuracy",
                        "",
                        os.path.join(output_dir, model_basename),
                        'Reliable_Accuracy.png',
                        lpf=11)

        plotListOfPlots(rk_x_axis_list,
                        sumS_gs_in_binS_list,
                        legend_list,
                        r"Confidence ($\delta$)",
                        "Frequency",
                        "",
                        os.path.join(output_dir, model_basename),
                        'Confidence_Incidence.png',
                        lpf=None)

        legend_list_extended = [r"$\alpha_{}(\delta)$ of {}".format(iii, s) for s,iii in zip(legend_list,range(nClassifiers))] + ["Frequency($\delta$) of "+s for s in legend_list]
        x_axis_list_extended = rk_x_axis_list + rk_x_axis_list
        y_axis_extended = [savgol_filter(np.array(data), 11, 1) for data in alpha_rk_y_axis_list] + sumS_gs_in_binS_list
        plotListOfPlots(x_axis_list_extended,
                        y_axis_extended,
                        legend_list_extended,
                        r"Confidence ($\delta$)",
                        r"Frequency and $\alpha_m(\delta)$",
                        "",
                        os.path.join(output_dir, model_basename),
                        'Alpha_Accuracy_Confidence_Incidence.png',
                        lpf=None)

        plotListOfPlots_and_Bars(x_axis_list_extended,
                        y_axis_extended,
                        legend_list_extended,
                        r"Confidence ($\delta$)",
                        r"Bars: Frequency$(\delta)$ ; Lines: $\alpha_m(\delta)$",
                        "",
                        os.path.join(output_dir, model_basename),
                        'Alpha_Accuracy_Confidence_Incidence_Bar.png',
                        3,
                        lpf=None)