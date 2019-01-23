import os
import pickle
from Utils.IO import *
import logging
import sys
import numpy as np
import cPickle
from six.moves import urllib
import tarfile
import shutil

def onehot(targets, n_labels):
    '''
    receives the targets (1D array of decimals in [0...n_labels -1]
    returns 2D matrix, where each row i is one hot encoding of targets[i]
    '''
    ohm = np.zeros((targets.shape[0], n_labels), dtype=np.float32)
    #empty one-hot matrix
    ohm[np.arange(targets.shape[0]), targets] = 1
    #set target idx to 1

    return ohm

def saveMatrixToImage(w, shape, outputDir, filename):
    '''
    Receives the 1-D image w, that represents red,green,blue channels flattened
    and concatenated. This function reshapes the image to the "shape" specified 
    and stores it to the "outputDir" directory, under the "filename" specified.
    Expects: the shape to be in format CHW for example (3,32,32) in CIFAR10.
    '''
    low=np.min(w)
    w = w - low
    w = w.dot(255/np.max(w))
    import scipy.misc as sm
    sm.toimage(np.reshape(w,shape)).save(os.path.join(outputDir,filename))

def hwc2chw(w):
    '''
    converts the channel ordering
    '''
    shape = w.shape
    ret = np.zeros((shape[2],shape[0],shape[1]), dtype=np.float32)
    for c in range(shape[2]):
        ret[c,:,:] = w[:,:,c]
    return ret

def batch_1d_to_batch_nwhc(batch_1d, width, height, num_channels):
    '''
    Transforms a batch of 1D samples into a batch of 3D samples

    :param batch_1d: a 2D array with every row for a 1D flat sample
    :param width: desired width of a single sample
    :param height:
    :param num_channels:

    :return: a numpy ndarray with a shape (batch_1d.shape[0], width, height, num_channels)

    '''
    shape = batch_1d.shape

    temp_train_x = np.zeros((shape[0], width, height, num_channels), dtype=np.float32)
    for c in range(num_channels):
        oneD = batch_1d[:,c*width*height:(c+1)*width*height]
        temp_train_x[:,:,:,c] = oneD.reshape((shape[0],width,height))
    return temp_train_x

def maybe_download_and_extract(datafolder, DATA_URL):
  """Download and extract the tarball from Alex's website."""
  
  if not os.path.exists(datafolder):
    os.makedirs(datafolder)
    print("Created temporary directory \"" + datafolder + "\"")
  filename = DATA_URL.split('/')[-1]
  filepath = os.path.join(datafolder, filename)
  if not os.path.exists(filepath):
    def _progress(count, block_size, total_size):
      sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename,
          float(count * block_size) / float(total_size) * 100.0))
      sys.stdout.flush()
    filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')

  tarfile.open(filepath, 'r:gz').extractall(datafolder)

def unpickle(file):
    
    fo = open(file, 'rb')
    dict = cPickle.load(fo)
    fo.close()
    return dict

def getCIFAR(datafolder, dataset_name="CIFAR10", use_onehot=True):
    '''
    :param datafolder - contains the path to the specific directory of the dataset. e.g. it ends with
                        "CIFAR10"

    Description: This function reads the CIFAR10/100 from the datafolder, and if the data is not there
                 then it will be automaticaly downloaded and extracted. The function reads the dataset
                 into numpy arrays, and chops it into train-test, converts images to 3D and returns 5
                 neat numpy arrays as follows:

    Returns:
        train_set_X_t - float32 numpy array of a size 50000*32*32*3 training samples
        train_set_y_t - float32 numpy array of a size 50000*n_classes    training labels
        test_set_X    - float32 numpy array of a size 10000*32*32*3 test samples
        test_set_y    - float32 numpy array of a size 10000*n_classes    test labels
        test_set_y_non_onehot - uint32 1-D numpy array of a size nTest, containing the numbers of the classes for each test-sample

    Note: n_classes will be the last dimension of the 3 labels only if the use_onehot is True.
          Otherwise, the last dimension of the labels sets will be 1,and it will simply specify
          the number of the class [0,n_classes-1] of the corresponding sample.
    '''

    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    nUnused = 0
    nTrain = 50000
    nValid = 0
    nTest = 10000

    assert(nUnused + nTrain + nValid + nTest == 60000)


    datafolder = os.path.join(datafolder,"Raw")
    image_width = 32
    image_height = 32
    num_channels = 3
    test_filename_identification_str = "test"


    if dataset_name == "CIFAR10":
        whiteneddatafolder = "/data/konsta9/pylearn2_data/cifar10/pylearn2_gcn_whitened"
        datasubfolder = "cifar-10-batches-py"
        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        datafilesLst = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5", "test_batch"]
        label_key = 'labels'
        train_filename_identification_str = "data_batch_1"
        n_classes = 10
    elif dataset_name == "CIFAR100":
        datasubfolder = "cifar-100-python"
        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
        datafilesLst = ["test", "train"]
        label_key = 'fine_labels'
        train_filename_identification_str = "train"
        n_classes = 100
    else:
        ld("getCIFAR() error: Invalid dataset name provided: {}".format(dataset_name))



    maybe_download_and_extract(datafolder, DATA_URL)
    for filename in datafilesLst:
        #print("reading " + filename + "...")
        try:
            fullpath = os.path.join(datafolder,datasubfolder,filename)
            d = unpickle(fullpath)
        except:
            ld("Failed reading " +fullpath+". Exiting...")
            sys.exit()
        if test_filename_identification_str in filename:
            test_set_X = d['data']
            test_set_y = d[label_key]
        elif train_filename_identification_str in filename:
            train_set_X = d['data']
            train_set_y = d[label_key]
        else:
            train_set_X = np.concatenate((train_set_X, d['data']), axis=0)
            train_set_y = np.concatenate((train_set_y, d[label_key]), axis=0)


    # Make the labels be a numpy integer point array
    train_set_y = np.array(train_set_y)
    test_set_y = np.array(test_set_y)

    # To this point the arrays are test_set_X, test_set_y, train_set_X, train_set_y
    # And they contain only the data that was requested (unused samples were dropped)

    # Convert the images to real number matrices in range [0,1]
    train_set_X = train_set_X.astype(np.float32) / np.max(train_set_X)
    test_set_X = test_set_X.astype(np.float32) / np.max(test_set_X)

    zcaStr = "" #"(not-whitened, but normalized in each channel)"
    ld("Downloaded " + dataset_name + zcaStr)

    # Onehot the targets
    test_set_y_non_onehot = np.copy(test_set_y)
    if (use_onehot):
        train_set_y = onehot(train_set_y, n_classes)    
        #valid_set_y = onehot(valid_set_y, n_classes)
        test_set_y = onehot(test_set_y, n_classes)

    assert(np.shape(train_set_X)[0]==nTrain and np.shape(test_set_X)[0]==nTest)
    assert(np.shape(train_set_y)[0]==nTrain and np.shape(test_set_y)[0]==nTest)

    ld("Containing " + str(nTrain) + " training and " + str(nTest) + " test images")
    ld("Train set has mean " + str(np.mean(train_set_X)) + " and std " + str(np.std(train_set_X)))
    #if (nValid>0):
    #    print("Validation set has mean " + `np.mean(valid_set_X)` + " and std " + `np.std(valid_set_X)`)
    ld("Test set has mean " + str(np.mean(test_set_X)) + " and std " + str(np.std(test_set_X)))

    shutil.rmtree(datafolder)
    print("Removed unnecessary temporary directory \"" + datafolder + "\"")


    # Reshape the samples into NWHC format
    train_set_X = batch_1d_to_batch_nwhc(train_set_X, image_width, image_height, num_channels)
    #valid_set_X = batch_1d_to_batch_nwhc(valid_set_X, image_width, image_height, num_channels)
    test_set_X = batch_1d_to_batch_nwhc(test_set_X, image_width, image_height, num_channels)
    
    #return train_set_X, train_set_y, valid_set_X, valid_set_y, test_set_X, test_set_y, test_set_y_non_onehot
    return train_set_X, train_set_y, test_set_X, test_set_y, test_set_y_non_onehot



def maybeDownloadCIFAR(dataset_name, datasets_dir):
    '''
    This function:
    1) Downloads the cifar dataset if required,
    2) Transforms the 1D images into 3D RGB images
    2) Normalizes the features along each feature
       (the Red pixels at place (i,j) is normalized
       against all other Red pixels (i,j) in all other
       input images.

    The function writes the preprocessed dataset to
    the following files:
       datasets_dir/CIFAR**/train_X.npy
       datasets_dir/CIFAR**/train_y.npy
       datasets_dir/CIFAR**/test_X.npy
       datasets_dir/CIFAR**/test_y.npy
       datasets_dir/CIFAR**/test_labels_non_onehot.npy

        (** can be either 10 or 100)

    The labels are stored in one-hot encoding
    and an additional non-onehot test-labels copy is stored
    to test_labels_non_onehot.npy
    '''
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] - %(message)s')
    ld = logging.debug

    n_classes = 10 if dataset_name == "CIFAR10" else 100

    #datasetDir = os.path.join(datasets_dir, dataset_name)

    ld("Gathering the original " + dataset_name + " dataset...")
    train_X, train_y, test_X, test_y, test_y_non_onehot = getCIFAR(datasets_dir,dataset_name)

    ld("Storing the train samples and labels to: " + datasets_dir)
    trainXpath = os.path.join(datasets_dir, "train_X.npy")
    writeNPY(train_X, trainXpath)
    ld(" Train-set samples of the " + str(train_X.shape) + " shape were stored to " + trainXpath)
    trainypath = os.path.join(datasets_dir, "train_y.npy")
    writeNPY(train_y, trainypath)
    ld(" Train-set labels of the " + str(train_y.shape) + " shape were stored to " + trainypath)

    ld("Storing test (samples,labels) in the (3d,one-hot) formats to:" + datasets_dir)
    testXpath = os.path.join(datasets_dir, "test_X.npy")
    writeNPY(test_X, testXpath)
    ld(" Test-set samples of the " + str(test_X.shape) + " shape were stored to " + testXpath)
    testypath = os.path.join(datasets_dir, "test_y.npy")
    writeNPY(test_y, testypath)
    ld(" Test-set labels of the " + str(test_y.shape) + " shape were stored to " + testXpath)

    non_onehot_path = os.path.join(datasets_dir, "test_labels_non_onehot.npy")
    ld("Storing the non-onehot test-labels in the (1d) format to:" + non_onehot_path)
    writeNPY(test_y_non_onehot, non_onehot_path)
    ld(" Test-set labels of the following shape were stored to " + non_onehot_path)
    ld(" " + str(test_y_non_onehot.shape))

def maybeDownloadCIFAR10(datasets_dir):
    maybeDownloadCIFAR("CIFAR10", datasets_dir)

def maybeDownloadCIFAR100(datasets_dir):
    maybeDownloadCIFAR("CIFAR100", datasets_dir)
