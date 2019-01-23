# import cPickle as pickle
# import os
import numpy as np

def writeNPY(data,file_path):
    '''
    Serializes the "data" numpy array into an NPY file

    :param data: a numpy array to be written to a pickle file
    :param file_path: a full path including a file name from writing
    :return: none
    '''
    with open(file_path, 'wb') as f_out:
        np.save(f_out, data)

def readNPY(file_path):
    '''
    Loads NPY file
    :param file_path: a full path to an NPY file
    :return: the loaded numpy array.
    '''
    with open(file_path, 'rb') as f_in:
        data = np.load(f_in)

    return data

# def writePickle(data,file_path):
#     '''
#     Serializes the "data" numpy array into a pickle file
#     Handles huge file sizes by chunking them into
#     2^31 bytes and then writing these chunks serially.
#
#     :param data: a numpy array to be written to a pickle file
#     :param file_path: a full path including a file name from writing
#     :return: none
#     '''
#     n_bytes = 2 ** 31
#     max_bytes = 2 ** 31 - 1
#     bytes_out = pickle.dumps(data)
#     with open(file_path, 'wb') as f_out:
#         for idx in range(0, n_bytes, max_bytes):
#             f_out.write(bytes_out[idx:idx+max_bytes])
#
# def readPickle(file_path):
#     '''
#     Loads even huge pickle files.
#     :param file_path: a full path to a pickle file
#     :return: the loaded numpy array.
#     '''
#     max_bytes = 2 ** 31 - 1
#
#     input_size = os.path.getsize(file_path)
#     with open(file_path, 'rb') as f_in:
#         for _ in range(0, input_size, max_bytes):
#
#             try:
#                 bytes_in += f_in.read(max_bytes)
#             except NameError:
#                 bytes_in = f_in.read(max_bytes)
#
#     return pickle.loads(bytes_in)