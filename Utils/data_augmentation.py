import tensorflow as tf
import numpy as np
from scipy import ndimage
from math import sqrt
import scipy.misc

# NOTE
# Images are assumed to be float32 mean-centered around 0 with ~0.5 std.
# For augment function:
#   images shape: (batch_size, height, width, channels=3)
#   labels shape: (batch_size, 3)

def add_blotch(image, max_dims=[0.4,0.4]):
    #add's small black/white box randomly in periphery of image
    new_image = np.copy(image)
    shape = new_image.shape
    max_x = shape[0] * max_dims[0]
    max_y = shape[1] * max_dims[1]
    rand_x = 0
    rand_y = np.random.randint(low=0, high=shape[1])
    rand_bool = np.random.randint(0,2)
    if rand_bool == 0:
        rand_x = np.random.randint(low=0, high=max_x)
    else:
        rand_x = np.random.randint(low=(shape[0]-max_x), high=shape[0])
    size = np.random.randint(low=1, high=7) #size of each side of box
    new_image[rand_x:(size+rand_x), rand_y:(size+rand_y), :] = np.random.uniform(-0.5,0.5)
    return new_image

def shift(image, max_amt=0.2):
    new_img = np.copy(image)
    shape = new_img.shape
    max_x = int(shape[0] * max_amt)
    max_y = int(shape[1] * max_amt)
    x = np.random.randint(low=-max_x, high=max_x)
    y = np.random.randint(low=-max_y, high=max_y)
    return ndimage.interpolation.shift(new_img,shift=[x,y,0])

def add_noise(image):
    image_min = abs(image.min())
    noise_mask = np.random.normal(image,0.4)
    noisy_img = image + (noise_mask)
    return noisy_img

def rotate(image):
    randnum = np.random.randint(1,360)
    new_image = np.copy(image)
    rotated_image = ndimage.rotate(new_image, angle=randnum, reshape=False)
    return rotated_image

def blur(image):
    return ndimage.filters.gaussian_filter(image, 0.2)

def medianf(image):
    return ndimage.median_filter(image, 3)

def flipRows(image):
    return np.flipud(image)

def flipCols(image):
    return np.fliplr(image)

def simple(image):
    '''
    Performs the following 3 stages: as described in the
    Resnet paper, CIFAR-10 experiments:


    1)  Flips the image horizontally with probability 1/2.
    2)  Pads the first two dimensions of a 3D image by 4
        pixels from each side.
    3)  Randomly crops the original sized image from the padded one
    :param image:
    :return: the image, of the original size
    '''

    nPadPixels = 4

    # (1)
    original_shape = image.shape
    if np.random.randint(low=0, high=2) == 1:
        image = flipCols(image)

    # (2)
    padded_image = np.pad(image,[(nPadPixels,nPadPixels),(nPadPixels,nPadPixels),(0,0)],'constant')

    # (3)
    crop_axis0_first_idx = np.random.randint(low=0, high=2*nPadPixels)
    crop_axis0_last_idx = crop_axis0_first_idx + original_shape[0]
    crop_axis1_first_idx = np.random.randint(low=0, high=2*nPadPixels)
    crop_axis1_last_idx = crop_axis1_first_idx + original_shape[1]
    return padded_image[crop_axis0_first_idx:crop_axis0_last_idx,crop_axis1_first_idx:crop_axis1_last_idx,:]

def otf_augment3d_image_batch(images, labels, use_simple=False):
    '''
    randomly manipulates images batch of 3D images
    where the color channel is the last dimension
    namely, the format of the images is NHWC
    The augmentation transformations are:
    rotate, flip along axis, add blotch, shift , identity (no change)
    and recently added "shear" and "zoom" augmentations.
    INPUT:
    images shape: (batch_size, height, width, channels=3)
    labels shape: (batch_size, whatever)
    '''
    if use_simple:
        ops = {
            0: simple,
        }
    else:
        ops = {
            0: rotate,
            1: shift,
            2: add_blotch,
            3: add_noise,
            4: lambda x : x, # to have ~9% of the images unchanged
            5: lambda x : tf.contrib.keras.preprocessing.image.random_shear(x, 0.2, row_axis=0, col_axis=1, channel_axis=2),
            6: lambda x: tf.contrib.keras.preprocessing.image.random_zoom(x, (0.8, 0.8), row_axis=0, col_axis=1, channel_axis=2),
            7: blur,
            8: medianf,
            9: flipRows,
            10: flipCols
            #8: lambda x: tf.contrib.keras.preprocessing.image.random_brightness(x, (0.8, 0.8))
        }

    num_transforms = len(ops)

    shape = images.shape
    num_classes = labels.shape[1]

    new_images = np.zeros((shape[0], shape[1], shape[2], shape[3]), dtype=np.float32)
    if labels is not None:
        new_labels = np.zeros(((shape[0]), num_classes))
    for i in range(images.shape[0]):
        cur_img = np.copy(images[i])
        new_images[i] = cur_img
        if labels is not None:
            new_labels[i] = np.copy(labels[i])
        which_op = np.random.randint(low=0, high=num_transforms)
        new_images[i] = ops[which_op](cur_img)
        if labels is not None:
            new_labels[i] = np.copy(labels[i])
        # if i <100 :
        #    scipy.misc.imsave("Bikoret_Img/bikoret_{}_transform_{}.jpg".format(i,which_op), new_images[i, :, :, :])
        #    scipy.misc.imsave("Bikoret_Img/bikoret_{}_transform_{}_Orig.jpg".format(i, which_op), images[i, :, :, :])



    if labels is not None:
        return new_images, new_labels.astype(np.uint8)
    else:
        return new_images


def otf_augment1d_image_batch(images, labels, return_3d_batch=False, use_simple=False):
    '''
    randomly manipulates images batch of 1D images
    i.e. the each image contains flattened 1D array
    of pixels where the first third of the pixels
    is a row-major Red channel. The second third of
    the pixels are the row-major Green-channel and the
    last third of the pixels is the Blue-channel.
    The augmentation transformations are:
    rotate, flip along axis, add blotch, shift.

    Assuming RECTANGULAR IMAGES !!!

    :param use_simple - applies only basic augmentations - padding 4 pixels from each side and applying random crop
                          of the size of the original image

    INPUT:
    images shape: (batch_size, flattened pixels of R,G,B channels)
    labels shape: (batch_size, whatever)

    if return_3d_batch=True, the 1d images batch is augmented and returned as a
    batch of 3d images.
    '''
    shape1d = images.shape
    shape3d = (shape1d[0],int(sqrt(shape1d[1]/3)),int(sqrt(shape1d[1]/3)),3)
    images_3d = images.reshape(shape3d)
    augmented_images_3d, augmented_labels = otf_augment3d_image_batch(images_3d, labels, use_simple=use_simple)
    if return_3d_batch:
        return augmented_images_3d, augmented_labels
    else:
        augmented_images_1d = augmented_images_3d.flatten().reshape(shape1d)
        return augmented_images_1d, augmented_labels

def otf_augment(images, labels=None, use_simple=False):
    '''
    Determines the format of the input batch
    and applies the appropriate agmentation
    function.

    :param use_simple - applies only basic augmentations - padding 4 pixels from each side and applying random crop
                          of the size of the original image
    '''
    shape = images.shape
    if len(shape)==2:
        ret = otf_augment1d_image_batch(images, labels, use_simple=use_simple)
    elif len(shape)==4:
        ret = otf_augment3d_image_batch(images, labels, use_simple=use_simple)
    else:
        raise Exception('"Error in augment() - the provided image batch neither contains 1D nor 3D images."')

    return ret