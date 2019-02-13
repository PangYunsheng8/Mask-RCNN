"""
    Here is some utils function for read annotation, generate mask .etc
"""

import os
import tensorflow as tf 


def read_file_from_directory(directory, postfix=".xml"):
    """
    Args:
        :param directory: the target directory
        :param postfix: the postfix of file to be find
        :return: a generator which returns the absolute path to the file which meets the postfix
    """
    assert os.path.isdir(directory), "{} no such file or directory".format(directory)
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(postfix):
                yield os.path.join(root, file)


def get_corresponding_image(image_dir, filename):
    """
    Args:
        :param image_dir: the root directory of image
        :param filename: type str , the filename of image
    :return:
        the absolute path name to the image
    """
    image = os.path.join(image_dir, filename)
    if os.path.exists(image):
        return image
    else:
        raise NotADirectoryError("{} no such file or directory found".format(image))


def crop_resize_addAlpha(input_image, boxes, masks, crop_size=(224,224)):
    """
        Crop patches from original image and add an alpha channel to the 
        patches according to the masks.
    Args:
        input_image: uint8 tensor with shape (batch_size, height, width, 3)
        boxes : a tensor of shape (batch_size, numbox, 4),
        (ymin, xmin, ymax, xmax) the coordinates are in normalized format between [0, 1]
        masks : a uint8 tensor of shape (batch_size, numbox, height, width) with value between either 0 or 1.
        crop_size : the size of the patches 
    """
    # check the valid of the data 
    masks = tf.squeeze(masks, [0])
    masks = tf.expand_dims(masks, axis=3)
    masks = tf.image.resize_bilinear(masks, size=[224, 224])
    detection_masks = tf.cast(tf.greater(masks, 0.5), tf.float32)
    ind = tf.reshape(tf.range(tf.shape(boxes)[0]), [-1, 1])
    box_ind = ind * tf.ones(tf.shape(boxes)[:2], dtype=tf.int32)

    patches = tf.image.crop_and_resize(input_image,
                                    boxes=tf.reshape(boxes, [-1, 4]),
                                    box_ind=tf.reshape(box_ind, [-1]),
                                    crop_size=[224, 224])
    output = add_mask(patches, detection_masks)
    return output


def add_mask(im, mask):
    """ 
        im tensor (num_box, h, w, c)
        mask tensor (num_box, h, w, 1)
    """
    mask *= 255
    return tf.concat((im, mask), axis=-1)


