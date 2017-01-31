import math
import numpy as np
import scipy.misc
import tensorflow as tf

class Container(object):
    """Dumb container object"""
    def __init__(self, dictionary):
        self.__dict__.update(dictionary)

def _edge_filter():
    """Returns a 3x3 edge-detection functionally filter similar to Sobel"""

    # See https://en.wikipedia.org/w/index.php?title=Talk:Sobel_operator&oldid=737772121#Scharr_not_the_ultimate_solution
    a = .5*(1-math.sqrt(.5))
    b = math.sqrt(.5)

    # Horizontal filter as a 4-D tensor suitable for tf.nn.conv2d()
    h = np.zeros([3,3,3,3])

    for d in range(3):
        # I.e. each RGB channel is processed independently
        h[0,:,d,d] = [ a,  b,  a]
        h[2,:,d,d] = [-a, -b, -a]

    # Vertical filter
    v = np.transpose(h, axes=[1, 0, 2, 3])

    return h, v

def total_variation_loss(images, name='total_variation_loss'):
    """Returns a loss function that penalizes high-frequency features in the image.
    Similar to the 'total variation loss' but using a different high-pass filter."""

    filter_h, filter_v = _edge_filter()
    strides = [1,1,1,1]

    hor_edges = tf.nn.conv2d(images, filter_h, strides, padding='VALID', name='horizontal_edges')
    ver_edges = tf.nn.conv2d(images, filter_v, strides, padding='VALID', name='vertical_edges')

    l2_edges  = tf.add(hor_edges*hor_edges, ver_edges*ver_edges, name='L2_edges')

    total_variation_loss = tf.reduce_mean(l2_edges, name=name)

    return total_variation_loss

def distort_image(image):
    """Perform random distortions to the given 4D image and return result"""

    # Switch to 3D as that's what these operations require
    slices = tf.unpack(image)
    output = []

    # Perform pixel-wise distortions
    for image in slices:
        image  = tf.image.random_flip_left_right(image)
        image  = tf.image.random_saturation(image, .2, 2.)
        image += tf.truncated_normal(image.get_shape(), stddev=.05)
        image  = tf.image.random_contrast(image, .85, 1.15)
        image  = tf.image.random_brightness(image, .3)
        
        output.append(image)

    # Go back to 4D
    image   = tf.pack(output)
    
    return image

def downscale(images, K):
    """Differentiable image downscaling by a factor of K"""
    arr = np.zeros([K, K, 3, 3])
    arr[:,:,0,0] = 1.0/(K*K)
    arr[:,:,1,1] = 1.0/(K*K)
    arr[:,:,2,2] = 1.0/(K*K)
    dowscale_weight = tf.constant(arr, dtype=tf.float32)
    
    downscaled = tf.nn.conv2d(images, dowscale_weight,
                              strides=[1, K, K, 1],
                              padding='SAME')
    return downscaled

def upscale(images, K):
    """Differentiable image upscaling by a factor of K"""
    prev_shape = images.get_shape()
    size = [K * int(s) for s in prev_shape[1:3]]
    out  = tf.image.resize_nearest_neighbor(images, size)

    return out

def save_image(image, filename, verbose=True):
    """Saves a (height,width,3) numpy array into a file"""
    scipy.misc.toimage(image, cmin=0., cmax=1.).save(filename)
    print("    Saved %s" % (filename,))
