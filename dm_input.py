import tensorflow as tf

import dm_celeba

FLAGS = tf.app.flags.FLAGS

def input_data(sess, mode, filenames, capacity_factor=3):

    # Separate training and test sets
    # TBD: Use partition given by dataset creators
    assert mode == 'inference' or len(filenames) >= FLAGS.test_vectors
    
    if mode == 'train':
        filenames  = filenames[FLAGS.test_vectors:]
        batch_size = FLAGS.batch_size
    elif mode == 'test':
        filenames  = filenames[:FLAGS.test_vectors]
        batch_size = FLAGS.batch_size
    elif mode == 'inference':
        filenames  = filenames[:]
        batch_size = 1
    else:
        raise ValueError('Unknown mode `%s`' % (mode,))
    
    # Read each JPEG file
    reader = tf.WholeFileReader()
    filename_queue = tf.train.string_input_producer(filenames)
    key, value = reader.read(filename_queue)
    channels = 3
    image = tf.image.decode_jpeg(value, channels=channels, name="dataset_image")
    image.set_shape([None, None, channels])

    # Crop and other random augmentations
    if mode == 'train':
        image = tf.image.random_flip_left_right(image)
        #image = tf.image.random_saturation(image, .95, 1.05)
        #image = tf.image.random_brightness(image, .05)
        #image = tf.image.random_contrast(image, .95, 1.05)

    size_x, size_y = 80, 100

    if mode == 'inference':
        # TBD: What does the 'align_corners' parameter do? Stretch blit?
        image = tf.image.resize_images(image, (size_y, size_x), method=tf.image.ResizeMethod.AREA)
    else:
        # Dataset samples are 178x218 pixels
        # Select face only without hair
        off_x, off_y   = 49, 90
        image = tf.image.crop_to_bounding_box(image, off_y, off_x, size_y, size_x)
    
    feature = tf.cast(image, tf.float32)/255.0

    # Using asynchronous queues
    features = tf.train.batch([feature],
                              batch_size=batch_size,
                              num_threads=4,
                              capacity = capacity_factor*batch_size,
                              name='features')

    tf.train.start_queue_runners(sess=sess)
      
    return features
