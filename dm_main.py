import os

# Disable Tensorflow's INFO and WARNING messages
# See http://stackoverflow.com/questions/35911252
if 'TF_CPP_MIN_LOG_LEVEL' not in os.environ:
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import numpy as np
import numpy.random
import os.path
import random
import tensorflow as tf

import dm_celeba
import dm_flags
import dm_infer
import dm_input
import dm_model
import dm_show
import dm_train
import dm_utils

FLAGS = tf.app.flags.FLAGS


def _setup_tensorflow():
    # Create session
    config = tf.ConfigProto(log_device_placement=False) #, intra_op_parallelism_threads=1)
    sess   = tf.Session(config=config)

    # Initialize all RNGs with a deterministic seed
    with sess.graph.as_default():
        tf.set_random_seed(FLAGS.random_seed)
    
    random.seed(FLAGS.random_seed)
    np.random.seed(FLAGS.random_seed)

    return sess


# TBD: Move to dm_train.py?
def _prepare_train_dirs():
    # Create checkpoint dir (do not delete anything)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
        tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    
    # Cleanup train dir
    if tf.gfile.Exists(FLAGS.train_dir):
        try:
            tf.gfile.DeleteRecursively(FLAGS.train_dir)
        except:
            pass
    tf.gfile.MakeDirs(FLAGS.train_dir)

    # Ensure dataset folder exists
    if not tf.gfile.Exists(FLAGS.dataset) or \
       not tf.gfile.IsDirectory(FLAGS.dataset):
        raise FileNotFoundError("Could not find folder `%s'" % (FLAGS.dataset,))


# TBD: Move to dm_train.py?
def _get_train_data():
    # Setup global tensorflow state
    sess = _setup_tensorflow()

    # Prepare directories
    _prepare_train_dirs()

    # Which type of transformation?
    # Note: eyeglasses and sunglasses are filtered out because they tend to produce artifacts
    if FLAGS.train_mode == 'ftm' or FLAGS.train_mode == 'f2m':
        # Trans filter: from female to attractive male
        # Note: removed facial hair from target images because otherwise the network becomes overly focused on rendering facial hair
        source_filter = {'Male':False, 'Blurry':False, 'Eyeglasses':False}
        target_filter = {'Male':True,  'Blurry':False, 'Eyeglasses':False, 'Attractive':True, 'Goatee':False, 'Mustache':False, 'No_Beard':True}
    elif FLAGS.train_mode == 'mtf' or FLAGS.train_mode == 'm2f':
        # Trans filter: from male to attractuve female
        source_filter = {'Male':True,  'Blurry':False, 'Eyeglasses':False}
        target_filter = {'Male':False, 'Blurry':False, 'Eyeglasses':False, 'Attractive':True}
    elif FLAGS.train_mode == 'ftf' or FLAGS.train_mode == 'f2f':
        # Vanity filter: from female to attractive female
        source_filter = {'Male':False, 'Blurry':False, 'Eyeglasses':False}
        target_filter = {'Male':False, 'Blurry':False, 'Eyeglasses':False, 'Attractive':True}
    elif FLAGS.train_mode == "mtm" or FLAGS.train_mode == 'm2m':
        # Vanity filter: from male to attractive male
        source_filter = {'Male':True,  'Blurry':False, 'Eyeglasses':False}
        target_filter = {'Male':True,  'Blurry':False, 'Eyeglasses':False, 'Attractive':True}
    else:
        raise ValueError('`train_mode` must be one of: `ftm`, `mtf`, `ftf` or `mtm`')

    # Setup async input queues
    selected      = dm_celeba.select_samples(source_filter)
    source_images = dm_input.input_data(sess, 'train', selected)
    test_images   = dm_input.input_data(sess, 'test', selected)
    print('%8d source images selected' % (len(selected),))

    selected      = dm_celeba.select_samples(target_filter)
    target_images = dm_input.input_data(sess, 'train', selected)
    print('%8d target images selected' % (len(selected),))
    print()

    # Annealing temperature: starts at 1.0 and decreases exponentially over time
    annealing = tf.Variable(initial_value=1.0, trainable=False, name='annealing')
    halve_annealing = tf.assign(annealing, 0.5*annealing)

    # Create and initialize training and testing models
    train_model  = dm_model.create_model(sess, source_images, target_images, annealing, verbose=True)

    print("Building testing model...")
    test_model   = dm_model.create_model(sess, test_images, None, annealing)
    print("Done.")
    
    # Forget this line and TF will deadlock at the beginning of training
    tf.train.start_queue_runners(sess=sess)

    # Pack all for convenience
    train_data = dm_utils.Container(locals())

    return train_data


# TBD: Move to dm_infer.py?
def _get_inference_data():
    # Setup global tensorflow state
    sess = _setup_tensorflow()

    # Load single image to use for inference
    if FLAGS.infile is None:
        raise ValueError('Must specify inference input file through `--infile <filename>` command line argument')
                         
    if not tf.gfile.Exists(FLAGS.infile) or tf.gfile.IsDirectory(FLAGS.infile):
        raise FileNotFoundError('File `%s` does not exist or is a directory' % (FLAGS.infile,))
    
    filenames    = [FLAGS.infile]
    infer_images = dm_input.input_data(sess, 'inference', filenames)

    print('Loading model...')
    # Create inference model
    infer_model  = dm_model.create_model(sess, infer_images)

    # Load model parameters from checkpoint
    checkpoint = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
    try:
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint.model_checkpoint_path)
        del saver
        del checkpoint
    except:
        raise RuntimeError('Unable to read checkpoint from `%s`' % (FLAGS.checkpoint_dir,))
    print('Done.')

    # Pack all for convenience
    infer_data = dm_utils.Container(locals())

    return infer_data


def main(argv=None):
    if FLAGS.run == 'train':
        train_data = _get_train_data()
        dm_train.train_model(train_data)
    elif FLAGS.run == 'inference':
        infer_data = _get_inference_data()
        dm_infer.inference(infer_data)
    else:
        print("Operation `%s` not supported" % (FLAGS.run,))

if __name__ == '__main__':
    dm_flags.define_flags()
    tf.app.run()
