import numpy as np
import tensorflow as tf

import dm_utils

FLAGS = tf.app.flags.FLAGS


def inference(infer_data):

    sess = infer_data.sess
    idm  = infer_data.infer_model

    image = sess.run(idm.gene_out)
    image = np.squeeze(image, axis=0)

    dm_utils.save_image(image, FLAGS.outfile)
