
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

def define_flags():
    # Configuration (alphabetically)
    tf.app.flags.DEFINE_integer('annealing_half_life', 1000,
                                "Number of batches until annealing temperature is halved")

    tf.app.flags.DEFINE_string('attribute_file', 'list_attr_celeba.txt',
                               "Celeb-A dataset attribute file")

    tf.app.flags.DEFINE_integer('batch_size', 16,
                                "Number of samples per batch.")

    tf.app.flags.DEFINE_string('checkpoint_dir', 'checkpoint',
                               "Output folder where checkpoints are dumped.")

    tf.app.flags.DEFINE_string('dataset', 'dataset',
                               "Path to the dataset directory.")

    tf.app.flags.DEFINE_float('epsilon', 1e-8,
                              "Fuzz term to avoid numerical instability")

    tf.app.flags.DEFINE_string('infile', None,
                               "Inference input file. See also `outfile`")

    tf.app.flags.DEFINE_float('instance_noise', 0.5,
                              "Standard deviation (amplitude) of instance noise")

    tf.app.flags.DEFINE_float('learning_rate_start', 0.000200,
                              "Starting learning rate used for AdamOptimizer")

    tf.app.flags.DEFINE_float('learning_rate_end',   0.000001,
                              "Ending learning rate used for AdamOptimizer")

    tf.app.flags.DEFINE_string('outfile', 'inference_out.png',
                               "Inference output file. See also `infile`")

    tf.app.flags.DEFINE_float('pixel_loss_max', 0.95,
                              "Initial pixel loss relative weight")

    tf.app.flags.DEFINE_float('pixel_loss_min', 0.70,
                              "Asymptotic pixel loss relative weight")

    tf.app.flags.DEFINE_string('run', None,
                                "Which operation to run. [train|inference]")

    tf.app.flags.DEFINE_integer('summary_period', 20,
                                "Number of batches between summary data dumps")

    tf.app.flags.DEFINE_integer('random_seed', 10,
                                "Seed used to initialize rng.")

    tf.app.flags.DEFINE_integer('test_vectors', 16,
                                """Number of features to use for testing""")
                                
    tf.app.flags.DEFINE_string('train_dir', 'train',
                               "Output folder where training logs are dumped.")

    tf.app.flags.DEFINE_string('train_mode', 'mtf',
                               "Training mode. Can be male-to-female (`mtf`), female-to-male (`ftm`), male-to-male (`mtm`) or female-to-female (`ftf`)")

    tf.app.flags.DEFINE_integer('train_time', 180,
                                "Time in minutes to train the model")
