import numpy as np
import os.path
import random
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS

# For convenience, here are the available attributes in the dataset:
# 5_o_Clock_Shadow Arched_Eyebrows Attractive Bags_Under_Eyes Bald Bangs Big_Lips \
# Big_Nose Black_Hair Blond_Hair Blurry Brown_Hair Bushy_Eyebrows Chubby Double_Chin \
# Eyeglasses Goatee Gray_Hair Heavy_Makeup High_Cheekbones Male Mouth_Slightly_Open Mustache \
# Narrow_Eyes No_Beard Oval_Face Pale_Skin Pointy_Nose Receding_Hairline Rosy_Cheeks Sideburns \
# Smiling Straight_Hair Wavy_Hair Wearing_Earrings Wearing_Hat Wearing_Lipstick Wearing_Necklace
# Wearing_Necktie Young

def _read_attributes(attrfile):
    """Parses attributes file from Celeb-A dataset and returns"""

    # The first line is the number of images in the dataset. Ignore.
    f = open(attrfile, 'r')
    f.readline()

    # The second line contains the names of the boolean attributes
    names = f.readline().strip().split()

    attr_names = {}
    for i in range(len(names)):
        attr_names[names[i]] = i

    # The remaining lines contain file name and a list of boolean attributes
    attr_values = []
    for _, line in enumerate(f):
        fields = line.strip().split()
        img_name = fields[0]
        assert img_name[-4:] == '.jpg'
        attr_bitfield = [field == '1' for field in fields[1:]]
        attr_bitfield = np.array(attr_bitfield, dtype=np.bool)
        attr_values.append((img_name, attr_bitfield))
        
    return attr_names, attr_values


def _filter_attributes(attr_names, attr_values, sel):
    """Returns the filenames that match the attributes given by 'dic'"""

    # Then select those files whose attributes all match the selection
    filenames = []
    for filename, attrs in attr_values:
        all_match = True
        for name, value in sel.items():
            column = attr_names[name]
            #print("name=%s, value=%s, column=%s, attrs[column]=%s" % (name, value, column, attrs[column]))
            if attrs[column] != value:
                all_match = False
                break

        if all_match:
            filenames.append(filename)

    return filenames


def select_samples(selection={}):
    """Selects those images in the Celeb-A dataset whose
    attributes match the constraints given in 'sel'"""

    attrfile = os.path.join(FLAGS.dataset, FLAGS.attribute_file)
    names, attributes = _read_attributes(attrfile)

    filenames = _filter_attributes(names, attributes, selection)

    filenames = sorted(filenames)
    random.shuffle(filenames)

    filenames = [os.path.join(FLAGS.dataset, file) for file in filenames]

    return filenames

