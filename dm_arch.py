import math
import numpy as np
import tensorflow as tf

import dm_utils

FLAGS = tf.app.flags.FLAGS

# Global switch to enable/disable training of variables
_glbl_is_training = tf.Variable(initial_value=True, trainable=False, name='glbl_is_training')

# Global variable dictionary. This is how we can share variables across models
_glbl_variables = {_glbl_is_training.name : _glbl_is_training}


def initialize_variables(sess):
    """Run this function only once and before the model begins to train"""
    sess.run(tf.global_variables_initializer())

def enable_training(onoff):
    """Switches training on or off globally (all models are affected).
    It is expected that dropout will be enabled during training and disabled afterwards. Batch normalization also affected.
    """
    tf.assign(_glbl_is_training, bool(onoff))


# TBD: Add "All you need is a good init"

class Model:
    """A neural network model.

    Currently only supports a feedforward architecture."""
    
    def __init__(self, name, features, enable_batch_norm=True):
        self.name    = name
        self.locals  = set()
        self.outputs = [features]
        
        self.enable_batch_norm = enable_batch_norm

    def _get_variable(self, name, initializer=None):
        # Variables are uniquely identified by a triplet: model name, layer number, and variable name
        layer = 'L%03d' % (self.get_num_layers()+1,)
        full_name = '/'.join([self.name, layer, name])

        if full_name in _glbl_variables:
            # Reuse existing variable
            #print("Reusing variable %s" % full_name)
            var = _glbl_variables[full_name]
            assert var.get_shape() == initializer.get_shape()
        elif initializer is not None:
            # Create new variable
            var = tf.Variable(initializer, name=full_name)
            _glbl_variables[full_name] = var
        else:
            raise ValueError("Initializer must be provided if variable is new") 

        self.locals.add(var)
        return var

    def _get_num_inputs(self):
        return int(self.get_output().get_shape()[-1])

    def _variable_initializer(self, prev_units, num_units, stddev_factor=1.0):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        assert prev_units > 0 and num_units > 0
        stddev  = np.sqrt(float(stddev_factor) / np.sqrt(prev_units*num_units))
        return tf.truncated_normal([prev_units, num_units],
                                    mean=0.0, stddev=stddev)

    def _variable_initializer_conv2d(self, prev_units, num_units, mapsize, is_residual):
        """Initialization in the style of Glorot 2010.

        stddev_factor should be 1.0 for linear activations, and 2.0 for ReLUs"""

        assert prev_units > 0 and num_units > 0
        size   = [mapsize, mapsize, prev_units, num_units]
        stddev_factor = 1e-1 / (mapsize * mapsize * prev_units * num_units)
        result = stddev_factor * np.random.uniform(low=-1, high=1, size=size)

        if not is_residual:
            # Focus nearly all the weight on the center
            for i in range(min(prev_units, num_units)):
                result[mapsize//2, mapsize//2, i, i] += 1.0
        # else leaving all parameters near zero is the right thing to do

        result = tf.constant(result.astype(np.float32))

        return result

    def get_num_layers(self):
        return len(self.outputs)

    def add_batch_norm(self, scale=False):
        """Adds a batch normalization layer to this model.

        See ArXiv 1502.03167v3 for details."""

        if not self.enable_batch_norm:
            return self

        out = tf.contrib.layers.batch_norm(self.get_output(), scale=scale, is_training=_glbl_is_training)
        
        self.outputs.append(out)
        return self

    def add_dropout(self, keep_prob=.5):
        """Applies dropout to output of this model"""

        is_training = tf.to_float(_glbl_is_training)
        keep_prob = is_training * keep_prob + (1.0 - is_training)
        out = tf.nn.dropout(self.get_output(), keep_prob=keep_prob)

        self.outputs.append(out)
        return self

    def add_flatten(self):
        """Transforms the output of this network to a 1D tensor"""

        batch_size = int(self.get_output().get_shape()[0])
        out = tf.reshape(self.get_output(), [batch_size, -1])

        self.outputs.append(out)
        return self

    def add_reshape(self, shape):
        """Reshapes the output of this network"""

        out = tf.reshape(self.get_output(), shape)

        self.outputs.append(out)
        return self        

    def add_dense(self, num_units, stddev_factor=1.0):
        """Adds a dense linear layer to this model.

        Uses Glorot 2010 initialization assuming linear activation."""
        
        assert len(self.get_output().get_shape()) == 2, "Previous layer must be 2-dimensional (batch, channels)"

        prev_units = self._get_num_inputs()
        
        # Weight term
        initw   = self._variable_initializer(prev_units, num_units,
                                           stddev_factor=stddev_factor)
        weight  = self._get_variable('weight', initw)

        # Bias term
        initb   = tf.constant(0.0, shape=[num_units])
        bias    = self._get_variable('bias', initb)

        # Output of this layer
        out     = tf.matmul(self.get_output(), weight) + bias

        self.outputs.append(out)
        return self

    def add_sigmoid(self, rnge=1.0):
        """Adds a sigmoid (0,1) activation function layer to this model."""

        prev_units = self._get_num_inputs()
        out = 0.5 + rnge * (tf.nn.sigmoid(self.get_output()) - 0.5)
        
        self.outputs.append(out)
        return self

    def add_tanh(self):
        """Adds a tanh (-1,+1) activation function layer to this model."""

        prev_units = self._get_num_inputs()
        out = tf.nn.tanh(self.get_output())
        
        self.outputs.append(out)
        return self

    def add_softmax(self):
        """Adds a softmax operation to this model"""

        this_input = tf.square(self.get_output())
        reduction_indices = list(range(1, len(this_input.get_shape())))
        acc = tf.reduce_sum(this_input, reduction_indices=reduction_indices, keep_dims=True)
        out = this_input / (acc+FLAGS.epsilon)
        #out = tf.verify_tensor_all_finite(out, "add_softmax failed; is sum equal to zero?")
        
        self.outputs.append(out)
        return self

    def add_relu(self):
        """Adds a ReLU activation function to this model"""

        out = tf.nn.relu(self.get_output())

        self.outputs.append(out)
        return self        

    def add_elu(self):
        """Adds a ELU activation function to this model"""

        out = tf.nn.elu(self.get_output())

        self.outputs.append(out)
        return self

    def add_lrelu(self, leak=.2):
        """Adds a leaky ReLU (LReLU) activation function to this model"""

        t1  = .5 * (1 + leak)
        t2  = .5 * (1 - leak)
        out = t1 * self.get_output() + \
              t2 * tf.abs(self.get_output())
            
        self.outputs.append(out)
        return self

    def add_conv2d(self, num_units, mapsize=1, stride=1, is_residual = False):
        """Adds a 2D convolutional layer."""

        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"
        
        prev_units = self._get_num_inputs()
        
        # Weight term and convolution
        initw  = self._variable_initializer_conv2d(prev_units, num_units, mapsize, is_residual=is_residual)
        weight = self._get_variable('weight', initw)
        out    = tf.nn.conv2d(self.get_output(), weight,
                              strides=[1, stride, stride, 1],
                              padding='SAME')

        # Bias term
        initb  = tf.constant(0.0, shape=[num_units])
        bias   = self._get_variable('bias', initb)
        out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_conv2d_transpose(self, num_units, mapsize=1, stride=1, is_residual = False):
        """Adds a transposed 2D convolutional layer"""

        assert not "This function is broken right now due to how _variable_initializer_conv2d is built. Use a regular convolution instead"
        
        assert len(self.get_output().get_shape()) == 4 and "Previous layer must be 4-dimensional (batch, width, height, channels)"

        prev_units = self._get_num_inputs()
        
        # Weight term and convolution
        initw  = self._variable_initializer_conv2d(prev_units, num_units, mapsize, is_residual=is_residual)
        weight = self._get_variable('weight', initw)
        weight = tf.transpose(weight, perm=[0, 1, 3, 2])
        prev_output = self.get_output()
        output_shape = [FLAGS.batch_size,
                        int(prev_output.get_shape()[1]) * stride,
                        int(prev_output.get_shape()[2]) * stride,
                        num_units]
        out    = tf.nn.conv2d_transpose(self.get_output(), weight,
                                        output_shape=output_shape,
                                        strides=[1, stride, stride, 1],
                                        padding='SAME')

        # Bias term
        initb  = tf.constant(0.0, shape=[num_units])
        bias   = self._get_variable('bias', initb)
        out    = tf.nn.bias_add(out, bias)
            
        self.outputs.append(out)
        return self

    def add_concat(self, terms):
        """Adds a concatenation layer"""

        if len(terms) > 0:
            axis = len(self.get_output().get_shape()) - 1
            terms = terms + [self.get_output()]
            out = tf.concat(axis, terms)
            self.outputs.append(out)
        
        return self

    def add_sum(self, term):
        """Adds a layer that sums the top layer with the given term"""

        prev_shape = self.get_output().get_shape()
        term_shape = term.get_shape()
        #print("%s %s" % (prev_shape, term_shape))
        assert prev_shape[1:] == term_shape[1:] and "Can't sum terms with a different size"
        out = tf.add(self.get_output(), term)
        
        self.outputs.append(out)
        return self

    def add_mean(self):
        """Adds a layer that averages the inputs from the previous layer"""

        prev_shape = self.get_output().get_shape()
        reduction_indices = list(range(len(prev_shape)))
        assert len(reduction_indices) > 2 and "Can't average a (batch, activation) tensor"
        reduction_indices = reduction_indices[1:-1]
        out = tf.reduce_mean(self.get_output(), reduction_indices=reduction_indices)
        
        self.outputs.append(out)
        return self

    def add_avg_pool(self, height=2, width=2):
        """Adds a layer that performs average pooling of the given size"""

        ksize   = [1, height, width, 1]
        strides = [1, height, width, 1]
        out     = tf.nn.avg_pool(self.get_output(), ksize, strides, 'VALID')
        
        self.outputs.append(out)
        return self

    def add_upscale(self, factor=2):
        """Adds a layer that upscales the output by 2x through nearest neighbor interpolation.
        See http://distill.pub/2016/deconv-checkerboard/"""

        out = dm_utils.upscale(self.get_output(), factor)

        self.outputs.append(out)
        return self        

    def get_output(self):
        """Returns the output from the topmost layer of the network"""
        return self.outputs[-1]

    def get_num_parameters(self):
        """Return the number of parameters in this model"""
        num_params = 0
        for var in self.locals:
            size = 1
            for dim in var.get_shape():
                size *= int(dim)
            num_params += size
        return num_params

    def get_all_variables(self):
        """Returns all variables used in this model"""
        return list(self.locals)
