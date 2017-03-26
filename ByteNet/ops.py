import numpy as np
import tensorflow as tf

def time_to_batch(value, dilation):
	# FOR DILATED CONVOLUTION, code adapted from tensorflow-wavenet
	with tf.name_scope('time_to_batch'):
		shape = value.get_shape()
		shape = [int(s) for s in shape]
		pad_elements = dilation - 1 - (int(shape[1]) + dilation - 1) % dilation
		padded = tf.pad(value, [[0, 0], [0, pad_elements], [0, 0]])
		reshaped = tf.reshape(padded, [-1, dilation, shape[2]])
		transposed = tf.transpose(reshaped, perm=[1, 0, 2])
		return tf.reshape(transposed, [shape[0] * dilation, -1, shape[2]])

def batch_to_time(value, dilation):
	with tf.name_scope('batch_to_time'):
		shape = value.get_shape()
		shape = [int(s) for s in shape]
		prepared = tf.reshape(value, [dilation, -1, shape[2]])
		transposed = tf.transpose(prepared, perm=[1, 0, 2])
		return tf.reshape(transposed,
						  [int(shape[0] / dilation), -1, shape[2]])

# See http://arxiv.org/abs/1502.01852
def he_uniform(filter_width, in_dim, scale=1):
	fan_in = filter_width * in_dim
	return np.sqrt(1. * scale / fan_in)

def conv1d(input_,
		   output_channels,
		   filter_width = 1,
		   stride       = 1,
		   name         = 'conv1d'
		  ):
	with tf.variable_scope(name):
		in_dim = input_.get_shape().as_list()[-1]

		scale = he_uniform(filter_width, in_dim)
		w = tf.get_variable('W', [filter_width, in_dim, output_channels],
			initializer = tf.random_uniform_initializer(minval=-scale, maxval=scale),
			trainable   = True)
		b = tf.get_variable('b', [output_channels], initializer=tf.constant_initializer(0.0))

		conv = tf.nn.conv1d(input_, w, stride = stride, padding = 'SAME')
		return tf.reshape(tf.nn.bias_add(conv, b), conv.get_shape())

def dilated_conv1d(input_, output_channels, dilation, 
	filter_width = 1, causal = False, name = 'dilated_conv'):
	
	
	if causal:
		# padding for masked convolution
		padding = [[0, 0], [(filter_width - 1) * dilation, 0], [0, 0]]
		padded = tf.pad(input_, padding)
	else:
		padding = [[0, 0], [(filter_width - 1) * dilation/2, (filter_width - 1) * dilation/2], [0, 0]]
		padded = tf.pad(input_, padding)
	
	if dilation > 1:
		transformed = time_to_batch(padded, dilation)
		conv = conv1d(transformed, output_channels, filter_width, name = name)
		restored = batch_to_time(conv, dilation)
	else:
		restored = conv1d(padded, output_channels, filter_width, name = name)

	
	result = tf.slice(restored,[0, 0, 0],[-1, int(input_.get_shape()[1]), -1])
	
	return result
