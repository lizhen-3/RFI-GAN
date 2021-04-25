from tensorflow.contrib.layers import l2_regularizer
import tensorflow as tf

def init_w(lamb,shape,name=None):
    stddev = tf.sqrt(x=2.0 / (shape[0] * shape[1] * shape[2] * shape[3]))
    w = tf.get_variable(
        name=name, regularizer=l2_regularizer(scale=lamb),
        initializer=tf.truncated_normal(shape=shape, stddev=stddev, dtype=tf.float32))
    return w

def init_b(shape, name):
    with tf.name_scope('init_b'):
        return tf.Variable(initial_value=tf.random_normal(shape=shape, dtype=tf.float32), name=name)
def batch_norm(x, is_training, eps=10e-5, decay=0.9, affine=True, var_scope_name='BatchNorm2d'):
        from tensorflow.python.training.moving_averages import assign_moving_average
        with tf.variable_scope(var_scope_name):
            params_shape = x.shape[-1:]
            moving_mean = tf.get_variable(name='mean', shape=params_shape, initializer=tf.zeros_initializer,
                                          trainable=False)
            moving_var = tf.get_variable(name='variance', shape=params_shape, initializer=tf.ones_initializer,
                                         trainable=False)

            def mean_var_with_update():
                mean_this_batch, variance_this_batch = tf.nn.moments(x, list(range(len(x.shape) - 1)), name='moments')
                with tf.control_dependencies([
                    assign_moving_average(moving_mean, mean_this_batch, decay),
                    assign_moving_average(moving_var, variance_this_batch, decay)
                ]):
                    return tf.identity(mean_this_batch), tf.identity(variance_this_batch)

            mean, variance = tf.cond(is_training, mean_var_with_update, lambda: (moving_mean, moving_var))
            if affine:  # If you want to scale with  and gamma
                beta = tf.get_variable('beta', params_shape, initializer=tf.zeros_initializer)
                gamma = tf.get_variable('gamma', params_shape, initializer=tf.ones_initializer)
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=beta, scale=gamma,
                                                   variance_epsilon=eps)
            else:
                normed = tf.nn.batch_normalization(x, mean=mean, variance=variance, offset=None, scale=None,
                                                   variance_epsilon=eps)
            return normed


def unit_down(is_training,lamb,layer_num, input_data):
        layer_name = 'res_unit_down_%d' % layer_num
        channels_num = input_data.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name):
            # w_1= init_w(lamb,shape=[1, 1, channels_num, 2 * channels_num], name='w_1')#Double the number of channels
            # result_conv_1 = tf.nn.conv2d(input=input_data, filter=w_1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            # split_from_input = batch_norm(x=result_conv_1, is_training=is_training, var_scope_name='%s_split' % layer_name)

            w_2 =init_w(lamb,shape=[3, 3, channels_num, 2 * channels_num], name='w_2')
            result_conv_2 = tf.nn.conv2d(input=input_data, filter=w_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch =batch_norm(x=result_conv_2, is_training=is_training, var_scope_name='%s_conv_2' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_3
            w_3 =init_w(lamb,shape=[3, 3, 2 * channels_num, 2 * channels_num], name='w_3')
            result_conv_2 = tf.nn.conv2d(input=result_relu_2, filter=w_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch =batch_norm(x=result_conv_2, is_training=is_training, var_scope_name='%s_conv_3' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_4
            w_4 =init_w(lamb,shape=[3, 3, 2 * channels_num, 2 * channels_num], name='w_4')
            result_conv_1 = tf.nn.conv2d(input=result_relu_2, filter=w_4, strides=[1, 1, 1, 1], padding='SAME', name='conv_4')
            normed_batch =batch_norm(x=result_conv_1, is_training=is_training, var_scope_name='%s_conv_4' % layer_name)

            result_relu_add = tf.nn.relu(normed_batch, name='relu')

            return result_relu_add

def unit_up(is_training,lamb,layer_num, input_data):
        layer_name = 'res_unit_up_%d' % layer_num
        channels_num = input_data.get_shape().as_list()[-1]
        with tf.variable_scope(layer_name):

            # conv_1
            w_1 =init_w(lamb,shape=[3, 3, channels_num, channels_num // 2], name='w_1')
            result_conv_1 = tf.nn.conv2d(input=input_data, filter=w_1, strides=[1, 1, 1, 1], padding='SAME', name='conv_1')
            normed_batch =batch_norm(x=result_conv_1, is_training=is_training, var_scope_name='%s_conv_1' % layer_name)
            result_relu_1 = tf.nn.relu(normed_batch, name='relu')

            # conv_2
            w_2 =init_w(lamb,shape=[3, 3, channels_num // 2, channels_num // 2], name='w_2')
            result_conv_2 = tf.nn.conv2d(input=result_relu_1, filter=w_2, strides=[1, 1, 1, 1], padding='SAME', name='conv_2')
            normed_batch =batch_norm(x=result_conv_2, is_training=is_training, var_scope_name='%s_conv_2' % layer_name)
            result_relu_2 = tf.nn.relu(features=normed_batch, name='relu')

            # conv_3
            w_3 =init_w(lamb,shape=[3, 3, channels_num // 2, channels_num // 2], name='w_3')
            result_conv_2 = tf.nn.conv2d(input=result_relu_2, filter=w_3, strides=[1, 1, 1, 1], padding='SAME', name='conv_3')
            normed_batch =batch_norm(x=result_conv_2, is_training=is_training, var_scope_name='%s_conv_3' % layer_name)

            result_relu_add = tf.nn.relu(normed_batch, name='relu')
            return result_relu_add


def up_sample(is_training,lamb,layer_num, input_data):
        batch_size, height, wide, channels_num = input_data.get_shape().as_list()
        w_upsample =init_w(lamb,shape=[2, 2, channels_num // 2, channels_num], name='w_upsample')
        result_up = tf.nn.conv2d_transpose(value=input_data, filter=w_upsample,output_shape=[batch_size, height * 2, wide * 2, channels_num // 2],
                                           strides=[1, 2, 2, 1], padding='VALID', name='Up_Sample')
        normed_batch =batch_norm(x=result_up, is_training=is_training, var_scope_name='layer_%d_conv_up' % layer_num)
        result_relu_3 = tf.nn.relu(features=normed_batch, name='relu')
        return result_relu_3
