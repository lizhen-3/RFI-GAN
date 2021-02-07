# tensorboard --logdir "./data_set/logs/rfi_net"
from __future__ import division
import os
import math
import time
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.colors
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
# import numpy as np
from six.moves import xrange
from sklearn.metrics import precision_score, recall_score, f1_score
import argparse
import cv2
from utils import *
from opsWithoutRes import *

EPOCH_NUM = 3
TRAIN_BATCH_SIZE =8
VALIDATION_BATCH_SIZE = 1
TEST_BATCH_SIZE = 1
PREDICT_BATCH_SIZE = 1
TRAIN_SET_SIZE = 2100
TEST_SET_SIZE = 76
# TEST_SET_SIZE = 96
TEST_SET_FAST_SIZE = 216
# TEST_SET_SIZE = 96
VALIDATION_SET_SIZE=50
EPS = 10e-5
FLAGS = None
CLASS_NUM = 2
TIMES = 2
smooth=0.1
PREDICT_DIRECTORY = './data_set/test'
TEST_DIRECTORY = './data_set/test'
PREDICT_SAVED_DIRECTORY = './data_set/predictions'
TEST_RESULT_DIRECTORY = './data_set/test_result/rfi_net_test_result'
TRAIN_RESULT_DIRECTORY = './data_set/test_result/rfi_net_train_result'
VALIDATION_RESULT_DIRECTORY='./data_set/test_result/rfi_net_validation_result'
CHECK_POINT_PATH = './data_set/saved_models/train/model.ckpt'
DATA_DIR = './data_set/'
MODEL_DIR = './data_set/saved_models'
LOG_DIR = './data_set/logs'

INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL = 256, 128, 1
OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, OUTPUT_IMG_CHANNEL = 256, 128, 1

TEST_SET_NAME1='HIMap_RSG7M_A1_24_MP_PXX_Z0_C0-M9703A_DPUA_20160321_001403_deal.h5'
TRAIN_SET_NAME = 'train_set.tfrecords'
VALIDATION_SET_NAME = 'real_tod_with_mask.h5'
TEST_SET_NAME = 'test_set.h5'

def read_image(file_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(file_queue)
    features = tf.parse_single_example(
        serialized_example,
        features={
            'label': tf.FixedLenFeature([], tf.string),
            'image_raw': tf.FixedLenFeature([], tf.string)
        })
    image = tf.decode_raw(features['image_raw'], tf.float64)
    image = tf.cast(image, tf.float32)
    image = tf.reshape(image, [INPUT_IMG_HEIGHT, INPUT_IMG_WIDE, INPUT_IMG_CHANNEL])

    label = tf.decode_raw(features['label'], tf.uint8)
    label = tf.cast(label, tf.float32)
    label = tf.reshape(label, [OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, OUTPUT_IMG_CHANNEL])
    print(image)
    print(label)
    return image, label

def read_image_batch(file_queue, batch_size):
    img, label = read_image(file_queue)
    min_after_dequeue = 100
    capacity = 4000
    image_batch, label_batch = tf.train.shuffle_batch(
        tensors=[img, label], batch_size=batch_size,
        capacity=capacity, min_after_dequeue=min_after_dequeue)
    image_batch = tf.reshape(image_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, OUTPUT_IMG_CHANNEL])
    label_batch = tf.reshape(label_batch, [batch_size, OUTPUT_IMG_HEIGHT, OUTPUT_IMG_WIDE, OUTPUT_IMG_CHANNEL])
    print(image_batch)
    print(label_batch)
    return image_batch, label_batch

class RFI_Gan:

    def __init__(self, train_set_name=TRAIN_SET_NAME, test_set_name=TEST_SET_NAME, validation_set_name=VALIDATION_SET_NAME,
                 input_img_height=INPUT_IMG_HEIGHT, input_img_wide=INPUT_IMG_WIDE, input_img_channel=INPUT_IMG_CHANNEL,
                 output_img_channel=OUTPUT_IMG_CHANNEL,output_img_height=OUTPUT_IMG_HEIGHT, output_img_wide=OUTPUT_IMG_WIDE,
                 L1_lambda=10):
        self.input_image = None
        self.input_label = None
        self.fake_label=None
        self.cast_image = None
        self.cast_label = None
        self.keep_prob = None
        # self.lamb = None
        self.result_expand = None
        self.is_traing = None
        self.loss, self.loss_mean, self.loss_all, self.train_step = [None] * 4
        self.prediction, self.correct_prediction, self.accuracy = [None] * 3
        self.result_conv = {}
        self.result_relu = {}
        self.result_maxpool = {}
        self.result_from_contract_layer = {}
        self.w_0 = None
        self.learning_rate = None
        self.train_set_name = train_set_name
        self.test_set_name = test_set_name
        self.validation_set_name = validation_set_name
        self.input_img_height = input_img_height
        self.input_img_wide = input_img_wide
        self.input_img_channel = input_img_channel
        self.output_img_channel=output_img_channel
        self.output_img_height = output_img_height
        self.output_img_wide = output_img_wide
        self.L1_lambda = L1_lambda
        # self.checkpoint_dir = checkpoint_dir


    def discriminator(self, image, y=None, reuse=False):

        with tf.variable_scope("discriminator"):
            # image is 256 x 128 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False

            #layer_1_expand
            w1=init_w(self.lamb,shape=[3,3,INPUT_IMG_CHANNEL*2,32],name='d_w1')
            result_conv=tf.nn.conv2d(input=image , filter=w1,strides=[1,1,1,1],padding='SAME',name='d_conv1')
            normed_batch = batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='d_layer_1_expand')
            h1 = tf.nn.relu(features=normed_batch, name='d_relu1')
            l1=tf.nn.max_pool(h1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #(128,64,32)

            #layer_2_expand
            w2=init_w(self.lamb,shape=[3, 3, 32, 64], name='d_w2')
            result_conv = tf.nn.conv2d(
                input=l1,
                filter=w2, strides=[1, 1, 1, 1], padding='SAME', name='d_conv2'
            )
            normed_batch =batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='d_layer_2_expand')
            h2 = tf.nn.relu(features=normed_batch, name='d_relu2')
            l2 = tf.nn.max_pool(h2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            #(64,32,64)

            # layer_3_expand
            w3 =init_w(self.lamb,shape=[3, 3, 64, 128], name='d_w3')
            result_conv = tf.nn.conv2d(
                input=l2,
                filter=w3, strides=[1, 1, 1, 1], padding='SAME', name='d_conv3'
            )
            normed_batch =batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='d_layer_3_expand')
            h3 = tf.nn.relu(features=normed_batch, name='d_relu3')
            l3 = tf.nn.max_pool(h3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
            # (32,16,128)

            # layer_4_expand
            w4 =init_w(self.lamb,shape=[3, 3, 128, 1], name='d_w4')
            result_conv = tf.nn.conv2d(
                input=l3,
                filter=w4, strides=[1, 1, 1, 1], padding='SAME', name='d_conv4'
            )
            normed_batch =batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='d_layer_4_expand')
            h4= tf.nn.relu(features=normed_batch, name='d_relu4')  # print(h4.shape)  (8, 32, 16, 1)
            l4 = tf.nn.max_pool(h4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # (8,16,8,1)
            print(l4)
            shape=l4.shape
            size0=shape[0]
            size0=tf.cast(size0, tf.int32)
            size1 = shape[1]
            size1 = tf.cast(size1, tf.int32)
            size2 = shape[2]
            size2 = tf.cast(size2, tf.int32)
            # #fully connected 1
            w5 = tf.get_variable(
                    name='d_w5', regularizer=l2_regularizer(scale=self.lamb),
                    initializer=tf.truncated_normal(shape=[size1 * size2, 1], dtype=tf.float32))
            b5 = init_b(shape=[1],name='d_b5')
            l5_flat = tf.reshape(l4, [size0,-1])
            l5 = tf.matmul(l5_flat,w5) + b5
            return tf.nn.sigmoid(l5), l5
    def generator(self):
        print(self.input_image)
        normed_batch =batch_norm(x=self.input_image, is_training=self.is_traing, var_scope_name='g_input')

        with tf.name_scope('g_layer_1'), tf.variable_scope('g_layer_1'):
            w_expand =init_w(self.lamb,shape=[3, 3, INPUT_IMG_CHANNEL, 32], name='w_expand') #8，256，128，1
            result_conv = tf.nn.conv2d(input=normed_batch, filter=w_expand, strides=[1, 1, 1, 1], padding='SAME', name='conv')#
            normed_batch =batch_norm(x=result_conv, is_training=self.is_traing, var_scope_name='layer_1_expand')
            result_relu = tf.nn.relu(features=normed_batch, name='relu')

            result_unit_down =unit_down(self.is_traing,self.lamb,layer_num=1, input_data=result_relu)#channels is 64
            self.result_from_contract_layer[1] = result_unit_down

            result_maxpool = tf.nn.max_pool(value=result_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')# (256*128*64)to(128*64*64)
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_2'), tf.variable_scope('g_layer_2'): # layer 2(128*64*64)
            # unit_down 128*64*128
            result_unit_down =unit_down(self.is_traing,self.lamb,layer_num=2, input_data=result_dropout)
            self.result_from_contract_layer[2] = result_unit_down

            result_maxpool = tf.nn.max_pool(value=result_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool')# maxpool  64*32*128
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_3'), tf.variable_scope('g_layer_3'):
            result_unit_down =unit_down(self.is_traing,self.lamb,layer_num=3, input_data=result_dropout)# unit_down  64*32*256
            self.result_from_contract_layer[3] = result_unit_down

            result_maxpool = tf.nn.max_pool(value=result_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool') # maxpool  32*16*256
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_4'), tf.variable_scope('g_layer_4'):
            result_unit_down =unit_down(self.is_traing,self.lamb,layer_num=4, input_data=result_dropout)# unit_down 32*16*512
            self.result_from_contract_layer[4] = result_unit_down

            result_maxpool = tf.nn.max_pool(value=result_unit_down, ksize=[1, 2, 2, 1],
                strides=[1, 2, 2, 1], padding='VALID', name='maxpool') # maxpool 16*8*512
            result_dropout = tf.nn.dropout(x=result_maxpool, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_5'), tf.variable_scope('g_layer_5'):# layer 5 (bottom 16 * 8 * 1024)
            result_unit_down =unit_down(self.is_traing,self.lamb,layer_num=5, input_data=result_dropout)
            result_relu_3 =up_sample(self.is_traing,self.lamb,layer_num=5, input_data=result_unit_down)
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_6'), tf.variable_scope('g_layer_6'):
            result_merge =tf.concat(values=[self.result_from_contract_layer[4], result_dropout], axis=-1) # copy and merge  32*16*1024
            result_merge_normed =batch_norm(x=result_merge, is_training=self.is_traing,var_scope_name='layer_6_merge')
            result_unit_up =unit_up(self.is_traing,self.lamb ,layer_num=6, input_data=result_merge_normed) # unit_up32* 16*512

            result_relu_3 =up_sample(self.is_traing,self.lamb,layer_num=5, input_data=result_unit_up)# up sample64* 32*256
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_7'), tf.variable_scope('g_layer_7'):
            result_merge =tf.concat(values=[self.result_from_contract_layer[3], result_dropout], axis=-1)# copy and merge 64*32*512
            result_merge_normed =batch_norm(x=result_merge, is_training=self.is_traing,var_scope_name='layer_7_merge')
            result_unit_up =unit_up(self.is_traing,self.lamb,layer_num=7, input_data=result_merge_normed) # unit_up  64* 32*256
            result_relu_3 =up_sample(self.is_traing,self.lamb,layer_num=5, input_data=result_unit_up)#128* 64*128
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_8'), tf.variable_scope('g_layer_8'):
            result_merge = tf.concat(values=[self.result_from_contract_layer[2], result_dropout], axis=-1)# copy and merge  128*64*256
            result_merge_normed =batch_norm(x=result_merge, is_training=self.is_traing,var_scope_name='layer_8_merge')
            result_unit_up =unit_up(self.is_traing,self.lamb,layer_num=8, input_data=result_merge_normed) # unit_up  128*64*128
            result_relu_3 =up_sample(self.is_traing,self.lamb,layer_num=5, input_data=result_unit_up)#256*128*64
            result_dropout = tf.nn.dropout(x=result_relu_3, keep_prob=self.keep_prob)

        with tf.name_scope('g_layer_9'), tf.variable_scope('g_layer_9'):
            result_merge = tf.concat(values=[self.result_from_contract_layer[1], result_dropout], axis=-1)# copy and merge  256*128*128
            result_merge_normed =batch_norm(x=result_merge, is_training=self.is_traing,var_scope_name='layer_9_merge')
            result_unit_up =unit_up(self.is_traing,self.lamb,layer_num=9, input_data=result_merge_normed)#256*128*64

            w =init_w(self.lamb,shape=[1, 1, 64,1], name='w')
            result_conv_3 = tf.nn.conv2d(input=result_unit_up, filter=w,
                strides=[1, 1, 1, 1], padding='VALID', name='conv_3') #256*128*1  set CLASS-NUM=1
            normed_batch =batch_norm(x=result_conv_3, is_training=self.is_traing, var_scope_name='layer_9_conv_3')
            return tf.sigmoid(normed_batch)
            # return normed_batch

    def train(self, train_batch_size=TRAIN_BATCH_SIZE, train_file_path=None, train_result_path=None,log_path=None, model_file_path=None,
              model_name="rfi_net/model.ckpt"):
        self.input_image = tf.placeholder(dtype=tf.float32,
            shape=[TRAIN_BATCH_SIZE, self.input_img_height, self.input_img_wide, self.input_img_channel],
            name='input_images'
        )

        self.input_label = tf.placeholder(
            dtype=tf.float32,
            shape=[TRAIN_BATCH_SIZE, self.output_img_height, self.output_img_wide,self.output_img_channel],
            name='input_labels'
        )
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
        self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')


        if train_file_path is None:
            train_file_path = os.path.join(DATA_DIR, TRAIN_SET_NAME)
        if log_path is None: # LOG_DIR='../data_set/logs'
            log_path = os.path.join(LOG_DIR, "rfi_net")
        if model_file_path is None:# MODEL_DIR='../data_set/saved_models' model_name="rfi_net/model.ckpt"
            model_file_path = os.path.join(MODEL_DIR, model_name)
        if train_result_path is None:
            train_result_path = TRAIN_RESULT_DIRECTORY
        if not os.path.lexists(train_result_path):
            os.makedirs(train_result_path)

        self.fake_label = self.generator()

        random=tf.truncated_normal([TRAIN_BATCH_SIZE,256,128, 1])
        self.input_label_after=tf.add(self.input_label,random)
        self.fake_label_after=tf.add(self.fake_label,random)
        self.real = tf.concat([self.input_image, self.input_label_after], 3)
        self.fake = tf.concat([self.input_image, self.fake_label_after], 3)

        self.D_, self.D_logits_ = self.discriminator(self.fake, reuse=False)
        self.D,self.D_logits = self.discriminator(self.real, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D)*(1-smooth),logits=self.D_logits))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.zeros_like(self.D_),logits=self.D_logits_))

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.ones_like(self.D_)*(1-smooth), logits=self.D_logits_))\
        + self.L1_lambda * tf.reduce_mean(tf.abs(self.input_label - self.fake_label))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        d_optim = tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.9,beta2=0.99) \
            .minimize(self.d_loss,var_list=self.d_vars)
        g_optim = tf.train.AdamOptimizer(learning_rate=0.00001,beta1=0.9,beta2=0.99) \
            .minimize(self.g_loss,var_list=self.g_vars)

        train_image_filename_queue = tf.train.string_input_producer(
            string_tensor=tf.train.match_filenames_once(train_file_path), num_epochs=EPOCH_NUM, shuffle=True)
        trainImages, trainLabels = read_image_batch(train_image_filename_queue, train_batch_size) # train_batch_size=8
        all_parameters_saver = tf.train.Saver()

        with tf.Session() as sess:
            self.g_sum = tf.summary.merge([self.d__sum,self.d_loss_fake_sum, self.g_loss_sum])
            self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
            self.writer = tf.summary.FileWriter(log_path, sess.graph)

            sess.run(tf.local_variables_initializer())
            sess.run(tf.global_variables_initializer())

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(coord=coord)
            try:
                epoch = 1
                counter = 1
                number = 0
                print(coord.should_stop())
                while not coord.should_stop():
                    train_images, train_labels = sess.run([trainImages, trainLabels])
                    print(train_images.shape)
                    # train_images = tf.cast(train_images, tf.float32)
                    # train_labels = tf.cast(train_labels, tf.uint8)

                    images = sess.run(self.fake_label,
                                      feed_dict={
                                          self.input_image: train_images,
                                          self.keep_prob: 1.0,
                                          self.lamb: 0.004, self.is_traing: True}
                                      )
                    _, summary_str = sess.run([d_optim,self.d_sum],
                                  feed_dict={self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_traing: True})

                    self.writer.add_summary(summary_str, epoch)

                    _, summary_str=sess.run([g_optim,self.g_sum],
                                  feed_dict={self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0,
                            self.lamb: 0.004, self.is_traing: True})
                    self.writer.add_summary(summary_str, epoch)

                    _, summary_str=sess.run([g_optim,self.g_sum],
                                  feed_dict={self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0,self.is_traing: True})
                    self.writer.add_summary(summary_str, epoch)

                    _, summary_str = sess.run([g_optim, self.g_sum],
                                              feed_dict={self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0, self.is_traing: True})
                    self.writer.add_summary(summary_str, epoch)

                    errD_fake = self.d_loss_fake.eval({self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0, self.is_traing: True})
                    errD_real = self.d_loss_real.eval({self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0, self.is_traing: True})
                    errG = self.g_loss.eval({self.input_image: train_images, self.input_label: train_labels,self.keep_prob: 1.0,self.is_traing: True})
                    precision=0.0
                    recall=0.0
                    f1=0.0
                    for i in xrange(len(images)):
                        img=np.squeeze(images[i])>0.5 #output true false
                        img=img.astype(int) #output 0 1
                        img=np.reshape(img,newshape=(-1))

                        mask=np.squeeze(train_labels[i])
                        mask=mask.astype(int)
                        mask = np.reshape(mask, newshape=(-1))

                        precision += precision_score(y_true=mask, y_pred=img, average="binary")
                        recall += recall_score(y_true=mask, y_pred=img, average="binary")
                        f1 += f1_score(y_true=mask, y_pred=img, average="binary")

                    precision=precision/8.0
                    recall=recall/8.0
                    f1=f1/8.0
                    print("Epoch: [%2d] , d_loss: %.8f, g_loss: %.8f, precision: %.8f, recall: %.8f, f1: %.8f" \
                    % (epoch, errD_fake + errD_real, errG, precision, recall, f1))

            except tf.errors.OutOfRangeError:
                print('Done training -- epoch limit reached')
            finally:
                # When done, ask the threads to stop.
                all_parameters_saver.save(sess=sess, save_path=model_file_path)
                coord.request_stop()
                coord.join(threads)
        print("Done training")

    def validate(self,validation_set_size=VALIDATION_SET_SIZE, validation_file_path=None, validation_result_path=None,model_file_path=None, model_name="rfi_net/model.ckpt"):
        import h5py as h5

        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)
        # DATA_DIR = '../data_set/'VALIDATION_SET_NAME = 'validation_set.tfrecords'
        if validation_file_path is None:
            validation_file_path = os.path.join(DATA_DIR, VALIDATION_SET_NAME)

        if validation_result_path is None:
            validation_result_path = VALIDATION_RESULT_DIRECTORY
        if not os.path.lexists(validation_result_path):
            os.makedirs(validation_result_path)
        validation_result_path = os.path.join(VALIDATION_RESULT_DIRECTORY, 'validation_result.h5')
        file_to_read = h5.File(validation_file_path, 'r')
        file_to_write = h5.File(validation_result_path, 'w')

        self.input_image = tf.placeholder(
            dtype=tf.float32,
            shape=[VALIDATION_BATCH_SIZE, self.input_img_height, self.input_img_wide, self.input_img_channel],
            name='input_images'
        )

        self.input_label = tf.placeholder(
            dtype=tf.float32,
            shape=[VALIDATION_BATCH_SIZE, self.output_img_height, self.output_img_wide, self.output_img_channel],
            name='input_labels'
        )
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
        self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')

        self.fake_label = self.generator()

        all_parameters_saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())

            all_parameters_saver.restore(sess=sess, save_path=model_file_path)
            for index in range(validation_set_size):
                tod = np.reshape(
                    a=file_to_read['%04d/tod' % index].value,
                    newshape=(VALIDATION_BATCH_SIZE, self.input_img_height, self.input_img_wide, self.input_img_channel))
                # print('%04d/tod' % index)
                images = sess.run(self.fake_label,
                                  feed_dict={self.input_image: tod,
                                             self.keep_prob: 1.0,
                                             self.lamb: 0.004, self.is_traing: False})

                img = np.squeeze(images[0]) > 0.5
                # img=img.astype(int)
                cv2.imwrite(os.path.join(VALIDATION_RESULT_DIRECTORY, '%d_temp.jpg' % index), img * 255)  # * 255
                img = img.astype(int)

                img = np.reshape(img, newshape=(-1))
                mask = np.reshape(file_to_read['%04d/mask' % index].value, newshape=(-1))
                # print(mask)
                precision = precision_score(y_true=mask, y_pred=img, average="binary")
                recall = recall_score(y_true=mask, y_pred=img, average="binary")
                f1 = f1_score(y_true=mask, y_pred=img, average="binary")
                print("Epoch: [%2d], precision: %.8f, recall: %.8f, f1: %.8f" \
                      % (index, precision, recall, f1))

                file_to_write['%02d/predict' % index] = img
                file_to_write['%02d/ground_truth' % index] = file_to_read['%04d/mask' % index].value
                if index % 10 == 0:
                    print('Done testing %.2f%%' % (index * 100 / VALIDATION_SET_SIZE))
            file_to_read.close()
            file_to_write.close()
        print('Done testing, validation result in floder validaton_saved')


    def test(self, test_set_size=TEST_SET_SIZE, test_file_path=None, test_result_path=None, model_file_path=None, model_name="rfi_net/model.ckpt"):
        import cv2
        import time
        import numpy as np
        import h5py as h5

        if model_file_path is None:
            model_file_path = os.path.join(MODEL_DIR, model_name)

        if test_file_path is None:
            test_file_path = os.path.join(DATA_DIR, TEST_SET_NAME)

        if test_result_path is None:
            test_result_path = TEST_RESULT_DIRECTORY

        if not os.path.lexists(test_result_path):
            os.makedirs(test_result_path)

        test_result_path = os.path.join(TEST_RESULT_DIRECTORY, 'Score_temp.h5')
        file_to_read = h5.File(test_file_path, 'r')
        file_to_write = h5.File(test_result_path, 'w')

        self.input_image = tf.placeholder(
            dtype=tf.float32,
            shape=[TEST_BATCH_SIZE, self.input_img_height, self.input_img_wide, self.input_img_channel],
            name='input_images'
        )

        self.input_label = tf.placeholder(
            dtype=tf.float32,
            shape=[TEST_BATCH_SIZE, self.output_img_height, self.output_img_wide, self.output_img_channel],
            name='input_labels'
        )
        self.keep_prob = tf.placeholder(dtype=tf.float32, name='keep_prob')
        self.lamb = tf.placeholder(dtype=tf.float32, name='lambda')
        self.is_traing = tf.placeholder(dtype=tf.bool, name='is_traing')

        self.fake_label = self.generator()

        all_parameters_saver=tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            all_parameters_saver.restore(sess=sess, save_path=model_file_path)

            precision_sum=0
            recall_sum=0
            f1_score_sum=0
            start_time=time.time();
            for index in range(test_set_size):
                tod = np.reshape(
                    a=file_to_read['%02d/tod' % index].value,
                    newshape=(1, self.input_img_height, self.input_img_wide, self.input_img_channel))
                image = sess.run(self.fake_label,
                                  feed_dict={self.input_image: tod,
                                             self.keep_prob: 1.0,
                                             self.lamb: 0.004, self.is_traing: False})

                img=np.squeeze(image)>0.5
                cv2.imwrite(os.path.join(TEST_RESULT_DIRECTORY, '%d_temp.jpg' % index), img * 255)  # * 255
                img = img.astype(int)
                img = np.reshape(img, newshape=(-1))
                mask = np.reshape(file_to_read['%02d/rfi_mask' % index].value, newshape=(-1))

                precision = precision_score(y_true=mask, y_pred=img, average="binary")
                recall = recall_score(y_true=mask, y_pred=img, average="binary")
                f1 = f1_score(y_true=mask, y_pred=img, average="binary")
                print("Epoch: [%2d], precision: %.8f, recall: %.8f, f1: %.8f" \
                  % (index,precision, recall, f1))
                precision_sum+=precision
                recall_sum+=recall
                f1_score_sum+=f1

                file_to_write['%02d/predict' % index] = img
                if index % 10 == 0:
                    print('Done testing %.2f%%' % (index *100/ test_set_size))
            end_time=time.time();
            time=end_time-start_time;
            print("time: %.8f" % time)
            print("precision: %.8f, recall: %.8f, f1: %.8f" \
                  % (precision_sum/TEST_SET_SIZE, recall_sum/TEST_SET_SIZE, f1_score_sum/TEST_SET_SIZE))
            file_to_read.close()
            file_to_write.close()
        print('Done testing, test result in floder test_saved')



def main():
    net = RFI_Gan()
    # net.set_up_net(TRAIN_BATCH_SIZE)
    # net.train()
    #
    # net.test()
    net.roc()
    # net.test(model_file_path="./data_set/saved_models/rfi_net/model.ckpt")

    # net.validate()

    # net.set_up_net(TEST_BATCH_SIZE)
    # net.test_time(test_batch_size=1, test_size=76, height=256, wide=128)
    # net.set_up_net(PREDICT_BATCH_SIZE)
    # net.predict()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # data path
    parser.add_argument(
        '--data_dir', type=str, default=DATA_DIR,
        help='Directory for storing input data_set')

    # model saved into
    parser.add_argument(
        '--model_dir', type=str, default=MODEL_DIR,
        help='output model path')

    # log saved into
    parser.add_argument(
        '--log_dir', type=str, default=LOG_DIR,
        help='TensorBoard log path')

    FLAGS, _ = parser.parse_known_args()

    if not os.path.exists(FLAGS.model_dir):
        os.makedirs(FLAGS.model_dir)

    if not os.path.exists(FLAGS.log_dir):
        os.makedirs(FLAGS.log_dir)

    main()
