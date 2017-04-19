#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frederik Kratzert. frederik.kratzert(at)boku.ac.at
"""

import os
import tensorflow as tf
import numpy as np

valid_modes = ['vgg', 'length', 'date', 'all']

class VGG16:
    """
    This class stores the model definition of the original VGG16 network,
    as well as the adapted versions for additional feature fusing into the
    second-to-last layer.
    """

    def __init__(self, mode = 'vgg', num_classes=7, keep_prob = 0.5):

        """
        Arguments:

          mode: one of ['vgg', 'length', 'date', 'all']. Creates the model
            graphs for the four different model versions.
          num_classes: number of classes. 7 for this experiment, 1000 for the
            original vgg16 model
          keep_prob: dropout probability. 0.5 is the value of the original model

        """

        if mode not in valid_modes:
            raise ValueError("Bad input argument. 'Mode' must be one of 'vgg', 'length', 'date' or 'all'.")

        self.mode = mode
        self.num_classes = num_classes
        self.keep_prob = keep_prob



    def build(self, images, length = None, weeks = None):
        """
        This function creates the model graph depending on the selected 'mode'.

        Inputs:
            - images: tf.placeholder of size (N, 224, 224, 3) with N number of
                      images per batch.
            - length: (optional) tf.placeholder of size (N,1)
            - weeks: (optional) tf.placeholder of size (N, 52)
        """

        # First convolutional block of VGG16
        conv1_1 = self.conv_layer(images, 3, 3, 64, 'conv1_1')
        conv1_2 = self.conv_layer(conv1_1, 3, 3, 64, 'conv1_2')
        pool1 = self.max_pool(conv1_2, 'pool1')

        # Second convolutional block of VGG16
        conv2_1 = self.conv_layer(pool1, 3, 3, 128, 'conv2_1')
        conv2_2 = self.conv_layer(conv2_1, 3, 3, 128, 'conv2_2')
        pool2 = self.max_pool(conv2_2, 'pool2')

        # Third convolutional block of VGG16
        conv3_1 = self.conv_layer(pool2, 3, 3, 256, 'conv3_1')
        conv3_2 = self.conv_layer(conv3_1, 3, 3, 256, 'conv3_2')
        conv3_3 = self.conv_layer(conv3_2, 3, 3, 256, 'conv3_3')
        pool3 = self.max_pool(conv3_3, 'pool3')

        # Fourth convolutional block of VGG16
        conv4_1 = self.conv_layer(pool3, 3, 3, 512, 'conv4_1')
        conv4_2 = self.conv_layer(conv4_1, 3, 3, 512, 'conv4_2')
        conv4_3 = self.conv_layer(conv4_2, 3, 3, 512, 'conv4_3')
        pool4 = self.max_pool(conv4_3, 'pool4')

        # Fifth convolutional block of VGG16
        conv5_1 = self.conv_layer(pool4, 3, 3, 512, 'conv5_1')
        conv5_2 = self.conv_layer(conv5_1, 3, 3, 512, 'conv5_2')
        conv5_3 = self.conv_layer(conv5_2, 3, 3, 512, 'conv5_3')
        pool5 = self.max_pool(conv5_3, 'pool5')

        # flatten pool5 into one long vector
        pool5_flat = tf.reshape(pool5, [-1, 512*7*7])

        # first fully connected layer of original VGG16 model + dropout
        fc6 = self.fc_layer(pool5_flat, 512*7*7, 4096, name = 'fc6')
        drpt6 = tf.nn.dropout(fc6, self.keep_prob)

        #apply batch norm
        bn6 = self.batch_norm(drpt6, 'bn6')


        # Depending on mode, select which top to but on the convolutional core
        if self.mode == 'vgg':

            self.fc8 = self.vgg_top(bn6)

        elif self.mode == 'length':

            self.fc8 = self.length_top(bn6, length)

        elif self.mode == 'date':

            self.fc8 = self.weeks_top(bn6, weeks)

        elif self.mode == 'all':

            self.fc8 = self.all_top(bn6, length, weeks)


    def conv_layer(self, x, filter_height, filter_width, num_filters, name):
        """
        Helper class to create a convolutional layer of arbitrary filter
        dimensions, input channels and number of filter
        """
        #number of channels of the input layer
        in_channels = int(x.shape[-1].value)

        with tf.variable_scope(name) as scope:

            #create variables for weights and biases
            weights = tf.get_variable('weights', shape = [filter_height, filter_width,
                                                          in_channels, num_filters])
            biases = tf.get_variable('biases', shape = [num_filters])

            #perform convolution, bias add and relu activation
            conv = tf.nn.conv2d(x, weights, strides = [1,1,1,1], padding = 'SAME')
            bias = tf.nn.bias_add(conv, biases)
            relu = tf.nn.relu(bias, name = scope.name)

            return relu


    def fc_layer(self, x, num_in, num_out, name, relu=True):
        """
        Helper class to create a fully connected layer of arbitrary number of
        input neurons and output neurons
        """
        with tf.variable_scope(name) as scope:

            #create variables for weights and biases
            weights = tf.get_variable('weights', shape=[num_in, num_out])
            biases = tf.get_variable('biases', shape=[num_out])

            #calculate activation
            act = tf.nn.xw_plus_b(x, weights, biases, name=scope.name)

            #apply non-linearity depending on set relu flag
            if relu == True:
                return tf.nn.relu(act)
            else:
                return act


    def max_pool(self, x, name):
        """
        Helper function for the max pooling layer
        """
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1],
                            padding = 'SAME', name = name)


    def batch_norm(self, x, name, is_train = False):
        """
        Helper function for the batch_norm layer. Note that for the sake of
        testing the 'is_train' flag needs to be set to false. If you are
        interested in training a model with this function, you need to pass
        a 'is_train' = True during the trainings phase to correctly update the
        moving mean/std.
        """
        with tf.variable_scope(name) as scope:

            out = tf.contrib.layers.batch_norm(x, decay = 0.9, center = True,
                                            scale = True, updates_collections = None,
                                            is_training = is_train, fused = True,
                                            scope = scope.name)
            return out


    def vgg_top(self, bn6):
        """
        Helper function to create the upper part of the original VGG16 model
        starting from the fc7 layer.
        """
        #second fully connected layer (fc7) + batchnorm + dropout
        fc7 = self.fc_layer(bn6, 4096, 4096, name = 'fc7')
        bn7 = self.batch_norm(fc7, name = 'fc7')
        drpt7 = tf.nn.dropout(bn7, self.keep_prob)

        #output layer with self.num_class neurons and without non-linearity
        fc8 = self.fc_layer(drpt7, 4096, self.num_classes, name = 'fc8', relu = False)

        return fc8


    def length_top(self, bn6, length):
        """
        Helper function to create the upper part of the model, were the length
        feature is fused in as additional input.
        """
        #process length through seperate batch norm layer
        bn_length = self.batch_norm(length, 'input_length')

        #fuse image features and length feature together to one long vector
        fused_inputs = tf.concat([bn6, bn_length], axis = 1)

        #add fully connected layer (fc7) + batchnorm + dropout
        fc7 = self.fc_layer(fused_inputs, 4097, 4096, name = 'fc7')
        bn7 = self.batch_norm(fc7, name = 'fc7')
        drpt7 = tf.nn.dropout(bn7, self.keep_prob)

        #output layer with self.num_class neurons and without non-linearity
        fc8 = self.fc_layer(drpt7, 4096, self.num_classes, name = 'fc8', relu = False)

        return fc8


    def weeks_top(self, bn6, weeks):
        """
        Helper function to create the upper part of the model, were the one-hot
        week vector serves as additional feature.
        """

        #fuse week vector together with the bn6 output vector
        fused_inputs = tf.concat([bn6, weeks], axis = 1)

        #add fully connected layer (fc7) + batchnorm + dropout
        fc7 = self.fc_layer(fused_inputs, 4148, 4096, name = 'fc7')
        bn7 = self.batch_norm(fc7, name = 'fc7')
        drpt7 = tf.nn.dropout(bn7, self.keep_prob)

        #output layer with self.num_class neurons and without non-linearity
        fc8 = self.fc_layer(drpt7, 4096, self.num_classes, name = 'fc8', relu = False)

        return fc8


    def all_top(self, bn6, length, weeks):
        """
        Helper function to create the upper part of the model, were the one-hot
        week vector and the length feature are fused into the model.
        """
        #process length through seperate batch norm layer
        bn_length = self.batch_norm(length, 'input_length')

        #fuse image features and length feature together to one long vector
        fused_inputs = tf.concat([bn6, bn_length, weeks], axis = 1)

        #add fully connected layer (fc7) + batchnorm + dropout
        fc7 = self.fc_layer(fused_inputs, 4149, 4096, name = 'fc7')
        bn7 = self.batch_norm(fc7, name = 'fc7')
        drpt7 = tf.nn.dropout(bn7, self.keep_prob)

        #output layer with self.num_class neurons and without non-linearity
        fc8 = self.fc_layer(drpt7, 4096, self.num_classes, name = 'fc8', relu = False)

        return fc8
