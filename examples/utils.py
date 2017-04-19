#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frederik Kratzert. frederik.kratzert(at)boku.ac.at
"""

import os
import itertools

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from skimage.io import imread
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes


class_names = '''Lota lota, burbot, Aalrutte
Squalius cephalus, chub, Aitel
Salmo trutta, brown trout, Bachforelle
Abramis brama, bream, Brachse
Perca fluviatilis, perch, Flussbarsch
Chondrostoma nasus, common nase, Nase
Oncorhynchus mykiss, rainbow trout, Regenbogenforelle'''.split("\n")


def plot_confusion_matrix(cm, classes, title='Confusion matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    """
    plt.figure(figsize=(15,10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, fontsize=15)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=14)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j], horizontalalignment="center", fontsize=12,
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label', fontsize=15)
    plt.xlabel('Predicted label', fontsize=15)


def plots(df, idx, data_dir, plot_corrects = True, figsize=(15,10)):
    """
    This is a plot function used in the explorer notebook. It is used to display
    several images per class with the achived score in the title.
    Through the 'plot_corrects' flag we can choose if the probability of the
    true class will be plotted in the title or the one with the highest probability
    which can be a false class.
    """
    f = plt.figure(figsize = figsize)
    for i in range(len(idx)):
        img_path = df.get_value(idx[i], 'image_path')
        img = imread(os.path.join(data_dir, img_path))
        sp = f.add_subplot(1, len(idx), i+1)
        plt.imshow(img, interpolation=None)
        plt.axis('off')
        #create image title depending on plot_corrects flag
        if plot_corrects:
            corr_label = int(df.get_value(idx[i], 'label'))
            score = df.get_value(idx[i], 'score'+str(corr_label))
            t = class_names[corr_label].split(',')[1] + ": {:.2f}".format(score)
        else:
            false_label = int(df.get_value(idx[i], 'pred'))
            score = df.get_value(idx[i], 'score'+str(false_label))
            t = class_names[false_label].split(',')[1] + ": {:.2f}".format(score)
        sp.set_title(t, fontsize=16)



class data_handler():
    """
    Class for efficient data reading using tensorflows input queue's.
    """

    def __init__(self, file_name, data_dir, capacity=150):

        # Initialize some class variables
        self.NUM_CLASSES = 7
        self.FILE_NAME = file_name
        self.DATA_DIR = data_dir
        self.VGG_MEAN = tf.constant([123.68, 116.779, 103.939], dtype=tf.float32)

        # read the csv content and store them in sperate lists
        index_list, image_list, length_list, weeks_list = self.read_data_file()

        # convert lists to tensorflow tensors
        index_list = ops.convert_to_tensor(index_list, dtype=dtypes.int32)
        image_list = ops.convert_to_tensor(image_list, dtype=dtypes.string)
        length_list = ops.convert_to_tensor(length_list, dtype=dtypes.float32)
        weeks_list = ops.convert_to_tensor(weeks_list, dtype=dtypes.float32)

        # create input queue
        self.input_queue = tf.train.slice_input_producer([index_list, image_list,
                                                          length_list, weeks_list],
                                                        shuffle = False,
                                                        num_epochs = 1,
                                                        capacity = capacity)


    def read_data_file(self):
        """
        Reads csv files and stores index, image, length and weeks information
        in seperate lists. The index is used to identify which sample is
        processed.
        """
        index_list = []
        image_list = []
        length_list = []
        weeks_list = []

        with open(self.FILE_NAME) as f:
            for i, row in enumerate(f):

                #skip header
                if i==0:
                    pass

                #add content to lists
                else:
                    row_content = row.split(';')
                    index_list.append(int(row_content[0]))
                    image_list.append(os.path.join(self.DATA_DIR, row_content[1]))
                    length_list.append(int(row_content[6]))
                    weeks_list.append([int(x) for x in row_content[7:]])

        return index_list, image_list, length_list, weeks_list


    def read_data_from_disk(self):
        """
        This function is used to read the data stored in the input queues into
        memory.
        """
        #parse numeric information from input queue
        index = self.input_queue[0]
        length = self.input_queue[2]
        weeks = self.input_queue[3]

        #read the image from disk and load into memory using tensorflow functions
        image_file = tf.read_file(self.input_queue[1])
        image = tf.image.decode_png(image_file, channels=3)

        #resize to [224,224,3]
        image = tf.image.resize_images(image, [224, 224])

        #subtract VGG mean
        image = tf.subtract(image, self.VGG_MEAN)

        #RGB -> BGR
        image = image[:,:,::-1]

        return index, image, length, weeks
