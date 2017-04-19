#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Frederik Kratzert. frederik.kratzert(at)boku.ac.at
"""
import sys
import os
import argparse


import numpy as np
import pandas as pd
import tensorflow as tf

from datetime import datetime
from scipy.stats import mode
from sklearn.metrics import accuracy_score

from model import VGG16
from utils import data_handler

BATCH_SIZE = 1


def evaluate_model(model, checkpoint_path, data_dir, test_file):
    """
    Note: Because I chose to have a prime number of images in the testset,
    the testset evaluation is done with a batch size of 1
    """
    
    #initialize dropout probability as tensorflow constant (1 for test phase)
    keep_prob = tf.constant(1, dtype=tf.float32)
    
    #initialize tensorflow placeholder
    image = tf.placeholder(tf.float32, shape=[1, 224, 224, 3])
    length = tf.placeholder(tf.float32, shape=[1,1])
    weeks = tf.placeholder(tf.float32, shape=[1, 52])
   
    # read in the test data file
    df = pd.read_csv(test_file, sep=';')
     
    # create dataframe in which the results will be stored
    results = pd.DataFrame(data=None, index=None, columns = ['image_path', 'video_name', 
                                                        'label', 'pred', 'score0',
                                                        'score1', 'score2', 'score3',
                                                        'score4', 'score5', 'score6'])
    
    print("{} Create model graph and start input queue.".format(datetime.now()))
    
    # use CPU (also if GPU available) to load and preprocess data
    with tf.device('/cpu:0'):
        input_data = data_handler(test_file, data_dir)
        
        index_op, image_op, length_op, weeks_op = input_data.read_data_from_disk()
        
        #reshape inputs
        image_op = tf.reshape(image_op, [BATCH_SIZE, 224, 224, 3])
        weeks_op = tf.reshape(weeks_op, [BATCH_SIZE, 52])
        length_op = tf.reshape(length_op, [BATCH_SIZE, 1])
        
    
        
    # initialize the model depending on the selected mode
    if model == 'vgg':
        model = VGG16(mode = 'vgg', num_classes = 7, keep_prob = keep_prob)
        model.build(image, length = None, weeks = None)
    elif model == 'vgg_w_length':
        model = VGG16(mode='length', num_classes = 7, keep_prob = keep_prob)
        model.build(image, length = length, weeks = None)
    elif model == 'vgg_w_date':
        model = VGG16(mode = 'date', num_classes=7, keep_prob = keep_prob)
        model.build(image, length = None, weeks = weeks)
    else:
        model = VGG16(mode = 'all', num_classes=7, keep_prob = keep_prob)
        model.build(image, length = length, weeks = weeks)   
        
    # compute class probabilities by model output
    predictions = tf.nn.softmax(model.fc8)
    
    # initialize saver object to restore model weights
    saver = tf.train.Saver()
    
    init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    
    # start tensorflow session
    with tf.Session() as sess:
        
        sess.run(init_op)
        
        #start tensorflow queue coordinator
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord = coord)
        
        #restore model weights
        print("{} Restore model weights from checkpoint file.".format(datetime.now()))
        saver.restore(sess, checkpoint_path)
        print("{} Model restored. Start the evaluation process.".format(datetime.now()))
    
        for i in range(len(df)):
            
            #print status every 50 steps
            if i % 50 == 0:
                print("{} Evaluation status: {} of {} images processed.".format(datetime.now(), i, len(df)))
                
            # get one batch of data at the time
            idx, img, leng, wks = sess.run([index_op, image_op, length_op, weeks_op])
            
            # create dict for feeding into graph depending on selected model
            if model == 'vgg':
                d = {image: img}
            elif model == 'vgg_w_length':
                d = {image: img, length: leng}
            elif model == 'vgg_w_date':
                d = {image: img, weeks: wks}
            else:
                d = {image: img, length: leng, weeks: wks}
            
            # get prediction for the current sample
            y_test = sess.run(predictions, feed_dict=d)
            y_test = y_test.flatten()
            
            #store result summary in dataframe
            row = df.loc[idx]
            results.loc[i] = [row['image'], row['video_name'], row['label'], 
                np.argmax(y_test), y_test[0], y_test[1], y_test[2], y_test[3],
                y_test[4], y_test[5], y_test[6]]
    
        coord.request_stop()
        coord.join(threads)
        sess.close()
        
    print("{} Evaluation process finished.".format(datetime.now()))
        
    return results


def assign_avg_probs(grp):
    """
    this function computes the group prediction based on the max value of 
    the summed probabilities
    """
    grp['prediction_max_prob'] = int(np.argmax(grp[['score0', 'score1', 'score2', 
                           'score3', 'score4', 'score5', 'score6']].sum())[-1])
    return grp
  
    
def assign_mode(grp):
    """
    this function computes the group prediction based on the mode of all single
    predictions
    """
    grp['prediction_mode'] = mode(grp['pred']).mode[0]
    return grp

 
def main():
    
    # Get and parse input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('model', 
                        choices=['vgg', 'vgg_w_length', 'vgg_w_date', 'vgg_w_all'],
                        help='Select which type of model to use for evaluating the test set.')
    parser.add_argument('-m', '--mode', choices=['all', 'sample'], 
                        help='Chose wether or not to evaluate the entire or sample test set.',
                        default = 'sample')
    args = parser.parse_args()
        
    # Compute location of the data folder
    par_dir = os.path.abspath(os.path.join(sys.path[0], os.pardir))
    data_dir = os.path.join(par_dir, 'data')
    
    # Raise error if not found in the root of this project
    if not os.path.isdir(data_dir):
        raise ValueError('No data directory found at %s.' %par_dir)
        
    # Check wether or not to evaluate the entire test set
    if args.mode == 'all':
        test_file = os.path.join(data_dir, 'test.csv')
        
        if not os.path.isfile(test_file):
            raise ValueError("Couldn't find the test.csv file. Should be placed at %s" %data_dir)
            
    elif args.mode == 'sample':
        test_file = os.path.join(data_dir, 'test-sample.csv')
        
        if not os.path.isfile(test_file):
            raise ValueError("Couldn't find the test-sample.csv file. Should be placed at %s" %data_dir)
            
    # Compute path of model checkpoint. Should be in examples/checkpoint/'model'
    checkpoint_path = os.path.join(par_dir, 'examples', 'checkpoints', args.model)
    
    # Check if folder exists and contains the model checkpoint
    if os.path.isdir(checkpoint_path):
        files = [f for f in os.listdir(checkpoint_path)]
        
        if not 'model_epoch40.ckpt.index' in files:
            raise ValueError('Could not find checkpoint file for the %s model. Should be found at %s' %(args.model, checkpoint_path))
        else:
            checkpoint_path = os.path.join(checkpoint_path, 'model_epoch40.ckpt')
    
    
    # get the model predictions for each sample
    results = evaluate_model(args.model, checkpoint_path, data_dir, test_file)
    
    # reset tensorflows default graph, so that one can start the script again
    tf.reset_default_graph()
    
    # compute the per fish scores
    results = results.groupby(['video_name', 'label']).apply(assign_avg_probs)
    results = results.groupby(['video_name', 'label']).apply(assign_mode)
    fish = results.groupby(['video_name','label']).first().reset_index()
    
    #compute & print accuracy of per-img and per-fish classification score
    acc_img = accuracy_score(results['label'], results['pred'])
    print("{} Accuracy score based on per-image classification: {:.4f}".format(datetime.now(), acc_img))
    
    acc_fish_mode = accuracy_score(fish['label'], fish['prediction_mode'])
    print("{} Accuracy score based on per-fish classification (via mode): {:.4f}".format(datetime.now(), acc_fish_mode))
    
    acc_fish_max_prob = accuracy_score(fish['label'], fish['prediction_max_prob'])
    print("{} Accuracy score based on per-fish classification (via max probability): {:.4f}".format(datetime.now(), acc_fish_max_prob))
    
    #save dataframe as csv file
    filename = os.path.join(os.getcwd(), args.model + '_results.csv')
    results.to_csv(filename, sep=';', index=False)
    print("{} Saved results table at: {}".format(datetime.now(), filename))
    
    return results

if __name__ == '__main__':
    main()
