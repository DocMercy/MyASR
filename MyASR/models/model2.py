"""
model2使用mfcc作为输入，汉字作为标签，使用WaveNet
"""

import tensorflow as tf
from data_reader import DataReader


class Model2:
    def __init__(self):
        self.batch_size = 16
        self.max_features = 510
        self.n_mfcc = 40
        self.learning_rate = 1e-4
    
    def layer(self, input, filters, stride):
        conv1 = tf.layers.conv1d(input, filters, [3], stride=stride, padding='same', activation=tf.nn.relu)
        conv2 = tf.layers.conv1d(input, filters, [3], stride=stride, padding='same', activation=tf.nn.sigmoid)
        return conv1 * conv2 + input
    
    def start_training(self):
        x = tf.placeholder(tf.float32, [self.batch_size, self.max_features, self.n_mfcc])
        y = tf.placeholder(tf.int32, [self.batch_size, None])
        label_length = tf.placeholder(tf.int32, [self.batch_size])
        
        net = self.layer(x, 32, 1)
        net = self.layer(net, 32, 2)
