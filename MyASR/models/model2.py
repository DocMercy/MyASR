"""
model2使用mfcc作为输入，汉字作为标签，使用WaveNet
"""

import tensorflow as tf
from data_reader import DataReader


class Model2:
    def __init__(self, train_handler):
        self.train_handler = train_handler
        self.batch_size = 16
        self.max_features = 510
        self.n_mfcc = 40
        self.learning_rate = 1e-4
        self.dict_size = len(train_handler.dict) + 1
    
    def residual_block(self, input_tensor):
        conv_filter = tf.nn.tanh(input_tensor)
        conv_gate = tf.nn.sigmoid(input_tensor)
        out = conv_filter * conv_gate
        return out + input_tensor, out
    
    def block(self, input, filters):
        net1 = tf.layers.conv1d(input, filters, [3], padding='same', dilation_rate=1)
        net1, out1 = self.residual_block(net1)
        net2 = tf.layers.conv1d(net1, filters, [3], padding='same', dilation_rate=2)
        net2, out2 = self.residual_block(net2)
        net3 = tf.layers.conv1d(net2, filters, [3], padding='same', dilation_rate=4)
        net3, out3 = self.residual_block(net3)
        net4 = tf.layers.conv1d(net3, filters, [3], padding='same', dilation_rate=6)
        net4, out4 = self.residual_block(net4)
        net5 = tf.layers.conv1d(net4, filters, [3], padding='same', dilation_rate=16)
        net5, out5 = self.residual_block(net5)
        
        return out1 + out2 + out3 + out4 + out5
    
    def start_training(self):
        x = tf.placeholder(tf.float32, [self.batch_size, self.max_features, self.n_mfcc])
        y = tf.placeholder(tf.int32, [self.batch_size, None])
        label_length = tf.placeholder(tf.int32, [self.batch_size])
        
        net1 = tf.layers.conv1d(x, 128, [3], strides=1, padding='same', activation=tf.nn.sigmoid)
        bn = tf.layers.batch_normalization(net1)
        net2 = self.block(bn, 128)
        bn = tf.layers.batch_normalization(net2)
        net3 = self.block(bn, 128)
        
        logits = tf.layers.dense(net3, self.dict_size)
        logits_len = [logits.get_shape()[1].value] * self.batch_size
        loss = tf.reduce_mean(tf.nn.ctc_loss_v2(y, logits, label_length, logits_len, logits_time_major=False))
        
        op = tf.train.AdamOptimizer(self.learning_rate)
        train_step = op.minimize(loss)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            count = 0
            while True:
                x_this_batch, y_this_batch, label_length_this_batch = self.train_handler.sample_x_y(self.batch_size)
                train_step.run(feed_dict={
                    x: x_this_batch,
                    y: y_this_batch,
                    label_length: label_length_this_batch
                })
                count += 1
                if count % 5 == 0 and count != 0:
                    _loss = sess.run(loss, feed_dict={
                        x: x_this_batch,
                        y: y_this_batch,
                        label_length: label_length_this_batch
                    })
                    print(f'{count}个batch完成，loss={_loss}')


if __name__ == '__main__':
    m = Model2(DataReader('../temp/train', '../temp/train/dict/dict.pkl', 'phone'))
    m.start_training()
