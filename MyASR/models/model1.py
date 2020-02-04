"""
model1 使用的是mfcc_feature和汉字标签，仅用一个单层rnn网络(实验性)
"""
import tensorflow as tf
from data_reader import DataReader

class Model1:
    def __init__(self, train_handler):
        self.train_handler = train_handler
        self.batch_size = 16
        self.max_features = 510
        self.n_mfcc = 40
        self.n_layers = 2
        self.rnn_features = 256
        self.dict_len = len(train_handler.dict) + 1
        self.learning_rate = 1e-5

    def train_start(self):
        x = tf.placeholder(tf.float32, [self.batch_size, self.max_features, self.n_mfcc])
        y = tf.placeholder(tf.int32, [self.batch_size, None])
        label_length = tf.placeholder(tf.int32, [self.batch_size])

        cell = tf.nn.rnn_cell.GRUCell
        rnn_layer = tf.nn.rnn_cell.MultiRNNCell([cell(self.rnn_features) for _ in range(self.n_layers)])
        output, states = tf.nn.dynamic_rnn(rnn_layer, x, dtype=tf.float32)

        logits = tf.layers.dense(output, self.dict_len)

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
                if count % 50 == 0 and count != 0:
                    _loss = sess.run(loss, feed_dict={
                        x: x_this_batch,
                        y: y_this_batch,
                        label_length: label_length_this_batch
                    })
                    print(f'{count}个batch完成，loss={_loss}')


m = Model1(DataReader('../temp/train', '../temp/train/dict/dict.pkl', 'phone'))
m.train_start()
