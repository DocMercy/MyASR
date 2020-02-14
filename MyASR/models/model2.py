"""
model2使用mfcc作为输入，汉字作为标签，使用WaveNet
"""

import tensorflow as tf
from data_reader import DataReader


class Model2:
    def __init__(self, train_handler, test_handler):
        self.train_handler = train_handler
        self.test_handler = test_handler
        self.batch_size = 24
        self.max_features = 510
        self.n_mfcc = 40
        self.learning_rate = 3e-5
        self.dict_size = len(train_handler.dict) + 1

    def residual_block(self, input_tensor):
        conv_filter = tf.nn.tanh(input_tensor)
        conv_gate = tf.nn.sigmoid(input_tensor)
        out = conv_filter * conv_gate
        return out + input_tensor, out

    def block(self, input, filters):
        net1 = tf.layers.conv1d(input, filters, [3], padding='same', dilation_rate=1)
        net1, out1 = self.residual_block(net1)
        bn = tf.layers.batch_normalization(net1)
        net2 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=2)
        net2, out2 = self.residual_block(net2)
        bn = tf.layers.batch_normalization(net2)
        net3 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=4)
        net3, out3 = self.residual_block(net3)
        bn = tf.layers.batch_normalization(net3)
        net4 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=8)
        net4, out4 = self.residual_block(net4)
        bn = tf.layers.batch_normalization(net4)
        net5 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=16)
        net5, out5 = self.residual_block(net5)
        bn = tf.layers.batch_normalization(net5)
        net6 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=32)
        net6, out6 = self.residual_block(net6)
        bn = tf.layers.batch_normalization(net6)
        net7 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=64)
        net7, out7 = self.residual_block(net7)
        bn = tf.layers.batch_normalization(net7)
        net8 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=128)
        net8, out8 = self.residual_block(net8)
        bn = tf.layers.batch_normalization(net8)
        net9 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=256)
        net9, out9 = self.residual_block(net9)
        bn = tf.layers.batch_normalization(net9)
        net10 = tf.layers.conv1d(bn, filters, [3], padding='same', dilation_rate=512)
        net10, out10 = self.residual_block(net10)

        return out1 + out2 + out3 + out4 + out5 + out6 + out7 + out8 + out9 + out10

    def start_training(self):
        x = tf.placeholder(tf.float32, [self.batch_size, self.max_features, self.n_mfcc])
        y = tf.placeholder(tf.int32, [self.batch_size, None])
        label_length = tf.placeholder(tf.int32, [self.batch_size])

        net1 = tf.layers.conv1d(x, 256, [3], strides=1, padding='same', activation=tf.nn.sigmoid)
        bn = tf.layers.batch_normalization(net1)
        net2 = self.block(bn, 256)
        bn = tf.layers.batch_normalization(net2)
        net3 = self.block(bn, 256)

        logits = tf.layers.dense(net3, self.dict_size)
        logits_len = [logits.get_shape()[1].value] * self.batch_size
        loss = tf.reduce_mean(tf.nn.ctc_loss_v2(y, logits, label_length, logits_len, logits_time_major=False))

        op = tf.train.AdamOptimizer(self.learning_rate)
        train_step = op.minimize(loss)
        reshaped_logits = tf.reshape(logits, [self.max_features, self.batch_size, self.dict_size])
        decoded_data = tf.nn.ctc_beam_search_decoder_v2(reshaped_logits, logits_len, beam_width=5)

        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, '../Model/model_2/')
            # sess.run(tf.global_variables_initializer())
            count = 0
            while True:
                x_this_batch, y_this_batch, label_length_this_batch = self.train_handler.sample_x_y(self.batch_size)
                train_step.run(feed_dict={
                    x: x_this_batch,
                    y: y_this_batch,
                    label_length: label_length_this_batch
                })
                count += 1
                if count % 100 == 0 and count != 0:
                    _loss, _decoded = sess.run((loss, decoded_data), feed_dict={
                        x: x_this_batch,
                        y: y_this_batch,
                        label_length: label_length_this_batch
                    })
                    _decoded_data, _decoded_prob = _decoded

                    print(f'{count}个batch完成，loss={_loss}')
                    print(f'解码出来的label是: {train_handler.decode(_decoded_data[0])[1]}')
                    saver.save(sess, '../Model/model_2/')
                    if _loss < 20:
                        break
                if count % 5000 == 0 and count != 0:
                    print('开始跑测试集')
                    test_acc = 0
                    for i in range(2400 // self.batch_size):
                        x_this_batch, y_this_batch, label_length_this_batch = self.test_handler.sample_x_y(
                            self.batch_size)
                        _decoded = sess.run(decoded_data, feed_dict={
                            x: x_this_batch,
                            y: y_this_batch,
                            label_length: label_length_this_batch
                        })
                        _decoded_data, _decoded_prob = _decoded
                        result_ids = test_handler.decode(_decoded_data[0])[0]
                        for i in y_this_batch:
                            while 0 in i:
                                i.remove(0)
                        batch_acc = 0
                        for i in range(self.batch_size):
                            one_sentence_acc = 0
                            for j in range(len(y_this_batch[i])):
                                if j < len(result_ids[i]):
                                    result_element = result_ids[i][j]
                                else:
                                    result_element = 0
                                if y_this_batch[i][j] == result_element:
                                    one_sentence_acc += 1
                            batch_acc += one_sentence_acc / len(y_this_batch[i])
                        test_acc += batch_acc / self.batch_size
                    print(f"测试集完成，准确率:{test_acc / (2400 // self.batch_size)}")

    def start_predicting(self, data):
        x = tf.placeholder(tf.float32, [1, self.max_features, self.n_mfcc])
        net1 = tf.layers.conv1d(x, 256, [3], strides=1, padding='same', activation=tf.nn.sigmoid)
        bn = tf.layers.batch_normalization(net1)
        net2 = self.block(bn, 256)
        bn = tf.layers.batch_normalization(net2)
        net3 = self.block(bn, 256)
        logits = tf.layers.dense(net3, self.dict_size)
        logits_len = [logits.get_shape()[1].value] * self.batch_size
        reshaped_logits = tf.reshape(logits, [self.max_features, self.batch_size, self.dict_size])
        decoded_data = tf.nn.ctc_beam_search_decoder_v2(reshaped_logits, logits_len, beam_width=5)
        with tf.Session() as sess:
            saver = tf.train.Saver(tf.global_variables())
            saver.restore(sess, '../Model/model_2/')
            # sess.run(tf.global_variables_initializer())
            _decoded = sess.run(decoded_data, feed_dict={
                x: data
            })
            _decoded_data, _decoded_prob = _decoded
            return test_handler.decode(_decoded_data[0])[1]

if __name__ == '__main__':
    train_handler = DataReader('../temp/train', '../temp/train/dict/dict.pkl', 'phone')
    test_handler = DataReader('../temp/test', '../temp/train/dict/dict.pkl', 'phone')
    m = Model2(train_handler, test_handler)
    m.start_training()
