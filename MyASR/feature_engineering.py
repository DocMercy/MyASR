import os
from features.thchs30_reader import *
from features.audio_handler import *
from features.label_handler import *
from features.dict_handler import *


def start_reading_audio(data_path, file_size):
    x_train = Thchs30AudioReader(os.path.join(data_path, 'train'), 'temp/train', 16000)
    x_train.read_audio(file_size)
    x_test = Thchs30AudioReader(os.path.join(data_path, 'test'), 'temp/test', 16000)
    x_test.read_audio(file_size)
    y_train = Thchs30LabelReader(os.path.join(data_path, 'train'), 'temp/train', os.path.join(data_path, 'data'))
    y_train.read_label('phone', file_size)
    y_train.read_label('chara', file_size)
    y_test = Thchs30LabelReader(os.path.join(data_path, 'test'), 'temp/test', os.path.join(data_path, 'data'))
    y_test.read_label('phone', file_size)
    y_test.read_label('chara', file_size)


def start_feature(feature_type):
    m1 = FeatureHandler('temp/test', 16000, feature_type)
    m1.start_feature()
    m2 = FeatureHandler('temp/train', 16000, feature_type)
    m2.start_feature()
    print(f'test的最大长度{m1.max_length}')
    print(f'train的最大长度{m2.max_length}')
    m1.start_padding(max(m1.max_length, m2.max_length))
    m2.start_padding(max(m1.max_length, m2.max_length))


def start_label_handle(label_type):
    train = LabelHandler('temp/train', 'temp/train/dict/dict.pkl', label_type)
    test = LabelHandler('temp/test', 'temp/train/dict/dict.pkl', label_type)
    train.start_handling()
    test.start_handling()
    train.start_padding(max(train.max_len, test.max_len))
    test.start_padding(max(train.max_len, test.max_len))


def start_buiding_dict():
    d = DictHandler('temp/train', 'temp/train/label_raw/chara')
    d.build_dict()
    d = DictHandler('temp/train', 'temp/train/label_raw/phone')
    d.build_dict()


if __name__ == '__main__':
    # 数据文件夹所在的位置
    data_path = r"E:\Data\data_thchs30"
    # 分块文件的大小
    file_size = 500
    # 特征类型：mel_spec:梅尔频谱|mfcc: mfcc
    feature_type = 'mfcc'
    # 标签类型：chara:汉字|phone:拼音
    label_type = 'chara'
    
    start_reading_audio(data_path, file_size)
    start_feature(feature_type)
    start_buiding_dict()
    start_label_handle(label_type)
