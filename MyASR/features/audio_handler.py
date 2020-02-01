import os
import pickle
import librosa
import numpy as np
from utils import path_check


class FeatureHandler:
    def __init__(self, root_path, sr, type, n_mfcc=40, n_fft=1024, hop_length=512):
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.sr = sr
        self.root_path = root_path
        self.max_length = 0
        self.type = type

    def get_mfcc(self, data):
        result = []
        for i in data:
            mfcc_data = librosa.feature.mfcc(i, sr=self.sr, n_mfcc=self.n_mfcc)
            if mfcc_data.shape[1] > self.max_length:
                self.max_length = mfcc_data.shape[1]
            result.append(mfcc_data)
        return result

    def pad_feature(self, data, max_length):
        result = []
        for i in data:
            diff = max_length - i.shape[1]
            if self.type == 'mfcc':
                result.append(np.concatenate([i, np.zeros([self.n_mfcc, diff])], axis=1).T)
            else:
                result.append(np.concatenate([i, np.zeros([128, diff])], axis=1).T)
        return np.array(result)

    def get_mel_spec(self, data):
        result = []
        for i in data:
            mel_spec_data = librosa.feature.melspectrogram(i, sr=self.sr, n_fft=self.n_fft, hop_length=self.hop_length)
            if mel_spec_data.shape[1] > self.max_length:
                self.max_length = mel_spec_data.shape[1]
            result.append(mel_spec_data)
        return result

    def start_feature(self):
        audio_path = os.path.join(self.root_path, 'audio_raw')
        save_path1 = os.path.join(self.root_path, 'audio_mfcc')
        save_path2 = os.path.join(self.root_path, 'x_handled')
        path_check(save_path1)
        path_check(save_path2)

        count = 0
        for root, dirs, files in os.walk(audio_path):
            print(f'开始处理{len(files)}个音频文件')
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    audio_data = pickle.load(a)
                    if self.type == 'mfcc':
                        handled_data = self.get_mfcc(audio_data)
                    else:
                        handled_data = self.get_mel_spec(audio_data)
                    with open(os.path.join(save_path1, f'audio_mfcc_{count}.pkl'), 'wb') as b:
                        pickle.dump(handled_data, b)
                count += 1
                print(f'第{count}个处理完成')

    def start_padding(self, max_length):
        save_path1 = os.path.join(self.root_path, 'audio_mfcc')
        save_path2 = os.path.join(self.root_path, 'x_handled')
        count = 0
        for root, dirs, files in os.walk(save_path1):
            print(f'开始对{len(files)}个音频文件补零')
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    data = pickle.load(a)
                    pad_data = self.pad_feature(data, max_length)
                    with open(os.path.join(save_path2, f'x_handled_{count}.pkl'), 'wb') as b:
                        pickle.dump(pad_data, b)
                count += 1
                print(f'第{count}个处理完成')


if __name__ == '__main__':
    m1 = FeatureHandler('../temp/test', 16000, 'mel_spec')
    m1.start_feature()
    m2 = FeatureHandler('../temp/train', 16000, 'mel_spec')
    m2.start_feature()
    print(f'test的最大长度{m1.max_length}')
    print(f'train的最大长度{m2.max_length}')
    m1.start_padding(max(m1.max_length, m2.max_length))
    m2.start_padding(max(m1.max_length, m2.max_length))
