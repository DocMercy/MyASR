import os
import pickle
import librosa
import numpy as np
from utils import path_check


class MFCCHandler:
    def __init__(self, root_path, sr, n_mfcc):
        self.n_mfcc = n_mfcc
        self.sr = sr
        self.root_path = root_path
        self.mfcc_max_length = 0

    def get_mfcc(self, data):
        result = []
        for i in data:
            mfcc_data = librosa.feature.mfcc(i, sr=self.sr, n_mfcc=self.n_mfcc)
            if mfcc_data.shape[1] > self.mfcc_max_length:
                self.mfcc_max_length = mfcc_data.shape[1]
            result.append(mfcc_data)
        return result

    def pad_mfcc(self, data, max_length):
        result = []
        for i in data:
            diff = max_length - i.shape[1]
            result.append(np.concatenate([i, np.zeros([self.n_mfcc, diff])], axis=1).T)
        return np.array(result)

    def start_mfcc(self):
        audio_path = os.path.join(self.root_path, 'audio_raw')
        save_path1 = os.path.join(self.root_path, 'mfcc')
        save_path2 = os.path.join(self.root_path, 'x_handled')
        path_check(save_path1)
        path_check(save_path2)

        count = 0
        for root, dirs, files in os.walk(audio_path):
            print(f'开始处理{len(files)}个原始音频文件')
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    audio_data = pickle.load(a)
                    mfcc_data = self.get_mfcc(audio_data)
                    with open(os.path.join(save_path1, f'mfcc_data_{count}.pkl'), 'wb') as b:
                        pickle.dump(mfcc_data, b)
                count += 1
                print(f'第{count}个处理完成')

    def start_padding(self, max_length):
        save_path1 = os.path.join(self.root_path, 'mfcc')
        save_path2 = os.path.join(self.root_path, 'x_handled')
        count = 0
        for root, dirs, files in os.walk(save_path1):
            print(f'开始对{len(files)}个mfcc文件补零')
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    mfcc_data = pickle.load(a)
                    pad_data = self.pad_mfcc(mfcc_data, max_length)
                    with open(os.path.join(save_path2, f'padded_{count}.pkl'), 'wb') as b:
                        pickle.dump(pad_data, b)
                count += 1
                print(f'第{count}个处理完成')


if __name__ == '__main__':
    m1 = MFCCHandler('../temp/test', 16000, 40)
    m1.start_mfcc()
    m2 = MFCCHandler('../temp/train', 16000, 40)
    m2.start_mfcc()
    print(f'test的最大长度{m1.mfcc_max_length}')
    print(f'train的最大长度{m2.mfcc_max_length}')
    m1.start_padding(max(m1.mfcc_max_length, m2.mfcc_max_length))
    m2.start_padding(max(m1.mfcc_max_length, m2.mfcc_max_length))
