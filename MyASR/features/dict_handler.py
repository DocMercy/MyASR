import os
import pickle
import librosa
import numpy as np
from utils import path_check


class DictHandler:
    def __init__(self, root_path, label_path):
        self.path = label_path
        self.root_path = root_path
        self.dict = {' ': 0, '<UNK>': 1}
        self.count = 2

    def add_to_dict(self, data):
        """
        传入需要是str或者list
        """
        for i in data:
            if self.dict.get(i) is None:
                self.dict[i] = self.count
                self.count += 1

    def build_dict(self):
        for root, dirs, files in os.walk(self.path):
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    for item in pickle.load(a):
                        self.add_to_dict(item)
        print('字典构建完成')
        dict_path = os.path.join(self.root_path, 'dict')
        path_check(dict_path)
        with open(os.path.join(dict_path, 'dict.pkl'), 'wb') as a:
            pickle.dump(self.dict, a)
        print('字典保存完成')


if __name__ == '__main__':
    d = DictHandler('../temp/train', '../temp/train/label_raw/chara')
    d.build_dict()

    d = DictHandler('../temp/train', '../temp/train/label_raw/phone')
    d.build_dict()
