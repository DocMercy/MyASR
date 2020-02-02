import os
import pickle
import numpy as np


class DataReader:
    def __init__(self, root_path, dict_path, label_type):
        self.root_path = root_path
        self.label_type = label_type
        self.x_paths, self.y_paths, self.y_lens = self.get_x_y_paths()
        self.dict, self.reverse_dict = self.read_dict(dict_path)

    @staticmethod
    def read_dict(dict_path):
        with open(dict_path, 'rb') as a:
            dict = pickle.load(a)
        reverse_dict = {}
        for i, j in dict.items():
            reverse_dict[j] = i
        return dict, reverse_dict

    def get_x_y_paths(self):
        x_result = []
        y_result = []
        y_len = []
        x_root = os.path.join(self.root_path, 'x_handled')
        y_root = os.path.join(self.root_path, 'y_handled', self.label_type)
        for root, dirs, files in os.walk(x_root):
            for file in files:
                x_result.append(os.path.join(root, file))
        for root, dirs, files in os.walk(y_root):
            for file in files:
                if 'len' in file:
                    y_len.append(os.path.join(root, file))
                else:
                    y_result.append(os.path.join(root, file))
        return x_result, y_result, y_len

    def sample_x_y(self, batch_size):
        rand_file_num = np.random.randint(len(self.x_paths))
        with open(self.x_paths[rand_file_num], 'rb') as a:
            x_data = pickle.load(a)
        with open(self.y_paths[rand_file_num], 'rb') as b:
            y_data = pickle.load(b)
        with open(self.y_lens[rand_file_num], 'rb') as b:
            y_len = pickle.load(b)

        rand_batch_num = np.random.randint(0, len(x_data) - batch_size)
        x_this_batch = x_data[rand_batch_num:rand_batch_num + batch_size]
        y_this_batch = y_data[rand_batch_num:rand_batch_num + batch_size]
        label_lengths = y_len[rand_batch_num:rand_batch_num + batch_size]
        return x_this_batch, y_this_batch, label_lengths


if __name__ == '__main__':
    d = DataReader('../temp/train', '../temp/train/dict/dict.pkl', 'phone')
    print(d.dict)
    print(d.sample_x_y(64)[:5])
