import os
import pickle
import numpy as np


class DataReader:
    def __init__(self, root_path):
        self.root_path = root_path
        self.x_paths, self.y_paths = self.get_x_y_paths()
        self.dict, self.reverse_dict = self.read_dict()
    
    def read_dict(self):
        dict_path = os.path.join(self.root_path, 'dict', 'dict.pkl')
        with open(dict_path, 'rb') as a:
            dict = pickle.load(a)
        dict[' '] = 0
        reverse_dict = {}
        for i, j in dict.items():
            reverse_dict[j] = i
        return dict, reverse_dict
    
    def get_x_y_paths(self):
        x_result = []
        y_result = []
        x_root = os.path.join(self.root_path, 'x_handled')
        y_root = os.path.join(self.root_path, 'label')
        for root, dirs, files in os.walk(x_root):
            for file in files:
                x_result.append(os.path.join(root, file))
        for root, dirs, files in os.walk(y_root):
            for file in files:
                y_result.append(os.path.join(root, file))
        return x_result, y_result
    
    def get_label_length(self, labels, label_pad_len):
        label_lengths = []
        y = []
        
        for item in labels:
            label_lengths.append(len(item))
            result = []
            diff = label_pad_len - len(item)
            for i in item:
                result.append(self.dict[i])
            result.extend(['0'] * diff)
            y.append(result)
        return y, label_lengths
    
    def sample_x_y(self, batch_size, label_pad_len):
        rand_file_num = np.random.randint(len(self.x_paths))
        with open(self.x_paths[rand_file_num], 'rb') as a:
            x_data = pickle.load(a)
        with open(self.y_paths[rand_file_num], 'rb') as b:
            y_data = pickle.load(b)
        
        rand_batch_num = np.random.randint(0, len(x_data) - batch_size)
        x_this_batch = x_data[rand_batch_num:rand_batch_num + batch_size]
        y_this_batch = y_data[rand_batch_num:rand_batch_num + batch_size]
        y_this_batch, label_lengths = self.get_label_length(y_this_batch, label_pad_len)
        return x_this_batch, y_this_batch, label_lengths


if __name__ == '__main__':
    d = DataReader('../temp/test')
    print(d.dict)
    print(d.sample_x_y(64)[:5])
