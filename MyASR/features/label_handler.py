import os
import pickle
from utils import path_check


class LabelHandler:
    def __init__(self, root_path, dict_path, label_type):
        self.root_path = root_path
        self.dict = self.get_dict(dict_path)
        self.max_len = 0
        self.label_type = label_type

    @staticmethod
    def get_dict(dict_path):
        with open(dict_path, 'rb') as a:
            dict = pickle.load(a)
        return dict

    def label_to_id(self, data):
        result = []
        len_result = []
        for item in data:
            one_item_result = []
            for i in item:
                if self.dict.get(i) is not None:
                    one_item_result.append(self.dict[i])
                else:
                    one_item_result.append(self.dict['<UNK>'])
            result.append(one_item_result)
            len_result.append(len(one_item_result))
            if max(len_result) > self.max_len:
                self.max_len = max(len_result)
        return result, len_result

    def pad_label(self, data, max_len):
        result = []
        for item in data:
            diff = max_len - len(item)
            item.extend([0] * diff)
            result.append(item)
        return result

    def start_handling(self):
        handled_path = os.path.join(self.root_path, 'label_handled', self.label_type)
        y_path = os.path.join(self.root_path, 'y_handled', self.label_type)
        path_check(handled_path)
        path_check(y_path)

        count = 0
        for root, dirs, files in os.walk(os.path.join(self.root_path, 'label_raw', self.label_type)):
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    label_data, label_len = self.label_to_id(pickle.load(a))
                with open(os.path.join(handled_path, f'label_handled_{count}.pkl'), 'wb') as a:
                    pickle.dump(label_data, a)
                with open(os.path.join(y_path, f'label_len_{count}.pkl'), 'wb') as a:
                    pickle.dump(label_len, a)
                count += 1
        print('标签转换完成')

    def start_padding(self, max_len):
        handled_path = os.path.join(self.root_path, 'label_handled', self.label_type)
        y_path = os.path.join(self.root_path, 'y_handled', self.label_type)
        count = 0
        for root, dirs, files in os.walk(handled_path):
            for file in files:
                with open(os.path.join(root, file), 'rb') as a:
                    label_data = self.pad_label(pickle.load(a), max_len)
                with open(os.path.join(y_path, f'y_handled_{count}.pkl'), 'wb') as a:
                    pickle.dump(label_data, a)
                count += 1
        print('标签补零完成')


if __name__ == '__main__':
    train = LabelHandler('../temp/train', '../temp/train/dict/dict.pkl', 'chara')
    test = LabelHandler('../temp/test', '../temp/train/dict/dict.pkl', 'chara')
    train.start_handling()
    test.start_handling()
    train.start_padding(max(train.max_len, test.max_len))
    test.start_padding(max(train.max_len, test.max_len))

    # train = LabelHandler('../temp/train', '../temp/train/dict/dict.pkl', 'phone')
    # test = LabelHandler('../temp/test', '../temp/train/dict/dict.pkl', 'phone')
    # train.start_handling()
    # test.start_handling()
    # train.start_padding(max(train.max_len, test.max_len))
    # test.start_padding(max(train.max_len, test.max_len))
