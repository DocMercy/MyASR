import os
import pickle
import librosa
from utils import path_check


class Thchs30AudioReader:
    def __init__(self, root_path, target_path, sr):
        self.root_path = root_path
        self.target_path = target_path
        self.paths = self.get_paths()
        self.sr = sr

    def get_paths(self):
        x_result = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if '.scp' not in file:
                    if '.trn' not in file:
                        x_result.append(os.path.join(root, file))
        return x_result

    def read_audio(self, file_size):
        target_dir = os.path.join(self.target_path, 'audio_raw')
        path_check(target_dir)
        result = []
        count = 0
        print(f'开始读取{len(self.paths)}条音频')
        for i in range(len(self.paths)):
            wave_data, sr = librosa.load(self.paths[i], sr=self.sr)
            result.append(wave_data)
            if ((i + 1) % file_size == 0 and i != 0) or (i + 1) == len(self.paths):
                with open(os.path.join(target_dir, f'audio_raw_{count}.pkl'), 'wb') as a:
                    pickle.dump(result, a)
                result = []
                count += 1
                print(f'{i + 1}/{len(self.paths)}条音频处理完成')


class Thchs30LabelReader:
    def __init__(self, root_path, target_path, label_file_root):
        self.root_path = root_path
        self.target_path = target_path
        self.label_file_root = label_file_root
        self.paths = self.get_paths()

    def get_paths(self):
        result = []
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if '.scp' not in file:
                    if '.trn' in file:
                        result.append(os.path.join(self.label_file_root, file))
        return result

    def read_label(self, target_type, file_size):
        chara_target_path = os.path.join(self.target_path, 'label_raw', target_type)
        path_check(chara_target_path)

        result = []
        count = 0
        print(f'开始读取{len(self.paths)}条标签')

        for i in range(len(self.paths)):
            with open(self.paths[i], 'r', encoding='utf-8') as a:
                data = a.readlines()
                if target_type == 'chara':
                    result.append(data[0].replace('\n', '').replace(' ', ''))
                else:
                    result.append(data[1].replace('\n', '').split(' '))

            if ((i + 1) % file_size == 0 and i != 0) or (i + 1) == len(self.paths):
                with open(os.path.join(chara_target_path, f'label_raw_{count}.pkl'), 'wb') as a:
                    pickle.dump(result, a)
                result = []
                count += 1
                print(f'{i + 1}/{len(self.paths)}条标签处理完成')


if __name__ == '__main__':
    root_path = r"E:\Data\data_thchs30"
    file_size = 500
    # x_train = Thchs30AudioReader(os.path.join(root_path, 'train'), '../temp/train', 16000)
    # x_train.read_audio(file_size)
    # x_test = Thchs30AudioReader(os.path.join(root_path, 'test'), '../temp/test', 16000)
    # x_test.read_audio(file_size)
    y_train = Thchs30LabelReader(os.path.join(root_path, 'train'), '../temp/train', os.path.join(root_path, 'data'))
    # y_train.read_label('phone', file_size)
    y_train.read_label('chara', file_size)
    y_test = Thchs30LabelReader(os.path.join(root_path, 'test'), '../temp/test', os.path.join(root_path, 'data'))
    y_test.read_label('phone', file_size)
    y_test.read_label('chara', file_size)
