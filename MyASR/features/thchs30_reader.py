import os
import pickle
import librosa
from utils import path_check


class Thchs30Reader:
    def __init__(self, root_path, target_path):
        self.root_path = root_path
        self.target_path = target_path
        self.x_train_path, self.y_train_path = self.get_paths(os.path.join(root_path, 'train'))
        self.x_test_path, self.y_test_path = self.get_paths(os.path.join(root_path, 'test'))
        self.sr = 0
    
    def get_paths(self, path):
        x_result = []
        y_result = []
        for root, dirs, files in os.walk(path):
            for file in files:
                if '.scp' not in file:
                    if '.trn' in file:
                        y_result.append(os.path.join(self.root_path, 'data', file))
                    else:
                        x_result.append(os.path.join(root, file))
        return x_result, y_result
    
    def read_audio(self, paths, target_dir):
        target_dir = os.path.join(target_dir, 'audio_raw')
        path_check(target_dir)
        result = []
        count = 0
        print(f'开始读取{len(paths)}条音频')
        for i in range(len(paths)):
            wave_data, sr = librosa.load(paths[i], sr=16000)
            if self.sr == 0:
                self.sr = sr
            elif self.sr != sr:
                raise Exception(f'sr not match,file is {paths[i]}')
            result.append(wave_data)
            if ((i + 1) % 1000 == 0 and i != 0) or (i + 1) == len(paths):
                with open(os.path.join(target_dir, f'wave_data_{count}.pkl'), 'wb') as a:
                    pickle.dump(result, a)
                result = []
                count += 1
                print(f'{i + 1}/{len(paths)}条音频处理完成')
    
    def read_label(self, paths, target_dir, target_type):
        chara_target_path = os.path.join(target_dir, 'label')
        path_check(chara_target_path)
        
        result = []
        count = 0
        print(f'开始读取{len(paths)}条标签')
        
        for i in range(len(paths)):
            with open(paths[i], 'r', encoding='utf-8') as a:
                data = a.readlines()
                if target_type == 'chara':
                    result.append(data[0].replace('\n', '').replace(' ', ''))
                else:
                    result.append(data[1].replace('\n', '').replace(' ', ''))
            
            if ((i + 1) % 1000 == 0 and i != 0) or (i + 1) == len(paths):
                with open(os.path.join(chara_target_path, f'label_{count}.pkl'), 'wb') as a:
                    pickle.dump(result, a)
                result = []
                count += 1
                print(f'{i + 1}/{len(paths)}条标签处理完成')
    
    def start_reading(self, type):
        train_save_path = os.path.join(self.target_path, 'train')
        test_save_path = os.path.join(self.target_path, 'test')
        path_check(train_save_path)
        path_check(test_save_path)
        
        self.read_audio(self.x_train_path, train_save_path)
        self.read_audio(self.x_test_path, test_save_path)
        
        self.read_label(self.y_train_path, train_save_path, type)
        self.read_label(self.y_test_path, test_save_path, type)
        
        return self.sr


if __name__ == '__main__':
    t = Thchs30Reader(r'E:\Data\data_thchs30', '../temp')
    sr = t.start_reading('chara')
    print(f'音频采样率：{sr}')
