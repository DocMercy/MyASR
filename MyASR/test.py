import pickle

with open('temp/train/x_handled/x_handled_0.pkl', 'rb') as a:
    print('x_train的前五条数据为：')
    print(pickle.load(a)[:5])
with open('temp/test/x_handled/x_handled_0.pkl', 'rb') as a:
    print('x_test的前五条数据为：')
    print(pickle.load(a)[:5])
with open('temp/train/y_handled/chara/y_handled_0.pkl', 'rb') as a:
    print('y_train的前五条数据为：')
    print(pickle.load(a)[:5])
with open('temp/test/y_handled/chara/y_handled_0.pkl', 'rb') as a:
    print('y_test的前五条数据为：')
    print(pickle.load(a)[:5])
with open('temp/train/dict/dict.pkl', 'rb') as a:
    print('字典为：')
    print(pickle.load(a))

