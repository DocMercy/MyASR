import os


def path_check(path):
    if not os.path.exists(path):
        os.makedirs(path)
