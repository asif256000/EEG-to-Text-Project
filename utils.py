import os


def init_dirs(dirs):
    if not isinstance(dirs, list):
        dirs = [dirs]

    for dir in dirs:
        os.makedirs(dir, exist_ok=True)