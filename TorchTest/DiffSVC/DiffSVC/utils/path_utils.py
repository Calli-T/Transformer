import os


def rel2abs(rel_path):
    # return os.path.join(os.path.dirname(__file__), rel_path
    return os.path.join(os.getcwd(), rel_path)


def dir2list(raw_path):
    abs_raw_path = rel2abs(raw_path)
    name_list = os.listdir(abs_raw_path)
    return [os.path.join(abs_raw_path, name) for name in name_list]
