import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--project_name', default='Adele_augmented', type=str, help='')
arg_parsed = parser.parse_args()

project_root = os.sep.join(os.path.abspath(__file__).split(os.sep)[:-2])
os.chdir(project_root)

dir_path_list = []
dir_path_list.append(os.path.join(project_root, "models", "singer", arg_parsed.project_name))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "f0"))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "raw"))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "separated"))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "separated", "final"))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "separated", "norm"))
dir_path_list.append(os.path.join(project_root, "train_dataset", arg_parsed.project_name, "separated", "voice"))

for dir_path in dir_path_list:
    try:
        os.makedirs(dir_path)
        print(f"디렉터리 '{dir_path}'가 생성되었습니다.")
    except FileExistsError:
        print(f"디렉터리 '{dir_path}'가 이미 존재합니다.")

# python3 /mnt/additional/projects/Transformer/TorchTest/DiffSVC/DiffSVC/utils/gen_project_dirs.py --project_name=Hutao