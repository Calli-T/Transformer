import os
import shutil

root_directory = ("/media/joy14479/A82E54D82E54A15C/Study_H/데이터셋/138.뉴스_대본_및_앵커_음성_데이터/01-1.정식개방데이터/Validation/01"
                  ".원천데이터/VS")

male_announcer_list = ['SPK008', 'SPK009', 'SPK011', 'SPK013', 'SPK019', 'SPK021', 'SPK023', 'SPK024', 'SPK027',
                       'SPK029', 'SPK030', 'SPK031', 'SPK034', 'SPK036', 'SPK038', 'SPK050', 'SPK051', 'SPK052',
                       'SPK054', 'SPK055', 'SPK056', 'SPK057', 'SPK058', 'SPK059', 'SPK060', 'SPK061', 'SPK064',
                       'SPK073', 'SPK074', 'SPK077', 'SPK078', 'SPK079', 'SPK080', 'SPK081', 'SPK082', 'SPK083',
                       'SPK084', 'SPK085', 'SPK086', 'SPK087', 'SPK088', 'SPK089']
root_directory = [os.path.join(root_directory, ann) for ann in male_announcer_list]
'''for i  in root_directory[:5]:
    print(len(os.listdir(i)))'''


def copy_wav_files(source_dirs, target_dir):
    """
    주어진 소스 디렉토리들에서 .wav 파일을 찾아 목표 디렉토리로 복사합니다.

    Args:
      source_dirs: 소스 디렉토리들의 절대 경로 리스트
      target_dir: 목표 디렉토리의 절대 경로
    """

    for source_dir in source_dirs:
        for root, _, files in os.walk(source_dir):
            for file in files:
                if file.endswith('.wav'):
                    source_file = os.path.join(root, file)
                    destination_file = os.path.join(target_dir, file)
                    shutil.copy2(source_file, destination_file)


train_dataset_path = "/mnt/additional/projects/Transformer/TorchTest/DiffSVC/DiffSVC/train_dataset/male_announcers/raw"
copy_wav_files(root_directory, train_dataset_path)