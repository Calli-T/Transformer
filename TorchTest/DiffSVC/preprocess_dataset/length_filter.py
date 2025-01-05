import os
import wave
import shutil


def filter_wav_files(directory_path):
    """
    주어진 디렉토리에서 14.8초 이상 15.2초 이하의 WAV 파일을 필터링하여 리스트로 반환합니다.

    Args:
      directory_path: 검색할 디렉토리의 절대 경로

    Returns:
      list: 조건에 맞는 WAV 파일의 절대 경로 리스트
    """

    filtered_files = []
    for root, _, files in os.walk(directory_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                with wave.open(file_path, 'rb') as wav_file:
                    # 채널 수, 샘플링 레이트, 바이트 깊이, 프레임 수 확인
                    params = wav_file.getparams()
                    nchannels, sampwidth, framerate, nframes = params[:4]
                    # 파일 길이 계산 (초)
                    duration = nframes / framerate
                    if 14.2 <= duration <= 15.5:
                        filtered_files.append(file_path)
    return filtered_files


# 예시 사용법
directory = "/mnt/additional/projects/Transformer/TorchTest/DiffSVC/DiffSVC/train_dataset/male_announcers/raw"
filtered_list = filter_wav_files(directory)
print(len(filtered_list))

unused_dir = "/mnt/additional/dataset/unused_male_announcer"
original_list = [os.path.join(directory, wav_name) for wav_name in os.listdir(directory)]

for original_wav in original_list:
    if original_wav not in filtered_list:
        shutil.move(original_wav, unused_dir)
