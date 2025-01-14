import os
from pydub import AudioSegment
from pydub.effects import normalize
import shutil


# 디렉터리 내부의 모든 파일을 normalize하고, dB 변화 리스트를 반환하는 함수
def normalize_files_and_report(path_for_normalize):
    dB_changes = []  # dB 변화 리스트 초기화
    fname_list = os.listdir(path_for_normalize)
    fname_list.sort()

    # path_a 디렉토리에서 모든 오디오 파일을 처리
    for filename in fname_list:
        if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
            file_path = os.path.join(path_for_normalize, filename)

            # 오디오 파일 로드
            audio = AudioSegment.from_file(file_path)

            # 정상화 전의 최고 음량 (dBFS)
            original_dBFS = audio.dBFS

            # normalize() 효과를 적용하여 음량을 정상화
            normalized_audio = normalize(audio)

            # 정상화 후의 최고 음량 (dBFS)
            normalized_dBFS = normalized_audio.dBFS

            # 음량 변화 계산 (정상화 전후의 dB 차이)
            dB_change = normalized_dBFS - original_dBFS

            # dB 변화 리스트에 추가
            dB_changes.append((filename, original_dBFS, normalized_dBFS, dB_change))

            # 정상화된 오디오 파일을 같은 파일명으로 덮어쓰기
            normalized_audio.export(file_path, format=filename.split('.')[-1])

    return dB_changes


# path_b와 dB 변화 리스트를 매개변수로 받아서 역정상화하는 함수
def denormalize_files(path_for_denormalize, dB_changes):
    fname_list = os.listdir(path_for_denormalize)
    fname_list.sort()

    # dB 변화 리스트를 딕셔너리로 변환 (파일명 -> dB 변화)
    dB_change_dict = {filename: dB_change for filename, _, _, dB_change in dB_changes}

    # path_b 디렉터리에서 모든 오디오 파일을 처리
    for filename in fname_list:
        if filename.endswith((".wav", ".mp3", ".flac", ".ogg")):
            file_path = os.path.join(path_for_denormalize, filename)

            # dB 변화량을 가져옴
            if filename in dB_change_dict:
                dB_change = dB_change_dict[filename]

                # 오디오 파일 로드
                audio = AudioSegment.from_file(file_path)

                # 역정상화 (dB 변화량을 반대로 적용)
                reversed_audio = audio - dB_change

                # 역정상화된 오디오 파일을 덮어씁니다.
                reversed_audio.export(file_path, format=filename.split('.')[-1])

                print(f"Reversed normalization for {filename}: {dB_change:.2f} dB")

# 정상화 끝난것은 출력 디렉터리로 치우자
def move_files(path_a, path_b):
    # path_a 디렉터리 내의 모든 파일을 path_b로 옮기기
    for filename in os.listdir(path_a):
        # 전체 경로를 만들어서 파일을 옮기기
        file_path_a = os.path.join(path_a, filename)
        file_path_b = os.path.join(path_b, filename)

        # 파일이 존재하는지 확인하고 이동
        if os.path.isfile(file_path_a):
            shutil.move(file_path_a, file_path_b)
            print(f"Moved: {filename}")
