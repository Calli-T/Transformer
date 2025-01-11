import librosa, soundfile
import os


def pitch_shift(input_dir, output_dir, key_range):
    """
    음성 데이터 키 변조 증강 함수

    Args:
        input_dir (str): 원본 음성 파일 디렉토리
        output_dir (str): 변조된 음성 파일 저장 디렉토리
        key_range (int): 키 변조 범위
    """

    # 출력 디렉토리 생성
    os.makedirs(output_dir, exist_ok=True)

    # wav/flac 파일 목록 가져오기
    file_list = [f for f in os.listdir(input_dir) if f.endswith('.wav') or f.endswith('.flac')]

    for file in file_list:
        file_path = os.path.join(input_dir, file)
        file_name, ext = os.path.splitext(file)

        # 음성 파일 로드
        y, sr = librosa.load(file_path, sr=44100)

        # 원본 파일 저장 (origin)
        output_path = os.path.join(output_dir, f"{file_name}_origin{ext}")
        # librosa.output.write_wav(output_path, y, sr)
        soundfile.write(output_path, y, sr)

        # 키 변조 및 저장
        for i in range(-key_range, key_range + 1):
            if i == 0:
                continue  # 원본은 이미 저장했으므로 건너뛰기
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=i)
            output_path = os.path.join(output_dir, f"{file_name}_{'+' if i > 0 else '-'}{abs(i)}key{ext}")
            # librosa.output.write_wav(output_path, y_shifted, sr)
            soundfile.write(output_path, y_shifted, sr)


# 사용 예시
input_dir = "/home/joy14479/link/dataset/Adele_raw"
output_dir = "/home/joy14479/link/dataset/Adele_augmented"
key_range = 2

pitch_shift(input_dir, output_dir, key_range)
