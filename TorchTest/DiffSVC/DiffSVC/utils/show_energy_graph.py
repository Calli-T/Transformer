import os
import librosa
import numpy as np
import matplotlib.pyplot as plt


def hz_to_midi_note_name(hz):
    """
    Hz 값을 MIDI 노트와 음 이름으로 변환
    """
    if hz <= 0:  # 피치가 유효하지 않을 경우
        return None
    midi_note = int(round(69 + 12 * np.log2(hz / 440.0)))  # Hz를 MIDI 노트로 변환
    note_names = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
    octave = midi_note // 12 - 1
    note_name = note_names[midi_note % 12]
    return f"{note_name}{octave}", midi_note


def plot_pitch_distribution_with_notes(directory):
    # 피치 값을 저장할 리스트
    pitches = []

    # 디렉터리 내 모든 파일 순회
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.wav') or file.endswith('.flac'):
                filepath = os.path.join(root, file)
                print(filepath)

                try:
                    # 오디오 파일 로드
                    y, sr = librosa.load(filepath, sr=None)
                    # 피치 추출
                    pitches_file, _ = librosa.piptrack(y=y, sr=sr)
                    # 피치 값 중 0이 아닌 값만 추출
                    pitches += [p for p in pitches_file.flatten() if p > 0]
                except Exception as e:
                    print(f"Error processing {filepath}: {e}")

    # 최소/최대 피치 확인
    pitch_min = min(pitches)
    pitch_max = max(pitches)

    # 최소/최대 피치를 MIDI 노트와 음 이름으로 변환
    min_note_name, min_midi = hz_to_midi_note_name(pitch_min)
    max_note_name, max_midi = hz_to_midi_note_name(pitch_max)

    print(f"Minimum pitch: {pitch_min:.2f} Hz ({min_note_name}), Maximum pitch: {pitch_max:.2f} Hz ({max_note_name})")

    # x축 범위에 맞는 MIDI 노트를 생성
    midi_range = range(min_midi, max_midi + 1)

    # 범위 내 MIDI 노트 이름 생성
    note_names = [hz_to_midi_note_name(440.0 * 2 ** ((m - 69) / 12))[0] for m in midi_range]

    # 히스토그램 데이터 생성
    bins = np.linspace(pitch_min, pitch_max, len(midi_range) + 1)  # bins 길이를 midi_range 길이에 맞게 설정
    hist, bin_edges = np.histogram(pitches, bins=bins)

    # 그래프 생성
    plt.figure(figsize=(12, 6))
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2  # 구간 중앙값 계산
    plt.plot(bin_centers, hist, marker='o', linestyle='-', color='skyblue', label='Pitch Distribution')

    # x축 라벨을 MIDI 노트 이름으로 설정
    plt.xticks(bin_centers, note_names, rotation=45)

    # 축 라벨 및 제목 설정
    plt.xlabel("Pitch (Note)")
    plt.ylabel("Number of occurrences")
    plt.title("Pitch Distribution with Note Names")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    plt.tight_layout()
    plt.show()


# 사용 예시
directory = "/mnt/additional/projects/Transformer/TorchTest/DiffSVC/DiffSVC/train_dataset/Adele/separated/final"# "/home/joy14479/link/dataset/Adele_raw"  # 분석할 디렉터리 경로로 바꾸세요
plot_pitch_distribution_with_notes(directory)
