import json
import os


def find_male_speakers(root_dir):
    """
    주어진 디렉토리에서 'speaker'-'sex' 값이 "남성"인 화자를 찾아 리스트로 반환합니다.

    Args:
        root_dir: 검색 시작 디렉토리의 절대 경로

    Returns:
        list: "남성"인 화자 이름의 리스트
    """

    male_speakers = []

    for speaker_dir in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_dir)
        if os.path.isdir(speaker_path):
            for utterance_dir in os.listdir(speaker_path):
                utterance_path = os.path.join(speaker_path, utterance_dir)
                if os.path.isdir(utterance_path):
                    for file in os.listdir(utterance_path):
                        if file.endswith(".json"):
                            json_path = os.path.join(utterance_path, file)
                            with open(json_path, 'r') as f:
                                data = json.load(f)
                                if data['speaker']['sex'] == "남성":
                                    if speaker_dir not in male_speakers:
                                        male_speakers.append(speaker_dir)
                                    break  # 한 화자당 한 개의 json 파일만 확인하므로 break

    return sorted(male_speakers)


# 예시 사용법
root_directory = ("/media/joy14479/A82E54D82E54A15C/Study_H/데이터셋/138.뉴스_대본_및_앵커_음성_데이터/01-1.정식개방데이터/Validation/02"
                  ".라벨링데이터/VL")
result = find_male_speakers(root_directory)
print(result)
