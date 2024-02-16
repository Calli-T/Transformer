import numpy as np
from get_vocab_dictionary import getVocabTable, getToken

token_to_id, id_to_token = getVocabTable()
train_tokens, test_tokens = getToken()


# 파이토치의 임베딩 층을 사용하기 위해 토큰을 정수로 변환
# 문장길이는 패딩으로 맞춰주는듯
def pad_sequences(sequences, max_length, pad_value):
    result = []
    for sequence in sequences:
        sequence = sequence[:max_length]
        pad_length = max_length - len(sequence)
        padded_sequence = sequence + [pad_value] * pad_length
        result.append(padded_sequence)

    return np.asarray(result)  # 넘파이 배열로 변환해서 바꿔줍니다


# 정수 인코딩
unk_id = token_to_id["<unk>"]  # OOV 처리를 위한 unknown토큰의 인덱스
train_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in train_tokens
]  # 토큰화된 학습 데이터셋의 각 문장 즉 리뷰를 가지고 토큰 하나하나를 정수로 임베딩, 테이블은 token_to_id에서 가져오며 없다면 <unk>의 인덱스 사용
test_ids = [
    [token_to_id.get(token, unk_id) for token in review] for review in test_tokens
]

# 패딩
max_length = 32
pad_id = token_to_id["<pad>"]  # 패딩의 정수 인코딩 인덱스
train_ids = pad_sequences(train_ids, max_length, pad_id)
test_ids = pad_sequences(test_ids, max_length, pad_id)


# 정수 인코딩 및 패딩된 리뷰를 가져옴
def getIds():
    return train_ids, test_ids


print(train_ids[0])
print(test_ids[0])
