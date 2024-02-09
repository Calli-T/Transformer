import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
# ------아니 내 주피터가!------------------------------------------------------------------
from collections import Counter  # 파이썬 표준 라이브러리, stack queue 말고도 많이 있나보다

# ------------------------------------------------------------------------------------
import torch
from torch.utils.data import TensorDataset, DataLoader
from torch_directml import device

# ------------------------------------------------------------------------------------
from torch import optim
from torch import nn

# ------------------------------------------------------------------------------------
import numpy as np
from numpy.linalg import norm

# ------------------------------------------------------------------------------------

corpus = Korpora.load("nsmc")  # 네이버 영화 리뷰 데이터
corpus = pd.DataFrame(corpus.test)  # 테스트 세트를 불러옴

# 토크나이징
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(tokens[:3])


# ----단어사전 구축-------------------------------------------------------------------
def build_vocab(_corpus, n_vocab, special_tokens):
    # 단어 등장 횟수 세기
    counter = Counter()
    for _tokens in _corpus:
        counter.update(_tokens)
    _vocab = special_tokens  # 특수 토큰을 받고
    # 많이 등장한 토큰과 그 횟수를 _vocab에 추가
    for token, count in counter.most_common(n_vocab):
        _vocab.append(token)
    return _vocab


vocab = build_vocab(_corpus=tokens, n_vocab=5000, special_tokens=["<unk>"])
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print(vocab[:10])
print(len(vocab))


# ------윈도우로 단어쌍 뽑기-------------------------------------------------------------------

def get_word_pairs(_tokens, window_size):
    pairs = []  # 단어쌍
    for sentence in _tokens:  # 문장별로 토큰화가 진행 다된게 sentence, 위에서 Okt로 작업했다
        sentence_length = len(sentence)
        for idx, center_word in enumerate(sentence):  # 인덱스랑 원소랑 동시에 가져옴, 각 문장의 모든 토큰을 한 번씩 중심단어로 쓰는듯
            window_start = max(0, idx - window_size)
            window_end = min(sentence_length, idx + window_size + 1)
            center_word = sentence[idx]  # 중심단어
            context_words = sentence[window_start:idx] + sentence[idx + 1:window_end]  # 주변단어

            for context_word in context_words:  # 중심단어와 윈도우 내부의 모든 주변단어를 하나의 쌍으로 묶어서 추가
                pairs.append([center_word, context_word])

    return pairs


word_pairs = get_word_pairs(tokens, window_size=2)
print(word_pairs[:5])


# -------단어쌍을 인덱스쌍으로-------------------------------------------------------------------

def get_index_pairs(_word_pairs, _token_to_id):
    pairs = []
    unk_index = token_to_id["<unk>"]
    for word_pair in _word_pairs:
        center_word, context_word = word_pair
        center_index = _token_to_id.get(center_word, unk_index)
        context_index = _token_to_id.get(context_word, unk_index)
        pairs.append([center_index, context_index])  # dict에 없으면 <unk> 반환
    return pairs


index_pairs = get_index_pairs(word_pairs, token_to_id)
'''
[[token_to_id.get(word_pair[0], token_to_id["<unk>"]), token_to_id.get(word_pair[1], token_to_id["<unk>"])] for
    word_pair in word_pairs]
'''
print(index_pairs[:5])

# ------데이터로더 선언----------------------------------------------------------------
dml = device()
# 인덱스 텐서로 만들고, 중심단어와 주변단어의 인덱스들을 각각 열벡터로 가져옴
index_pairs = torch.tensor(index_pairs, device=dml)
center_indexs = index_pairs[:, 0]
context_indexs = index_pairs[:, 1]

# 그것들을 데이터셋에 두고, 데이터로더 설정
dataset = TensorDataset(center_indexs, context_indexs)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# --------모델------------------------------------------------------------------

class VanillaSkipgram(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            device=dml
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
            device=dml
        )

    def forward(self, input_ids):
        embeddings = self.embedding(input_ids)
        output = self.linear(embeddings)
        return output


# 스킵그램 클래스 선언 밑 손실함수와 최적화기법 선정
word2vec = VanillaSkipgram(vocab_size=len(token_to_id), embedding_dim=128)
criterion = nn.CrossEntropyLoss().to(dml)
optimizer = optim.SGD(word2vec.parameters(), lr=0.1)

# --------학습------------------------------------------------------------------

for epoch in range(10):
    cost = 0.0
    for input_ids, target_ids in dataloader:  # 데이터로더에서 지정한대로 한 쌍씩 꺼내먹기
        input_ids = input_ids.to(dml)
        target_ids = target_ids.to(dml)

        logits = word2vec(input_ids)  # 스킵그램에서 룩업을 사용한 예측값
        loss = criterion(logits, target_ids)  # 실측/예측을 손실함수에 넣어 오차값 계산

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        cost += loss

    cost = cost / len(dataloader)  # 평균 비용(손실)
    print(f"Epoch : {epoch + 1:4d}, Cost : {cost:.3f}")

# --------임베딩 값 추출, V*E나 E*V 행렬 둘중 하나의 행렬을 선택가능하다고 한다??------------------------------------------------------------------

token_to_embedding = dict()
embedding_matrix = word2vec.embedding.weight.detach().cpu().numpy()

# 단어와 임베딩을 딕셔너리로
for word, embedding in zip(vocab, embedding_matrix):
    token_to_embedding[word] = embedding

# 예시로 하나 가져오는듯
index = 30
token = vocab[30]
token_embedding = token_to_embedding[token]
print(token)
print(token_embedding)


# -------코사인 유사도------------------------------------------------------------------
def cosine_similarity(a, b):
    return np.dot(b, a) / (norm(b, axis=1) * norm(a)) # 임베딩 행렬은 5001*128, axis는 https://www.kdnuggets.com/2023/05/vector-matrix-norms-numpy-linalg-norm.html


def top_n_index(_cosine_matrix, n):
    closest_indexes = _cosine_matrix.argsort()[::-1]
    top_n = closest_indexes[1:n + 1]  # 자기 자신은 코사인 유사도 1나올테니 빼고 처리하는듯?

    return top_n


cosine_matrix = cosine_similarity(token_embedding, embedding_matrix)
top_n = top_n_index(cosine_matrix, n=5)

print(f"{token}와 가장 유사한 5개 단어")
for index in top_n:
    print(f"{id_to_token[index]} - 유사도 : {cosine_matrix[index]:.4f}")
