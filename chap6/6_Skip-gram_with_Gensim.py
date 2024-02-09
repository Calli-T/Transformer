import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
from collections import Counter  # 파이썬 표준 라이브러리, stack queue 말고도 많이 있나보다
from gensim.models import Word2Vec


corpus = Korpora.load("nsmc")  # 네이버 영화 리뷰 데이터
corpus = pd.DataFrame(corpus.test)  # 테스트 세트를 불러옴

# 토크나이징
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
# print(tokens[:3])


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

'''
print(vocab[:10])
print(len(vocab))

'''

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
# print(word_pairs[:5])


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
# print(index_pairs[:5])

# ------데이터로더 선언----------------------------------------------------------------

word2vec = Word2Vec(sentences=tokens, vector_size=128, window=5, min_count=1, sg=1, epochs=3, max_final_vocab=10000)
word2vec.save('./models/word2vec.model')
