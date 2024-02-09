import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt
# ------아니 내 주피터가!------------------------------------------------------------------
from collections import Counter  # 파이썬 표준 라이브러리, stack queue 말고도 많이 있나보다

# ------------------------------------------------------------------------------------
corpus = Korpora.load("nsmc")  # 네이버 영화 리뷰 데이터
corpus = pd.DataFrame(corpus.test)  # 테스트 세트를 불러옴

# 토크나이징
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(tokens[:3])


# ------------------------------------------------------------------------------------
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

# ------------------------------------------------------------------------------------

