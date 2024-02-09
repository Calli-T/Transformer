import pandas as pd
from Korpora import Korpora
from konlpy.tag import Okt

corpus = Korpora.load("nsmc")  # 네이버 영화 리뷰 데이터
corpus = pd.DataFrame(corpus.test) # 테스트 세트를 불러옴

# 토크나이징
tokenizer = Okt()
tokens = [tokenizer.morphs(review) for review in corpus.text]
print(tokens[:3])