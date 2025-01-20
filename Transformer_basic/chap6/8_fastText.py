# korNLI 데이터셋 전처리
from Korpora import Korpora

corpus = Korpora.load("kornli")
corpus_texts = corpus.get_all_texts() + corpus.get_all_pairs()
tokens = [sentence.split() for sentence in corpus_texts]

print(tokens[:3])

# ----------------------------fastText 모델 생성, 저장

from gensim.models import FastText

fastText = FastText(
    sentences=tokens,
    vector_size=128,
    window=5,
    min_count=5,
    sg=1,
    epochs=3,
    min_n=2,
    max_n=6
)

fastText.save("./models/fastText.model")
