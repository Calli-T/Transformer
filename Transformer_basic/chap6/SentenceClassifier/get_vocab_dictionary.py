from konlpy.tag import Okt
from collections import Counter

from korpora_dataset import getDataset, getCorpus


def build_vocab(corpus, n_vocab, special_tokens):
    counter = Counter()
    for tokens in corpus:
        counter.update(tokens)
    _vocab = special_tokens

    for token, count in counter.most_common(n_vocab):
        _vocab.append(token)

    return _vocab


train, test = getDataset()

# 토큰화
tokenizer = Okt()
train_tokens = [tokenizer.morphs(review) for review in train.text]
test_tokens = [tokenizer.morphs(review) for review in test.text]

# 어휘 사전 구축
vocab = build_vocab(corpus=train_tokens, n_vocab=5000, special_tokens=["<pad>", "<unk>"])  # <pad>는 문장 길이 마추는 패딩용 토큰
token_to_id = {token: idx for idx, token in enumerate(vocab)}
id_to_token = {idx: token for idx, token in enumerate(vocab)}

print(vocab[: 10])
print(len(vocab))


def getVocab():
    return vocab


def getVocabTable():
    return token_to_id, id_to_token


def getToken():
    return train_tokens, test_tokens
