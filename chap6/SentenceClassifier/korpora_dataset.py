import pandas as pd
from Korpora import Korpora

# 데이터셋 가져오기
corpus = Korpora.load("nsmc")
corpus_df = pd.DataFrame(corpus.test)

train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)


def showExample():
    print(train.head(5).to_markdown())  # to_markdown은 dependency문제가 생긴다 -> 해결함 tabulate 설치
    print("Training Data Size: ", len(train))
    print("Tesing Data Size: ", len(test))


def getCorpus():
    return corpus, corpus_df


def getDataset():
    return train, test
