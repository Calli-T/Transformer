import pandas as pd
from Korpora import Korpora

corpus = Korpora.load("nsmc")
corpus_df = pd.DataFrame(corpus.test)

train = corpus_df.sample(frac=0.9, random_state=42)
test = corpus_df.drop(train.index)

print(train.head(5).to_markdown()) # to_markdown은 dependency문제가 생긴다
print("Training Data Size: ", len(train))
print("Tesing Data Size: ", len(test))