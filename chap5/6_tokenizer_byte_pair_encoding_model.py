from Korpora import Korpora

corpus = Korpora.load("korean_petitions")
dataset = corpus.train
petition = dataset[0]

print("청원 시작일 : ", petition.begin)
print("청원 종료일 : ", petition.end)
print("청원 동의 수 : ", petition.num_agree)
print("청원 범주 : ", petition.category)
print("청원 제목 : ", petition.title)
print("청원 본문 : ", petition.text[:30])
