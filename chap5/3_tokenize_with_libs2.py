from konlpy.tag import Kkma

Kkma = Kkma()

#sentence = '이 문장은 거짓이다'
sentence = '무엇이든 상상할 수 있는 사람은 무엇이든 만들어 낼 수 있다.'

nouns = Kkma.nouns(sentence)
sentences = Kkma.sentences(sentence)
morphs = Kkma.morphs(sentence)
pos = Kkma.pos(sentence)

print("명사 추출 : ", nouns)
print("문장 추출 : ", sentences)
print("형태소 추출 : ", morphs)
print("품사 태깅 : ", pos)
