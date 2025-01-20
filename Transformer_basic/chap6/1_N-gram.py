import nltk


def ngrams(sentence, n):
    words = sentence.split()
    ngrams = zip(*[words[i:] for i in range(n)])

    return list(ngrams)


sentence = "안녕하세요 만나서 진심으로 반가워요"
unigram = ngrams(sentence, 1)
bigram = ngrams(sentence, 2)
trigram = ngrams(sentence, 3)

print(unigram)
print(bigram)
print(trigram)

# nltk라이브러리와 간단한 함수는 같은 결과를 낸다
unigram = nltk.ngrams(sentence.split(), 1)
bigram = nltk.ngrams(sentence.split(), 2)
trigram = nltk.ngrams(sentence.split(), 3)

print([*unigram])
print([*bigram])
print([*trigram])