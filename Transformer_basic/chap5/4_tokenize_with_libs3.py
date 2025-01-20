import nltk

# 모델들 다운로드
# nltk.download("punkt")
# nltk.download("averaged_perceptron_tagger")

from nltk import tokenize
from nltk import tag

sentence = 'Those who can imagine anything, can create the impossible'

word_tokens = tokenize.word_tokenize(sentence)
sent_tokens = tokenize.sent_tokenize(sentence)
pos = tag.pos_tag(word_tokens)

print(word_tokens)
print(sent_tokens)
print(pos)
