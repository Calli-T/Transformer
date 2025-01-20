from gensim.models import Word2Vec

word2vec = Word2Vec.load('./models/word2vec.model')

word = "연기"
print(word2vec.wv[word])
print(word2vec.wv.most_similar(word, topn=5))
print(word2vec.wv.similarity(w1=word, w2='연기력'))
