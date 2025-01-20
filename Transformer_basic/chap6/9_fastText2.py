from gensim.models import FastText

fastText = FastText.load('./models/fastText.model')

oov_token = "사랑해요"
oov_vector = fastText.wv[oov_token]

print(oov_token in fastText.wv.index_to_key)  # oov 토큰이 단어사전에 없다는걸 보여줌
print(fastText.wv.most_similar(oov_vector, topn=5))  # 하위단어로 topn 처리가능
