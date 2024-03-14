import spacy

nlp = spacy.load("ko_core_news_sm")
sentence = '태초에 하나님께서 하늘과 땅을 창조하셨습니다.'
doc = nlp(sentence)

for token in doc:
    print(f"[{token.pos_:5} - {token.tag_:3}] : {token.text}")
