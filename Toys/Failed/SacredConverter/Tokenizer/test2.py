import spacy

nlp = spacy.load("ko_core_news_sm")
sentence = '태초에 하나님이 천지를 창조하시니라.'
doc = nlp(sentence)

for token in doc:
    print(f"[{token.pos_:5} - {token.tag_:3}] : {token.text}")
