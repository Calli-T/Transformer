import spacy

nlp = spacy.load("en_core_web_sm")
sentence = 'Those who can imagine anything, can create the impossible'
doc = nlp(sentence)

for token in doc:
    print(f"[{token.pos_:5} - {token.tag_:3}] : {token.text}")
