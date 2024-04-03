from sentencepiece import SentencePieceProcessor

# 모델 가져옴
tokenizer = SentencePieceProcessor()
tokenizer.Load("petition_bpe.model")  # 이거 대문자로 바뀐듯

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]

# 문장을 토큰화함
tokenized_sentence = tokenizer.EncodeAsPieces(sentence)  # 이것도 바뀐듯 함수명이
tokenized_sentences = tokenizer.EncodeAsPieces(sentences)
print("단일 문장 토큰화 :", tokenized_sentence)
print("여러 문장 토큰화 :", tokenized_sentences)

# 토큰을 정수로 인코딩
encoded_sentence = tokenizer.EncodeAsIds(sentence)
encoded_sentences = tokenizer.EncodeAsIds(sentences)
print("단일 문장 정수 인코딩 :", encoded_sentence)
print("여러 문장 정수 인코딩 :", encoded_sentences)

# 토크나이저 모델이나 자연어 처리 모델에서 나온 정수를 문자열 데이터로 변환
decode_ids = tokenizer.DecodeIds(encoded_sentences)
decode_pieces = tokenizer.DecodePieces(encoded_sentences)
print("정수 인코딩에서 문장 변환 :", decode_ids)
print("하위 단어 토큰에서 문장 변환 :", decode_pieces)
