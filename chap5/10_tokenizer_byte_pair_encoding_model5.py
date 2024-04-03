from sentencepiece import SentencePieceProcessor

# 어휘 사전 불러오기
tokenizer = SentencePieceProcessor()
tokenizer.Load('petition_bpe.model')

vocab = {idx: tokenizer.IdToPiece(idx) for idx in range(tokenizer.GetPieceSize())}
print(list(vocab.items())[:5])
print("vocab size :", len(vocab))


# GetPieceSize는 모델이 생성한 하위 단어의 개수
# IdToPiece는 정수값을 하위 단어로 변환
# <unk> 은 unknown, 즉 OOV
# <s>는 문장시작
# </s>는 문장의 끝
# 8000은 하위 단어의 수, 설정을 8000으로 했다