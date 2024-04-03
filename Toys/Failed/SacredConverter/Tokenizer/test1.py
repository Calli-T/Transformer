from sentencepiece import SentencePieceTrainer

'''
문자를 통해 인자를 전달받음
그 내용은 input이 경로
model_prefix는 모델이름
vocab_size는 어휘 사전 크기
model_type은 토크나이저 알고리즘
그 외에는 254p
'''
SentencePieceTrainer.Train(
    "--input=./genesis.txt \
    --model_prefix=petition_bpe\
    --vocab_size=8000 model_type=bpe"
)

'''
학습을 시키는 과정이고
끝나면 .model파일과 .vocab파일이 생성된다
각각 학습된 토크나이저와 어휘 사전이다
'''

from sentencepiece import SentencePieceProcessor

# 어휘 사전 불러오기
tokenizer = SentencePieceProcessor()
tokenizer.Load('petition_bpe.model')

vocab = {idx: tokenizer.IdToPiece(idx) for idx in range(tokenizer.GetPieceSize())}
for i in range(100):
    print(list(vocab.items())[10 * i:10 * i+10])
print("vocab size :", len(vocab))


# GetPieceSize는 모델이 생성한 하위 단어의 개수
# IdToPiece는 정수값을 하위 단어로 변환
# <unk> 은 unknown, 즉 OOV
# <s>는 문장시작
# </s>는 문장의 끝
# 8000은 하위 단어의 수, 설정을 8000으로 했다