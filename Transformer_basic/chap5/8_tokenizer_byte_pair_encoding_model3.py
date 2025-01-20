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
    "--input=./corpus.txt \
    --model_prefix=petition_bpe\
    --vocab_size=8000 model_type=bpe"
)

'''
학습을 시키는 과정이고
끝나면 .model파일과 .vocab파일이 생성된다
각각 학습된 토크나이저와 어휘 사전이다
'''