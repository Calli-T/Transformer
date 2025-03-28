from tokenizers import Tokenizer
from tokenizers.decoders import WordPiece as WordPieceDecoder

tokenizer = Tokenizer.from_file("models/petition_wordpiece.json")
tokenizer.decoder = WordPieceDecoder()

sentence = "안녕하세요, 토크나이저가 잘 학습되었군요!"
# sentences = ["이렇게 입력값을 리스트로 받아서", "쉽게 토크나이저를 사용할 수 있답니다"]
sentences = ["이건 세상에서 제일 비싼 단독 공연",
             "가수는 나고 관객은 너 하나",
             "화려한 막이 이제 곧 올라가기 전에",
             "그저 몇 가지만 주의해줘요",
             "세상에서 제일 편한 옷을 갈아 입고",
             "제일 좋아하는 자리에 누워",
             "배터리가 바닥나지 않게 조심하고",
             "통화상태를 항상 유지해줘요",
             "듣고 싶은 노래를 말 만해 everything",
             "입이 심심할 때는 coffee, popcorn, anything",
             "너무 부담주진 말고 편하게 들어줘",
             "아님 내가 너무 떨리니까",
             "오직 너에게만 감동적인 노래",
             "오직 너를 웃게 하기 위한 코너",
             "네가 너무 설레 잠 못 들게 만들 거야",
             "지금이야 크게 소리 질러줘",
             "누구보다 특별한 너의 취향을 알아",
             "달콤한데 슬픈 듯 아찔하게 (맞지)",
             "근데 다음 곡이 중요해 볼륨 높여봐",
             "기억 나니 우리 그 날 그 노래",
             "내가 너무 진지해 보여도 웃지마",
             "누가 봐도 완벽한 노래는 아니지만",
             "많이 연습한 부분을 너 때문에 틀리잖아",
             "아직 나는 너무 떨리니까",
             "오직 너에게만 감동적인 노래",
             "오직 너를 웃게 하기 위한 코너",
             "네가 너무 설레 잠 못 들게 만들 거야",
             "지금이야 크게 소리 질러",
             "이 공연은 거의 다 끝나 가고 있어",
             "어땠는지 말해줘 문자로",
             "너무나 아쉽지만 졸린 거 이미 알고 있어",
             "기대해줘 마지막 곡 이 중에서도 제일",
             "감동적인 노래",
             "오직 너를 웃게 하기 위한 코너",
             "네가 너무 설레 잠 못 들게 만들 거야",
             "지금이야 제일 원하는 걸 말해 어떤 노래를",
             "다시 듣고 싶어? 사실 내가 원해",
             "네가 너무 설레 잠 못 들지 모르지만",
             "앵콜이야 크게 소리 질러줘",
             "이건 세상에서 제일 비싼 단독공연",
             "가수는 나고 관객은 너 하나"]

encoded_sentence = tokenizer.encode(sentence)
encoded_sentences = tokenizer.encode_batch(sentences)

print("인코더 형식:", type(encoded_sentence))

print("단일 문장 토큰화:", encoded_sentence.tokens)
print("여러 문장 토큰화:")  # , [enc.tokens for enc in encoded_sentences])
for enc in encoded_sentences:
    print(enc.tokens)

print("단일 문장 정수 인코딩:", encoded_sentence.ids)
print("여러 문장 정수 인코딩:", [enc.ids for enc in encoded_sentences])

print("정수 인코딩에서 문장 변환:", tokenizer.decode(encoded_sentence.ids))
