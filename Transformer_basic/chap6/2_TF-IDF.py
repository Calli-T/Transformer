from sklearn.feature_extraction.text import TfidfVectorizer

corpus2 = ["이건 세상에서 제일 비싼 단독 공연",
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

# corpus = ["Fly me to the moon", "And let me play among the stars", "Let me see what is like a Jupiter and Mars"]

corpus = [
    "That movie is famous movie",
    "I like that actor",
    "I don’t like that actor"
]


tfidf_vectorizer = TfidfVectorizer()
tfidf_vectorizer.fit(corpus)
tfidf_matrix = tfidf_vectorizer.transform(corpus)

print(tfidf_matrix.toarray())
print(tfidf_vectorizer.vocabulary_)

'''
sklearn.feature_extraction.text.TfidfVectorizer클래스의 매개변수에
input은 입력될 데이터의 형태, 일반적으로 content이고 이는 문자열 혹은 바이트
encoding은 문자그대로
lowercase는 boolean값을 받고, 이름 그대로 소문자 변환 여부
stop_wards는 사전에 추가하지 않을 단어, None이 기본인듯
ngram_range는 (최솟값, 최댓값)튜플로 받으며 사용할 N-gram의 범위, (1,2)와같이하면 유니그램과 바이그램을 사용
max_df는 최댓값 문서 빈도는 일정횟수 이상 등장하는 단어를 불용어 처리, 정수로 입력하면 초과 커트라인을 그걸로 하고 1이하 실수는 비율로 자른다
min_dfs는 최솟값 문서 빈도이며 최댓값과 똑같이 입력, 미만으로 커트라인 처리
단어사전은 None가능하며 미리 구축한 단어사전을 활용하도록함, 입력x의 경우 TF-IDF학습시 자동처리
IDF 분모 처리는 분모에 1을 더하는지 여부인듯
'''
