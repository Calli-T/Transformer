

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