from Korpora import Korpora

corpus = Korpora.load("korean_petitions")
petitions = corpus.get_all_texts()
with open('./corpus.txt', 'w', encoding='utf-8') as f:
    for petition in petitions:
        f.write(petition + '\n')
        
# 다 가져와서 txt 파일에 때려박는작업
# 너무커서 지워놨다, gitignore도 안씀