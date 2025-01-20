from jamo import h2j, j2hcj

review = '현실과 구분 불가능한 cg. 시각적 즐거움은 최고! 더불어 ost는 더더욱 최고!!'
decomposed = j2hcj(h2j(review))
tokenized = list(decomposed)
print(tokenized)
