def normalization(arr):
    tempSum = 0

    for i in arr:
        tempSum += i

    arithMean = tempSum / len(arr)
    tempSum = 0

    for i in arr:
        tempSum = tempSum + (i - arithMean) ** 2

    var = tempSum / len(arr)
    tempStant = 10 ** (-5)

    return [((x - arithMean) / (var + tempStant) ** 0.5) for x in arr]

print(normalization([-0.6577, -0.5797, 0.6360]))

# 이건 감마 1 베타 0인 배치 정규화
