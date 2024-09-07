from math import log10 as loten
from math import sqrt


# A_weighting(f) = 20 log10(1 + (12994/f)^4) + 20 log10(1 + (6300/f)^2) - 20 log10(1 + (f/20)) - 20 log10(1 + (f/25))
def aweighting(freq):
    return 2.00 + 20 * loten((12200 ** 2 * freq ** 4) / ((freq ** 2 + 20.6 ** 2) * (freq ** 2 + 12200 ** 2) * sqrt(
        (freq ** 2 + 107.7 ** 2) * (freq ** 2 + 737.9 ** 2))))


print(aweighting(100000))
# 저거 오버플로우 안나나?
