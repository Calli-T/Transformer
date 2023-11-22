import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from matplotlib import pyplot as plt

man_height = stats.norm.rvs(loc=170, scale=10, size=500, random_state=1)
woman_height = stats.norm.rvs(loc=150, scale=10, size=500, random_state=1)

X = np.concatenate([man_height, woman_height])
Y = ["man"] * len(man_height) + ["woman"] * len(woman_height)

df = pd.DataFrame(list(zip(X, Y)), columns=["X", "Y"])
fig = sns.displot(data=df, x="X", hue="Y", kind="kde")
fig.set_axis_labels("cm", "count")
plt.show()

# 이하 비쌍체 t-검정
statistic, pvalue = stats.ttest_ind(man_height, woman_height, equal_var=True)

print("statistic:", statistic)
print("pvalue:", pvalue)
print("*:", pvalue < 0.05)
print("**:", pvalue < 0.001)

# 유의 확률이 낮고 통계량이 크면 귀무 가설이 참일 확률이 낮다.
# 귀무가설로 지정된건 남녀 키의 평균이 서로 같다
# 유의 확률이 0.001이므로 귀무가설을 폐기하고 사람의 키를성별을 구분하는데 유의미한 변수로 사용할 수 있다.

# 59p
