# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller

# 创建一个函数来检查数据的平稳性
def test_stationarity(timeseries):
    # 执行Dickey-Fuller测试
    print('Results of Dickey-Fuller Test:')
    dftest = adfuller(timeseries, autolag='AIC')
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical Value (%s)' % key] = value
    print(dfoutput)

# 生成不平稳的时间序列
np.random.seed(0)
n = 100
x = np.cumsum(np.random.randn(n))

# 把它转换成Pandas的DataFrame格式
df = pd.DataFrame(x, columns=['value'])

# 检查原始数据的平稳性
test_stationarity(df['value'])

# 进行一阶差分
df['first_difference'] = df['value'] - df['value'].shift(1)

# 检查一阶差分后的数据的平稳性
test_stationarity(df['first_difference'].dropna())

# 进行二阶差分
df['second_difference'] = df['first_difference'] - df['first_difference'].shift(1)

# 检查二阶差分后的数据的平稳性
test_stationarity(df['second_difference'].dropna())

# 可视化原始数据和差分后的数据
plt.figure(figsize=(12, 6))
plt.plot(df['value'], label='Original')
plt.plot(df['first_difference'], label='1st Order Difference')
plt.plot(df['second_difference'], label='2nd Order Difference')
plt.legend(loc='best')
plt.title('Original and Differenced Time Series')
plt.show()
