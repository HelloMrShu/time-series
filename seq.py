import numpy as np
import matplotlib.pyplot as plt

# 设置随机数种子以便结果可复现
np.random.seed(0)

# 生成长度为1000的白噪声序列
Z = np.random.normal(size=2000)

# 初始化MA模型的参数
theta = [0.6, 0.4]

# 初始化X
X = np.zeros_like(Z)

# 使用MA(2)模型生成X
for t in range(2, len(Z)):
    X[t] = Z[t] + theta[0]*Z[t-1] + theta[1]*Z[t-2]

# 绘制模拟的时间序列数据
plt.figure(figsize=(14, 6))
plt.plot(X)
plt.title("A Simulated MA(2) Time Series")
plt.show()
