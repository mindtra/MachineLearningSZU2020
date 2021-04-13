import random

import numpy as np
from time import time
# %matplotlib inline
import matplotlib.pyplot as plt
import tushare as ts
import pandas as pd

# r = random.random() * 0.1
# r = -r
r = (random.uniform(0, 2)-1)*0.1
S0 = 9.07  # 当前价格

code = '600073'
stockName = '上海梅林'
# code = '002567'
# stockName = '唐人神'
# code = '002100'
# stockName = '天康生物'
# code = '002548'
# stockName = '金新农'
# code = '600975'
# stockName = '新五丰'
# code = '002385'
# stockName = '大北农'
# code = '300498'
# stockName = '温氏股份'
# code = '002124'
# stockName = '天邦股份'
# code = '002157'
# stockName = '正邦科技'
# code = '000876'
# stockName = '新希望'
# code = '002714'
# stockName = '牧原股份'
# code = '603363'
# stockName = '傲农生物'
df = ts.get_k_data(code, ktype='D', autype='qfq',
                   start='2000-01-01')
df.index = pd.to_datetime(df.date)
tech_rets = df.close.pct_change()[1:]
rets = tech_rets.dropna()
# rets.head()
# 下面的结果说明，我们95%的置信，一天我们不会损失超过0.0264...
rets.quantile(0.11)
np.random.seed(2020)
# np.random.seed(2021)
# np.random.seed(2022)
t0 = time()
print(S0)
T = 1.0


print('r=', end='')
print(r)
sigma = rets.std()
# M = 1
# M = 65
M = 72
# M = 150
# M = 365
dt = T / M
I = 25000
S = np.zeros((M + 1, I))
S[0] = S0
for t in range(1, M + 1):
    z = np.random.standard_normal(I)
    S[t] = S[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
s_m = np.sum(S[-1]) / I
tnp1 = time() - t0
print(np.sum(S[1, 0:10]) / 10)
print('经过250000次模拟，得出' + str(M) + '天以后上证指数的预期平均收盘价为：%.2f' % s_m)
# 经过250000次模拟，得出1年以后上证指数的预期平均收盘价为：2776.85

plt.figure(figsize=(10, 6))
S = np.c_[np.zeros((S.shape[0])) + S0, S]
plt.plot(S[:, :11])
plt.grid(True)
# plt.title('猪肉价格蒙特卡洛模拟其中10条模拟路径图')
plt.title(stockName + '蒙特卡洛模拟其中10条模拟路径图')
plt.xlabel('时间')
plt.ylabel('指数')
plt.savefig(stockName + str(M) + '天预测走势图' + '.png', dpi=600)

plt.show()
