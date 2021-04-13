import numpy as np
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# 正常显示画图时出现的中文
from pylab import mpl

# 这里使用微软雅黑字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 画图时显示负号
mpl.rcParams['axes.unicode_minus'] = False
import seaborn as sns  # 画图用的
import tushare as ts


# Jupyter Notebook特有的magic命令
# 直接在行内显示图形
# %matplotlib inline

df = ts.get_k_data('sh', ktype='D', autype='qfq',
                   start='2005-01-01')
df.index = pd.to_datetime(df.date)
tech_rets = df.close.pct_change()[1:]
rets = tech_rets.dropna()
# rets.head()
# 下面的结果说明，我们95%的置信，一天我们不会损失超过0.0264...
rets.quantile(0.05)
def monte_carlo(start_price, days, mu, sigma):
    dt = 1 / days
    price = np.zeros(days)
    price[0] = start_price
    shock = np.zeros(days)
    drift = np.zeros(days)

    for x in range(1, days):
        shock[x] = np.random.normal(loc=mu * dt,
                                    scale=sigma * np.sqrt(dt))
        drift[x] = mu * dt
        price[x] = price[x - 1] + (price[x - 1] *
                                   (drift[x] + shock[x]))
    return price


# 模拟次数
runs = 10000
start_price = 2641.34  # 今日收盘价
days = 252
mu = rets.mean()
sigma = rets.std()
simulations = np.zeros(runs)

for run in range(runs):
    simulations[run] = monte_carlo(start_price,
                                   days, mu, sigma)[days - 1]
q = np.percentile(simulations, 1)
plt.figure(figsize=(8, 6))
plt.hist(simulations, bins=50, color='grey')
plt.figtext(0.6, 0.8, s="初始价格: %.2f" % start_price)
plt.figtext(0.6, 0.7, "预期价格均值: %.2f" % simulations.mean())
plt.figtext(0.15, 0.6, "q(0.99: %.2f)" % q)
plt.axvline(x=q, linewidth=6, color="r")
plt.title("经过 %s 天后上证指数模拟价格分布图" % days, weight="bold")
# Text(0.5, 1, '经过 252 天后上证指数模拟价格分布图')
