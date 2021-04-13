import numpy as np
import pandas as pd
# 正常显示画图时出现的中文
from pylab import mpl
import matplotlib.pyplot as plt

# 这里使用微软雅黑字体
mpl.rcParams['font.sans-serif'] = ['SimHei']
# 画图时显示负号
mpl.rcParams['axes.unicode_minus'] = False
import seaborn as sns  # 画图用的
import tushare as ts

# Jupyter Notebook特有的magic命令
# 直接在行内显示图形
# %matplotlib inline

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
startTime = '2010-1-1'
sh = ts.get_k_data(code=code, ktype='D',
                   autype='qfq', start=startTime)
# code:股票代码，个股主要使用代码，如‘600000’
# ktype:'D':日数据；‘m’：月数据，‘Y’:年数据
# autype:复权选择，默认‘qfq’前复权
# start：起始时间
# end：默认当前时间
# 查看下数据前5行
sh.head(5)

# 将数据列表中的第0列'date'设置为索引
sh.index = pd.to_datetime(sh.date)
# 画出上证指数收盘价的走势
sh['close'].plot(figsize=(12, 6))
plt.title(stockName + startTime[:4] + '-2020年走势图')
plt.xlabel('日期')
plt.savefig(stockName + '长期走势图' + '.png', dpi=600)
plt.show()
