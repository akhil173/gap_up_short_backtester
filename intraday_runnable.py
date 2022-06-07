# -*- coding: utf-8 -*-
"""
Created on Sat Sep  4 20:20:02 2021

@author: akhil
"""

import yfinance as yf
import datetime as dt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scalping_test import mom as scalp
from intraday_shorting_5min import mom as shorting
from tvDatafeed import TvDatafeed, Interval
import copy
import time
import openpyxl

sto = pd.read_excel('Kotak_intraday_without_margin.xlsx').iloc[:,0]
stocks = sto.tolist()
for i in range(len(stocks)):
    stocks[i] = stocks[i].rstrip()


sto2 = pd.read_csv('NSE_TV.csv').iloc[:,0]
stocks2 =[]
stocks2 = sto2.tolist()
stocks.remove('CADILAHC')

stocks = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']

tv = TvDatafeed(auto_login=(True))
a = dt.datetime(2022,4,1, 12, 0)
b = dt.datetime(2021, 12, 16, 12, 0)
ohlc_dict = {}
# begin = dt.datetime(2021, 1, 1)
# end = dt.datetime(2021,6,30)


ohlc_stocks = list(ohlc_dict.keys())
for ticker in stocks:
    if ticker in ohlc_stocks:
        print('abc')
        continue
    else:
        print(ticker)
        ohlc_dict[ticker] = tv.get_hist(ticker, 'NSE', Interval.in_5_minute, n_bars=5000)

        


nse = tv.get_hist('NIFTY', 'NSE', Interval.in_5_minute, n_bars=5000)
a = time.time()
ohlc_daily = {}
ohlc_stocks = list(ohlc_daily.keys())
for ticker in stocks:
    if ticker in ohlc_stocks:
        print('abc')
        continue
    else:
        print(ticker)
        ohlc_daily[ticker] = tv.get_hist(ticker, 'NSE', Interval.in_1_hour, n_bars=5000)
e = time.time()-a
# nifty = copy.deepcopy(nse)
# nifty.drop(['symbol'], axis=1, inplace=True)
# nifty.columns = ['Open','High','Low','Close','Volume']

ohlc_data=copy.deepcopy(ohlc_dict)

for ticker in stocks:
    # ohlc_data[ticker].set_index('datetime', inplace=True)
    # ohlc_data[ticker] = ohlc_data[ticker].loc[a:]
    ohlc_data[ticker].drop(['symbol'], axis=1, inplace=True)
    ohlc_data[ticker].columns = ['Open','High','Low','Close','Volume']
    # ohlc_data[ticker]['Return'] = ohlc_data[ticker]['Close'].pct_change()
# stocks = ohlc_dict.keys()

ohlc_data_daily = copy.deepcopy(ohlc_daily)
for ticker in stocks:
    # ohlc_data_daily[ticker].set_index('datetime', inplace=True)
    # ohlc_data_daily[ticker] = ohlc_data_daily[ticker].loc[a:]
    ohlc_data_daily[ticker].drop(['symbol'], axis=1, inplace=True)
    ohlc_data_daily[ticker].columns = ['Open','High','Low','Close','Volume']
    # for index, row in ohlc_data_daily[ticker].iterrows():
    #     if index.time() > dt.time(15,30):
    #         ohlc_data_daily[ticker].drop(index, inplace=True)


ohlc_data_daily[ticker]['High'].diff()

momentum = short_hour.strat(ohlc_data_daily, stocks2, 15, 0, sl=1.01, target=0.99, tgt = 0.97)
mom2 = short_hour.strat(ohlc_data_daily, stocks2, 15, 0, sl=1.01, target=0.994, tgt = 0.994)
mom1sl = short_hour.strat(ohlc_data_daily, stocks2, 15, 0, sl=1.01, target=0.985, tgt = 0.97)

with pd.ExcelWriter('Chirag.xlsx') as writer:
    for ticker in stocks:
        momentum['buy_sell'][ticker].to_excel(writer, sheet_name=ticker)


test = {}
stocks = list(mom2['buy_sell'].keys())
for ticker in stocks:
    print(ticker)
    test[ticker] = []
    for i in range(len(mom2['buy_sell'][ticker])):
        val = float(mom2['buy_sell'][ticker]['Return'][i])
        if  val == 0.01901:
            test[ticker].append(mom2['buy_sell'][ticker].index[i].date())


mom2pm = shorting.strat(ohlc_data, stocks2, 15, 0, sl=1.01, tgt=0.98)
mom_long = shorting.strat(ohlc_data, stocks, 15, 0)

mom_t4 = scalp.strat(ohlc_data, stocks2, 15, 15, sl=1.005, target=0.99, tgt=0.99)

(1+momentum['strategy_df']['Return']).cumprod()


momentum['buy_sell']['BANKNIFTY']['P&L'].cumsum()

# abc = [1,2,3,4,5,6]
# abc.remove(abc[-1])

len(momentum['strategy_df'])

# beg1 = mean_reversion['strategy_df'].index[0] - dt.timedelta(1)
# end1 = mean_reversion['strategy_df'].index[-1]

momentum['strategy_df']['Return'].fillna(0, inplace = True)

# t = dt.datetime(2021,8,27,12,15)
# rel = tv.get_hist('RELIANCE','NSE',Interval.in_1_minute, n_bars=5000)
# rel.loc[t]

ret_stock_nifty = []
stocks_nifty_returns = []

nse = tv.get_hist('NIFTY', 'NSE', Interval.in_daily, n_bars= 5000)
# nifty=nse.loc[mom2['strategy_df'].index[0]:mom2['strategy_df'].index[-1]]
nifty=nse.loc[momentum['strategy_df']['Return'].index[0]:]
nifty.drop(['symbol'], axis=1, inplace=True)
nifty.columns = ['Open','High','Low','Close','Volume']
nifty['Return'] = nifty['Close'].pct_change()

nifty_return = ((1+nifty['Return']).cumprod()[-1])-1

# car = momentum['KPI']['CAGR']/momentum['KPI']['max_dd']
# ((1+momentum['buy_sell']['EICHERMOT.NS']['Return']).cumprod()[-1])*100
# vizualization of strategy return vs  nifty
fig, ax = plt.subplots()
plt.plot((1+mom_long['strategy_df']['Return']).cumprod(), color='green')
# plt.plot((1+nifty['Return']).cumprod(), color='black')
plt.plot((1+nifty['Return']).cumprod(), color='red')
# plt.plot((1+nifty['Return']).cumprod(), color='green', )
# plt.plot((1+nifty["Return"][1:]).cumprod())
plt.title("Index Return vs Strategy Return")
plt.ylabel("cumutive return")
plt.xlabel("months")
ax.legend(['1.5 %', "1 %"])


momentum['ind_stock_returns'][momentum['ind_stock_returns'][0]<0].sum()
momentum['ind_stock_returns'][momentum['ind_stock_returns'][0]>0].sum()

# mom2['ind_stock_returns'][mom2['ind_stock_returns'][0]<0].sum()
# mom2['ind_stock_returns'][mom2['ind_stock_returns'][0]>0].sum()