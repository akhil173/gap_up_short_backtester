# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 12:18:47 2021

@author: akhil
"""
import numpy as np
import pandas as pd
from alpha_vantage.timeseries import TimeSeries
import copy
import time
import yfinance as yf
import datetime as dt
import statsmodels.api as sm
import matplotlib.pyplot as plt
import talib as ta


def ATR(DF,n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2['ATR']

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["cum_return"] = (1 + df["Return"]).cumprod()
    n = len(df)/(252*6)
    CAGR = (df["cum_return"].tolist()[-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    vol = df["Return"].std() * np.sqrt(252*6)
    return vol

def sharpe(DF,rf):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr
    

def max_dd(DF):
    "function to calculate max drawdown"
    df = DF.copy()
    df["cum_return"] = (1 + df["Return"]).cumprod()
    df["cum_roll_max"] = df["cum_return"].cummax()
    df["drawdown"] = df["cum_roll_max"] - df["cum_return"]
    df["drawdown_pct"] = df["drawdown"]/df["cum_roll_max"]
    max_dd = df["drawdown_pct"].max()
    return max_dd

def MACD(ohlv, tp_short, tp_long, tp_macd):
    df = ohlv.copy()
    macd=pd.DataFrame()
    macd['MA_Fast'] = df['Close'].ewm(span=tp_short, min_periods=tp_short).mean()
    macd['MA_Slow']=df['Close'].ewm(span=tp_long, min_periods=tp_long).mean()
    macd['MACD']=macd['MA_Fast']-macd['MA_Slow']
    macd['Signal'] = macd['MACD'].ewm(span=tp_macd, min_periods=tp_macd).mean()
    macd.dropna(inplace=True)
    macd = macd[['MACD','Signal']]
    return macd

def slope(ser,n):
    "function to calculate the slope of regression line for n consecutive points on a plot"
    ser = (ser - ser.min())/(ser.max() - ser.min())
    x = np.array(range(len(ser)))
    x = (x - x.min())/(x.max() - x.min())
    slopes = [i*0 for i in range(n-1)]
    for i in range(n,len(ser)+1):
        y_scaled = ser[i-n:i]
        x_scaled = x[:n]
        x_scaled = sm.add_constant(x_scaled)
        model = sm.OLS(y_scaled,x_scaled)
        results = model.fit()
        slopes.append(results.params[-1])
    slope_angle = (np.rad2deg(np.arctan(np.array(slopes))))
    return np.array(slope_angle)

def RSI(ohlv, n):
    df = ohlv.copy()
    df['delta'] = df['Close']-df['Close'].shift(1)
    df['gain'] = np.where(df['delta']>=0, df['delta'], 0)
    df['loss'] = np.where(df['delta']<0, abs(df['delta']), 0)
    avg_gain = []
    avg_loss = []
    gain = df['gain'].tolist()
    loss = df['loss'].tolist()
    for i in range(len(df)):
        if i<n:
            avg_gain.append(np.NaN)
            avg_loss.append(np.NaN)
        
        elif i==n:
            avg_gain.append(df['gain'].rolling(window=n, min_periods=n).mean().tolist()[n])
            avg_loss.append(df['loss'].rolling(window=n, min_periods=n).mean().tolist()[n])
        
        elif i>n:
            avg_gain.append((((n-1)*avg_gain[i-1])+gain[i])/n)
            avg_loss.append((((n-1)*avg_loss[i-1])+loss[i])/n)
    
    df['Avg gain'] = np.array(avg_gain)
    df['Avg loss'] = np.array(avg_loss)
    df['RS'] = df['Avg gain']/df['Avg loss']
    df['RSI'] = (100-(100/(1+df['RS'])))
    df.drop('Volume', axis=1)
    return df['RSI']
def BolBands(ohlv, n, st):
    df = ohlv.copy()
    df['SMA'] = df['High'].rolling(window=n, min_periods=n).mean()
    df['BB_up'] = df['SMA'] + (st*(df['High'].rolling(window=n, min_periods=n).std()))
    df['BB_down'] = df['SMA'] - (st*(df['High'].rolling(window=n, min_periods=n).std()))
    df['BB_range'] = df['BB_up']-df['BB_down']
    df.dropna(inplace=True)
    df = df.iloc[:,-4:]
    return df

def TR(DF, n):
    "function to calculate True Range and Average True Range"
    df = DF.copy()
    df['H-L']=abs(df['High']-df['Low'])
    df['H-PC']=abs(df['High']-df['Close'].shift(1))
    df['L-PC']=abs(df['Low']-df['Close'].shift(1))
    df['TR']=df[['H-L','H-PC','L-PC']].max(axis=1,skipna=False)
    df['ATR'] = df['TR'].rolling(n).mean()
    #df['ATR'] = df['TR'].ewm(span=n,adjust=False,min_periods=n).mean()
    df2 = df.drop(['H-L','H-PC','L-PC'],axis=1)
    return df2

def ADX(ohlv, n):
    df = ohlv.copy()
    df['True Range'] = TR(df,n)['TR']
    df['DM+'] = np.where((df['High']-df['High'].shift(1))>(df['Low'].shift(1)-df['Low']), df['High']-df['High'].shift(1), 0)
    df['DM+'] = np.where(df['DM+']<0, 0, df['DM+'])
    df['DM-'] = np.where((df['Low'].shift(1)-df['Low'])>(df['High']-df['High'].shift(1)), df['Low'].shift(1)-df['Low'], 0)
    df['DM-'] = np.where(df['DM-']<0, 0, df['DM-'])   
    tr14 = []
    dmp14 = []
    dmm14 = []
    dmp = df['DM+'].tolist()
    dmm = df['DM-'].tolist()
    tr = df['True Range'].tolist()
    for i in range(len(df)):
        if i<n:
            tr14.append(np.NaN)
            dmp14.append(np.NaN)
            dmm14.append(np.NaN)
        elif i==n:
            tr14.append(df['True Range'].rolling(n).sum().tolist()[n])
            dmp14.append(df['DM+'].rolling(n).sum().tolist()[n])
            dmm14.append(df['DM-'].rolling(n).sum().tolist()[n])
        elif i>n:
            tr14.append(tr14[i-1] - (tr14[i-1]/n) + tr[i])
            dmp14.append(dmp14[i-1] - (dmp14[i-1]/n) + dmp[i])
            dmm14.append(dmm14[i-1] - (dmm14[i-1]/n) + dmm[i])
    
    df['True Range 14'] = np.array(tr14)
    df['DM+ 14'] = np.array(dmp14)
    df['DM- 14'] = np.array(dmm14)
    df['DI+'] = 100*(df['DM+ 14']/df['True Range 14'])
    df['DI-'] = 100*(df['DM- 14']/df['True Range 14'])
    df['DI Sum'] = df['DI+'] + df['DI-']
    df['DI Difference'] = abs(df['DI+'] - df['DI-'])
    df['DX'] = 100*(df['DI Difference']/df['DI Sum'])
    adx = []
    dx = df['DX'].tolist()
    for j in range(len(df)):
        if j<2*n-1:
            adx.append(np.NaN)
        elif j==2*n-1:
            adx.append(df['DX'][j-n+1:j+1].mean())
        elif j>2*n-1:
            adx.append(((n-1)*adx[j-1]+dx[j])/n)
            
    df['ADX'] = np.array(adx)
    return df[['ADX', 'DI+', 'DI-']]

class mom:
    # sto = pd.read_csv('NSE_DATA.csv').iloc[:,0]
    # stocks =[]
    # stocks = sto.tolist()
    
    # begin = dt.datetime(2019, 9, 23) - dt.timedelta(380)
    # end = dt.datetime.today()
    # ohlc_dict = {}
    
    # coal = yf.download('COALINDIA.NS', begin, end, interval='1d')
    
    # for ticker in stocks:
    #     ohlc_dict[ticker] = yf.download(ticker, begin, end, interval='1d')
        
    # stocks = ohlc_dict.keys()
    
    def strat(ohlc_dict, stocks, a=15, b=30, sl=1.015, tgt=0.98, target=0.98):
        # sto = pd.read_csv('NSE_DATA.csv').iloc[:,0]
        # stocks =[]
        # stocks = sto.tolist()
        # begin = dt.datetime(2019, 9, 23) - dt.timedelta(380)
        # end = dt.datetime.today()
        # ohlc_dict={}
        # for ticker in stocks:
        #     ohlc_dict[ticker] = yf.download(ticker, begin, end, interval='1d')
        
        stocks = ohlc_dict.keys()

        ohlv_df = copy.deepcopy(ohlc_dict)
        signal = {}
        ret = {}
        # ret2={}
        return_dict = {}

        
        for ticker in stocks:
            # ohlv_df[ticker]['low_rolling'] = ohlv_df[ticker]['Low'].rolling(window = 72, min_periods=72).min()
            # ohlv_df[ticker] = pd.concat([ohlv_df[ticker], ADX(ohlv_df[ticker], 125)], axis=1)
            # ohlv_df[ticker] = pd.concat([ohlv_df[ticker], (MACD(ohlv_df[ticker], 12,26,9))], axis=1)
            # ohlv_df[ticker]['ATR'] = ATR(ohlv_df[ticker], 25)
            # # ohlv_df[ticker]['MACD_slope'] = slope(ohlv_df[ticker]['MACD'], 14)
            # ohlv_df[ticker]['RSI'] = RSI(ohlv_df[ticker], 14)
            # ohlv_df[ticker]['20_ema'] = ohlv_df[ticker]['Close'].ewm(span=20, min_periods=20).mean()
            # ohlv_df[ticker]['5_ema'] = ohlv_df[ticker]['Close'].ewm(span=5, min_periods=5).mean()
            # # ohlv_df[ticker]['30_ema'] = ohlv_df[ticker]['Close'].ewm(span=30, min_periods=30).mean()
            # # ohlv_df[ticker]['20_day_high'] = ohlv_df[ticker]['High'].rolling(window=126, min_periods=126).max()
            # # ohlv_df[ticker]['Momentum'] = ta.MOM(ohlv_df[ticker]['Low'], timeperiod=5)
            # ohlv_df[ticker] = pd.concat([ohlv_df[ticker], (BolBands(ohlv_df[ticker], 30, 1))], axis=1)
            # ohlv_df[ticker].dropna(inplace=True)
            # ohlv_df[ticker] = ohlv_df[ticker].loc[:la]
            signal[ticker] = ""
            ret[ticker] = []
            # a = dt.datetime(2007,9,17)
            # loc = ohlv_df[ticker].index.get_loc(a)
            # ohlv_df[ticker] = ohlv_df[ticker].iloc[loc:,:]
        
        trades = 0
        close = 0
        win =0
        lose = 0
        stoploss = 0
        pos_convert=0
        conv_target = 0
        conv_loss = 0
        stopwin=0
        sl_win = {}
        sl_lose = 0
        buy_sell = {}
        sell_price={}
        sell_date={}
        for ticker in stocks:
            position=[]
            st_price={}
            sell_price[ticker]=[]
            sell_date[ticker] = []
            print('Calcuting returns for',ticker)
            for i in range(len(ohlv_df[ticker])):
                if signal[ticker] == "":
                    # ret[ticker].append(np.NaN)
                    # if ohlv_df[ticker].index[i-1].time() == dt.time(15,40) and \
                    if ohlv_df[ticker].index[i].time() == dt.time(9, 15) and \
                        (ohlv_df[ticker]['Open'][i-1]) >= 1.0225*(ohlv_df[ticker]['Close'][i-2]) and \
                            (ohlv_df[ticker]['Open'][i-1]) > 1.02*(ohlv_df[ticker]['Open'][i-2]):
                                # abs((ohlv_df[ticker]['Open'][i-3])-(ohlv_df[ticker]['Close'][i-3]))>0:

                            signal[ticker] ="Sell"
                            close = ohlv_df[ticker]['Open'][i]
                            trades+=1
                            st_price[ohlv_df[ticker].index[i]] = close
                            position.append(-1)
                            # price = ohlv_df[ticker]['Open'][i-1] - ohlv_df[ticker]['Close'][i-2]
                        
                            # if ohlv_df[ticker]['Low'][i]<=(close-(0.5*price)):
                            #     signal[ticker] = ""
                            #     ret[ticker].append((((close-(0.5*price))-(0.0005*close)-(0.0005*(close-(0.5*price))))/ohlv_df[ticker]["Close"][i-1])-1)
                            #     sell_price[ticker].append(close-(0.5*price))
                            #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                            #     conv_loss+=1
                                
                            # elif ohlv_df[ticker]['High'][i]>=(close+price):
                            #     signal[ticker] = ""
                            #     ret[ticker].append((((close+price)-(0.0005*close)-(0.0005*(close+price)))/ohlv_df[ticker]["Close"][i-1])-1)
                            #     sell_price[ticker].append(close+price)
                            #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                            #     conv_target+=1
                            
                            # else:
                            ret[ticker].append((ohlv_df[ticker]['Open'][i]/ohlv_df[ticker]['Close'][i])-1)
                            
                            
                            
                            # if ohlv_df[ticker]['Close'][i] >= (1.01*close):
                            #     signal[ticker] = ""
                            #     ret[ticker].append((ohlv_df[ticker]["Open"][i]/((ohlv_df[ticker]['Close'][i])+(0.0005*(ohlv_df[ticker]['Close'][i]))+(0.0005*close)))-1)
                            #     sell_price[ticker].append(1.01*close)
                            #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                            #     lose+=1
                            
                            # elif ohlv_df[ticker]['Low'][i] <= (0.97*close):
                            #     signal[ticker] = ""
                            #     ret[ticker].append((ohlv_df[ticker]["Open"][i]/((0.97*close)+(0.0005*(0.97*close))+(0.0005*close)))-1)
                            #     sell_price[ticker].append(0.97*close)
                            #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                            #     target+=1
                            
                            # else:
                            
                            # if ohlv_df[ticker]['Low'][i] < (tgt*close) and ohlv_df[ticker].index[i].time() < dt.time(9,30):
                            #     signal[ticker] = ""
                            #     ret[ticker].append(((ohlv_df[ticker]['Open'][i])/((tgt*close)+(0.0005*(tgt*close))+(0.0005*close)))-1)
                            #     sell_price[ticker].append(tgt*close)
                            #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                            #     conv_target+=1
                            
                            # else:
                            #     ret[ticker].append(((ohlv_df[ticker]['Open'][i])/ohlv_df[ticker]['Close'][i])-1)
                            # # timeex = i
                            # if ohlv_df[ticker].index[i] not in sl_win:
                            #     sl_win[ohlv_df[ticker].index[i]] = {}
                            #     sl_win[ohlv_df[ticker].index[i]][ticker] = close
                            # else:
                            #     sl_win[ohlv_df[ticker].index[i]][ticker] = close
                            # continue
                        
                    else:
                        ret[ticker].append(np.NaN)
                    
                    # elif ohlv_df[ticker]['Close'][i-1] >= ohlv_df[ticker]['BB_down'][i-1] and \
                    #      ohlv_df[ticker]['Low'][i] < ohlv_df[ticker]['BB_down'][i] and \
                    #          ohlv_df[ticker].index[i].time() < dt.time(15, 15, 00):
                    #     # ohlv_df[ticker]['Open'][i] < (0.99*(ohlv_df[ticker]['Close'][i-1])): 
                            
                    #         signal[ticker] ="Sell"
                    #         close = ohlv_df[ticker]['Close'][i]
                    #         trades+=1
                    #         st_price[ohlv_df[ticker].index[i]] = close
                    #         position.append(-1)
                            # ret[ticker].append((ohlv_df[ticker]['Open'][i]/ohlv_df[ticker]['Close'][i])-1)
                            

                                        
                elif signal[ticker] == "Buy":       
                    if ohlv_df[ticker].index[i].time() == dt.time(15, 15, 00):
                        signal[ticker] = ""
                        ret[ticker].append((((ohlv_df[ticker]['Close'][i])-(0.0005*close)-(0.0005*ohlv_df[ticker]['Close'][i]))/ohlv_df[ticker]["Close"][i-1])-1)
                        sell_price[ticker].append(ohlv_df[ticker]['Close'][i])
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        if ohlv_df[ticker]['Close'][i] > close:
                            stopwin+=1
                        else:
                            stoploss+=1
                    
                    elif ohlv_df[ticker]['Low'][i]<=sl*close:
                        signal[ticker] = ""
                        ret[ticker].append((((sl*close)-(0.0005*close)-(0.0005*(sl*close)))/ohlv_df[ticker]["Close"][i-1])-1)
                        sell_price[ticker].append(sl*close)
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        conv_loss+=1
                        
                    elif ohlv_df[ticker]['High'][i]>=tgt*close:
                        signal[ticker] = ""
                        ret[ticker].append((((tgt*close)-(0.0005*close)-(0.0005*(tgt*close)))/ohlv_df[ticker]["Close"][i-1])-1)
                        sell_price[ticker].append(tgt*close)
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        conv_target+=1
                        
                    # elif ohlv_df[ticker]['High'][i]>=(ohlv_df[ticker]['Close'][i-1]+(2*ohlv_df[ticker]['ATR'][i-1])):
                    #     signal[ticker] = ""
                    #     ret[ticker].append((((ohlv_df[ticker]['Close'][i-1]+(2*ohlv_df[ticker]['ATR'][i-1]))-(0.0005*close)-(0.0005*(ohlv_df[ticker]['Close'][i-1]+(2*ohlv_df[ticker]['ATR'][i-1]))))/ohlv_df[ticker]["Close"][i-1])-1)
                    #     sell_price[ticker].append(ohlv_df[ticker]['Close'][i-1]+(2*ohlv_df[ticker]['ATR'][i-1]))
                    #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                    #     if (ohlv_df[ticker]['Close'][i-1]+(2*ohlv_df[ticker]['ATR'][i-1])) > close:
                    #         sl_win+=1
                    #     else:
                    #         sl_lose+=1
                    
                    else:
                        ret[ticker].append((ohlv_df[ticker]['Close'][i]/ohlv_df[ticker]['Close'][i-1])-1)
                
                elif signal[ticker] == 'Sell':
                    if ohlv_df[ticker].index[i].time() == dt.time(15, 00, 00):
                        signal[ticker] = ""
                        ret[ticker].append((ohlv_df[ticker]['Close'][i-1]/(ohlv_df[ticker]["Close"][i]+(0.0005*ohlv_df[ticker]["Close"][i])+(0.0005*close)))-1)
                        sell_price[ticker].append(ohlv_df[ticker]['Close'][i])
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        if ohlv_df[ticker]['Close'][i] < close:
                            stopwin+=1
                        else:
                            stoploss+=1
                            
                    # elif ohlv_df[ticker]['Low'][i] <= 0.97*close:
                    #     signal[ticker] = ""
                    #     ret[ticker].append((ohlv_df[ticker]["Close"][i-1]/((0.97*close)+(0.0005*(0.97*close))+(0.0005*close)))-1)
                    #     sell_price[ticker].append(0.97*close)
                    #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                    #     target+=1
                        
                    # elif ohlv_df[ticker]['Open'][i]>=1.01*(close):
                    #     signal[ticker] = ""
                    #     # if (1.01*(close))<ohlv_df[ticker]['Open'][i]:
                    #     ret[ticker].append((ohlv_df[ticker]["Close"][i-1]/((ohlv_df[ticker]["Open"][i])+(0.0005*(ohlv_df[ticker]["Open"][i]))+(0.0005*close)))-1)
                    #     sell_price[ticker].append(ohlv_df[ticker]["Open"][i])
                    #     sl_lose+=1
                    #         # sl_win[ticker] = ohlv_df[ticker].index[i]
                    #     # else:
                    #     #     ret[ticker].append((ohlv_df[ticker]["Close"][i-1]/((1.01*close)+(0.0005*(1.01*close))+(0.0005*close)))-1)
                    #     #     sell_price[ticker].append(1.01*close)
                    #     lose+=1
                    #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                        
                        
                    # else:
                    #     ret[ticker].append((ohlv_df[ticker]['Close'][i-1]/ohlv_df[ticker]['Close'][i])-1)
                    elif ohlv_df[ticker]['Low'][i] < (tgt*close) and ohlv_df[ticker].index[i].time() < dt.time(9,30):
                        signal[ticker] = ""
                        ret[ticker].append(((ohlv_df[ticker]['Close'][i-1])/((tgt*close)))-1)
                        sell_price[ticker].append(tgt*close)
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        conv_target+=1
                        
                        # pos_convert+=1
                        # close = ohlv_df[ticker]["Low"][i]
                        # st_price[ohlv_df[ticker].index[i]] = close
                        # position.append(1)
                        # print(ticker, ohlv_df[ticker].index[i])
                    
                    elif ohlv_df[ticker]['High'][i] > sl*close and sl*close > ohlv_df[ticker]['Low'][i]:
                        signal[ticker] = ""
                        ret[ticker].append(((ohlv_df[ticker]['Close'][i-1])/((sl*close)+(0.0005*(sl*close))+(0.0005*close)))-1)
                        sell_price[ticker].append(sl*close)
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        sl_lose+=1
                    
                    # elif ohlv_df[ticker].index[i].time() == dt.time(a,b):
                    #     signal[ticker] = ""
                    #     ret[ticker].append((ohlv_df[ticker]['Close'][i-1]/((ohlv_df[ticker]['Close'][i])+(0.0005*ohlv_df[ticker]['Close'][i])+(0.0005*close)))-1)
                    #     sell_price[ticker].append(ohlv_df[ticker]["Close"][i])
                    #     sell_date[ticker].append(ohlv_df[ticker].index[i])
                    #     if ((ohlv_df[ticker]['Close'][i])+(0.0005*ohlv_df[ticker]['Close'][i])+(0.0005*close)) < close:
                    #         win+=1
                    #     else:
                    #         lose+=1

                    elif ohlv_df[ticker]['Low'][i] <= target*close and ohlv_df[ticker].index[i].time()>=dt.time(9,30):
                        signal[ticker] = ""
                        ret[ticker].append(((ohlv_df[ticker]['Close'][i-1])/((target*close)+(0.0005*(target*close))+(0.0005*close)))-1)
                        sell_price[ticker].append(target*close)
                        sell_date[ticker].append(ohlv_df[ticker].index[i])
                        stopwin+=1

                            
                    # elif ohlv_df[ticker].index[i].time() == dt.time(9,15):
                    #     ret[ticker].append((ohlv_df[ticker]['Open'][i]/ohlv_df[ticker]['Close'][i])-1)                          
                            
                    else:
                        ret[ticker].append((ohlv_df[ticker]['Close'][i-1]/ohlv_df[ticker]['Close'][i])-1)
                        
            buy_sell[ticker] = pd.DataFrame.from_dict(st_price, orient='index')
            if signal[ticker] == "Buy":
                # print(ticker, ohlv_df[ticker].index[i], ohlv_df[ticker]['year_high'][i])
                position.remove(position[-1])
                buy_sell[ticker] = buy_sell[ticker].drop(buy_sell[ticker].index[-1], axis=0)
            if signal[ticker] == "Sell":
                position.remove(position[-1])
                buy_sell[ticker] = buy_sell[ticker].drop(buy_sell[ticker].index[-1], axis=0)
            buy_sell[ticker]['Sell'] = np.array(sell_price[ticker])
            buy_sell[ticker]['Sell_date'] = np.array(sell_date[ticker])
            buy_sell[ticker]['Position'] = np.array(position)
            if signal[ticker] == "Buy":
                trades=trades-1
            if signal[ticker] == "Sell":
                trades=trades-1
            ohlv_df[ticker]["Return"] = np.array(ret[ticker])
        
        # calcuting overall strategy's KPIs
        
        strategy_df = pd.DataFrame()
        new_ret = []
        
        for ticker in stocks:
            strategy_df[ticker] = ohlv_df[ticker]["Return"]
            buy_sell[ticker]['Trading Costs'] = ((0.0005*buy_sell[ticker].iloc[:,1])+(0.0005*buy_sell[ticker].iloc[:,0]))
            buy_sell[ticker]['P&L'] = ((buy_sell[ticker].iloc[:,1]-buy_sell[ticker].iloc[:,0])*buy_sell[ticker]['Position'])-buy_sell[ticker]['Trading Costs']
            buy_sell[ticker]['Return'] = buy_sell[ticker]['P&L']/buy_sell[ticker].iloc[:,0]
            new_ret.append(buy_sell[ticker]["P&L"].sum())
        
        return_dict['ohlv_df'] = ohlv_df
        return_dict['buy_sell'] = buy_sell
        return_dict['no_of_trades'] = trades
        # return_dict['win_trades'] = win
        return_dict['Converted Positions'] =pos_convert
        return_dict['Target 1%'] = conv_target
        return_dict['Stoploss'] = conv_loss
        return_dict['3pm loss'] = lose
        return_dict['3pm profit'] = win
        # return_dict['loss_trades'] = lose
        return_dict['sl lose time'] = sl_win
        return_dict['Stop loss 6%'] = sl_lose
        return_dict['Randomness'] = stoploss+stopwin
        # count = 0
        # for ticker in stocks:
        #     for j in range(len(buy_sell[ticker])):
        #         if buy_sell[ticker]['P&L'][j]==0:
        #             count+=1
        # strategy_df = pd.DataFrame()
        # strategy_df.index = stocks
        # strategy_df['Return'] = np.array(new_ret)
        
        # strategy_df['Return'].sum()
        # strategy_df = strategy_df.replace(0, np.NaN)
        
        strategy_df['Return'] = strategy_df.mean(axis=1)
        strategy_df.fillna(0, inplace=True)
        # strategy_df['Return'] = strategy_df['Return']*5
        
        ind_list={}
        
        for ticker in stocks:
            if buy_sell[ticker].empty:
                ind_list[ticker]=0
            else:   
                ind_list[ticker]=((((1+strategy_df[ticker]).cumprod()[-1])-1)*100)
        
        
        # CAGR(strategy_df)
        # sharpe(strategy_df,0.065)
        # max_dd(strategy_df)  
        return_dict['strategy_df'] = strategy_df
        
        kpi = {}
        
        # kpi['CAGR'] = CAGR(strategy_df)
        # kpi['sharpe'] = sharpe(strategy_df,(0.065/12))
        kpi['max_dd'] = max_dd(strategy_df)
        # kpi['volatility'] = volatility(strategy_df)
        
        return_dict['KPI'] = kpi
        # (strategy_df["Return"]).cumprod()[-1]/max_dd(strategy_df)  
        # beg1 = dt.datetime(2011, 12, 2)
        # beg1 = ohlv_df['RELIANCE.NS'].index[0] - dt.timedelta(1)
        # end1 = dt.datetime.today()
        
        # nse = yf.download('^NSEI', beg1, end1, interval='1d')
        # nse['Return'] = nse['Close'].pct_change()
        
        # (1+ohlv_df['IOC.NS']['Return']).cumprod().plot()
        
        # (1+strategy_df["Return"]).cumprod().plot()
        
        tot_ret = (((1+strategy_df['Return']).cumprod()[-1])-1)*100
        
        if trades!=0:
            win_ratio = ((win+conv_target+stopwin)/trades)*100
        else:
            win_ratio = 00
        
        return_dict['Total_returns'] = tot_ret
        return_dict['Win_loss_ratio'] = win_ratio

                
        ind_stock_ret = pd.DataFrame.from_dict(ind_list, orient='index')
        
        return_dict['ind_stock_returns'] = ind_stock_ret
        
        return return_dict
    
# vizualization of strategy return vs nifty
# fig, ax = plt.subplots()
# plt.plot((1+strategy_df['Return'].reset_index(drop=True)).cumprod())
# plt.plot((1+nse["Return"][1:].reset_index(drop=True)).cumprod())
# plt.title("Index Return vs Strategy Return")
# plt.ybel("cumutive return")
# plt.xbel("months")
# ax.legend(["Strategy Return","Nifty Return"])

# ((1+nse['Return']).cumprod()[-1]-1)*100


