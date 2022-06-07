# Backtest Gap up short strategy
Modifiable code template for backtesting gap-up short strategy for NIFTY 200 stocks.



## Principle of the Strategy

- Gaps are spaces on a chart that emerge when the price of the financial instrument significantly changes with little or no trading in-between.
- Gaps occur unexpectedly as the perceived value of the investment changes, due to underlying fundamental or technical factors.
- Gaps are classified as breakaway, exhaustion, common, or continuation, based on when they occur in a price pattern and what they signal.

The principle we use here is, when a stock gaps up 2% or more from the previous day's close, then we assume that the traders will take profit and exit the trade causing the prices to drop and we make a profit had we have shorted that particular stock.

## Required Libraries

- pandas
- numpy
- tvdatafeed
    
    Check the reference for installation and further usage of tvdatafeed : https://github.com/StreamAlpha/tvdatafeed
- talib
