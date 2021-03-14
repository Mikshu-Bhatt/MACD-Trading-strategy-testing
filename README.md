# MACD-Trading-strategy-testing

In this strategy we are going to analyze stock price movement with moving average convergence divergence method.

First of all we are going to import the required libraries and loading csv file to note book.

Than we are creating two different moving averages of different spans.

After that we need to create two different signals, one by subtracting long moving average from short moving average and another signal by taking moving average of span 9 of the first signal.

Now we are making a function by which we will determine wether to buy and sell, the logic behind this is when oscilator excceds MACD you should buy and when MACD exceeds oscilator you should sell.

we are also providing a definite amount to trade at beggining.

now we are plotting buy and sell decisions on Adj Close price line to visualize decisions.

Finally we are calculating the amount of profit you would make if you follow this strategy considering you are buying and selling 1 share at at time.

# we are making few assumptions here

You can not short sell.

You need to buy and sell same number of shares all the time.
