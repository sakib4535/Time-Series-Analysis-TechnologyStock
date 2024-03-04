import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_sma(data, column='Close', window=50):
    return data[column].rolling(window=window).mean()
def calculate_ema(data, window=50):
    return data['Close'].ewm(span=window, adjust=False).mean()

def calculate_macd(data, short_window=12, long_window=26):
    short_ema = calculate_ema(data, window=short_window)
    long_ema = calculate_ema(data, window=long_window)

    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=9, adjust=False).mean()

    return macd_line, signal_line

def calculate_rsi(data, window=14):
    daily_returns = data['Close'].pct_change()
    gain = daily_returns.where(daily_returns > 0, 0)
    loss = -daily_returns.where(daily_returns < 0, 0)

    avg_gain = gain.rolling(window=window, min_periods=1).mean()
    avg_loss = loss.rolling(window=window, min_periods=1).mean()

    relative_strength = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + relative_strength))

    return rsi

def calculate_stochastic_oscillator(data, k_window=14, d_window=3):
    low_min = data['Low'].rolling(window=k_window).min()
    high_max = data['High'].rolling(window=k_window).max()

    k_percent = ((data['Close'] - low_min) / (high_max - low_min)) * 100
    d_percent = k_percent.rolling(window=d_window).mean()

    # Identify overbought and oversold positions
    data['Overbought'] = np.where(k_percent > 80, 1, 0)
    data['Oversold'] = np.where(k_percent < 20, 1, 0)

    # Identify lower low, higher low, lower high, and higher high positions
    data['LowerLow'] = np.where(data['Low'].rolling(window=k_window).min().shift(-1) > data['Low'], 1, 0)
    data['HigherLow'] = np.where(data['Low'].rolling(window=k_window).min().shift(-1) < data['Low'], 1, 0)
    data['LowerHigh'] = np.where(data['High'].rolling(window=k_window).max().shift(-1) > data['High'], 1, 0)
    data['HigherHigh'] = np.where(data['High'].rolling(window=k_window).max().shift(-1) < data['High'], 1, 0)

    return k_percent, d_percent

def calculate_bollinger_bands(data, column='Close', window=20, num_std=2):
    sma = calculate_sma(data, column, window)
    rolling_std = data[column].rolling(window=window).std()

    upper_band = sma + (rolling_std * num_std)
    lower_band = sma - (rolling_std * num_std)

    return upper_band, lower_band


def calculate_atr(data, high='High', low='Low', close='Close', window=14):
    data['TR'] = np.maximum(data[high] - data[low],
                            np.maximum(np.abs(data[high] - data[close].shift()),
                                       np.abs(data[low] - data[close].shift())))

    data['ATR'] = data['TR'].rolling(window=window).mean()
    data.drop(columns=['TR'], inplace=True)

    return data['ATR']

def calculate_obv(data):
    obv = np.where(data['Close'] > data['Close'].shift(1), data['Volume'], 0)
    obv = pd.Series(obv, index=data.index)
    obv = obv.cumsum()
    return obv


def plot_technical_indicators(stock_data):
    # Close Price, 50-day SMA, Upper and Lower Bollinger Bands
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['SMA_50'], label='50-day SMA', color='orange')
    plt.plot(stock_data['Upper_Band'], label='Upper Bollinger Band', color='green')
    plt.plot(stock_data['Lower_Band'], label='Lower Bollinger Band', color='red')
    plt.title('Close Price and Bollinger Bands')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Exponential Moving Average (EMA)
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['Close'], label='Close Price', color='blue')
    plt.plot(stock_data['EMA'], label='EMA', color='orange')
    plt.title('Close Price and Exponential Moving Average (EMA)')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # MACD and Signal Line with Crossover Circles
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['MACD'], label='MACD', color='blue')
    plt.plot(stock_data['Signal_Line'], label='Signal Line', color='orange')

    # Highlight crossover points with circles
    crossover_points = stock_data[stock_data['MACD'] > stock_data['Signal_Line']].index
    plt.scatter(crossover_points, stock_data['MACD'].loc[crossover_points], marker='o', color='green',
                label='Crossover')

    crossover_points = stock_data[stock_data['MACD'] < stock_data['Signal_Line']].index
    plt.scatter(crossover_points, stock_data['MACD'].loc[crossover_points], marker='o', color='red', label='Crossunder')

    plt.title('MACD and Signal Line')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # RSI
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['RSI'], label='RSI', color='blue')
    plt.title('Relative Strength Index (RSI)')
    plt.xlabel('Date')
    plt.ylabel('RSI Value')
    plt.legend()
    plt.show()

    # Stochastic Oscillator (%K and %D) with Overbought and Oversold Arrows
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['%K'], label='%K', color='blue')
    plt.plot(stock_data['%D'], label='%D', color='orange')

    # Highlight overbought and oversold positions with arrows
    overbought_positions = stock_data[stock_data['%K'] > 80].index
    oversold_positions = stock_data[stock_data['%K'] < 20].index

    plt.scatter(overbought_positions, stock_data['%K'].loc[overbought_positions], marker='^', color='red',
                label='Overbought')
    plt.scatter(oversold_positions, stock_data['%K'].loc[oversold_positions], marker='v', color='green',
                label='Oversold')

    plt.title('Stochastic Oscillator (%K and %D)')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.legend()
    plt.show()

    # ATR
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['ATR'], label='ATR', color='blue')
    plt.title('Average True Range (ATR)')
    plt.xlabel('Date')
    plt.ylabel('ATR Value')
    plt.legend()
    plt.show()

    # On-Balance Volume (OBV)
    plt.figure(figsize=(12, 6))
    plt.plot(stock_data['OBV'], label='OBV', color='purple')
    plt.title('On-Balance Volume (OBV)')
    plt.xlabel('Date')
    plt.ylabel('OBV Value')
    plt.legend()
    plt.show()


    fig, ax1 = plt.subplots(figsize=(12, 6))


    ax1.bar(stock_data.index, stock_data['Volume'],
            color=np.where(stock_data['Close'] > stock_data['Open'], 'green', 'red'))


    ax2 = ax1.twinx()
    ax2.plot(stock_data['ATR'], label='ATR', color='blue')
    ax2.set_ylabel('ATR Value', color='blue')
    ax2.legend(loc='upper right')


    plt.title('Volume Bars with ATR Color Mapping')
    plt.xlabel('Date')
    plt.show()

csv_file_path = r"F:\Pycharm Central Zone\Time Series Event Analysis\data_collection\stock_data.csv"
stock_data = pd.read_csv(csv_file_path, index_col='Date', parse_dates=True)

stock_data['EMA'] = calculate_ema(stock_data, window=50)

stock_data['MACD'], stock_data['Signal_Line'] = calculate_macd(stock_data)

stock_data['RSI'] = calculate_rsi(stock_data, window=14)
stock_data['OBV'] = calculate_obv(stock_data)
stock_data['%K'], stock_data['%D'] = calculate_stochastic_oscillator(stock_data)
stock_data['SMA_50'] = calculate_sma(stock_data, column='Close', window=50)
stock_data['Upper_Band'], stock_data['Lower_Band'] = calculate_bollinger_bands(stock_data, column='Close', window=20, num_std=2)
stock_data['ATR'] = calculate_atr(stock_data, high='High', low='Low', close='Close', window=14)
print(stock_data.dropna()[['Close', 'SMA_50', 'Upper_Band', 'Lower_Band', 'EMA', 'MACD', 'Signal_Line', 'RSI', '%K', '%D', 'ATR']])
plot_technical_indicators(stock_data)