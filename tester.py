from backtesting import Backtest, Strategy
from backtesting.test import GOOG, SMA, EURUSD
# import talib
from backtesting.lib import crossover, resample_apply, TrailingStrategy
import pandas as pd
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mdates
import numpy as np
import mplfinance as mpf
import tensorflow as tf
import cv2
# from sklearn.means import KMeans
from scipy.signal import argrelextrema, find_peaks
from sklearn.neighbors import KernelDensity


'https://github.com/butterrobot2022/trading-bot'
class APIDataConverter:
    def __init__(self, api_data):
        self.api_data = api_data

    def convert_to_dataframe(self):
        
        return pd.DataFrame(self.api_data)


def is_bullish_run(candle1, candle2, candle3, candle4):
    if candle2.Close > candle1.Close and candle3.Close > candle2.Close and candle4.Close > candle3.Close:
        return True
    return False


def is_bearish_run(candle1, candle2, candle3, candle4):
    if candle2.Close < candle1.Close and candle3.Close < candle2.Close and candle4.Close < candle3.Close:
        return True
    return False


def is_bullish_run_3(candle1, candle2, candle3):
    if candle2.Close > candle1.Close and candle3.Close > candle2.Close:
        return True
    return False


def is_bearish_run_3(candle1, candle2, candle3):
    if candle2.Close < candle1.Close and candle3.Close < candle2.Close:
        return True
    return False


def is_bearish_candle(candle):
    if candle.Close < candle.Open:
        return True
    return False


def is_bullish_candle(candle):
    if candle.Close > candle.Open:
        return True
    return False


def process_image(live_image):
    image = cv2.imread(live_image)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize the image to match the expected input shape
    resized_image = cv2.resize(image, new_size)

    # Add a batch dimension to the resized image
    resized_image = np.expand_dims(resized_image, axis=0)

    # Preprocess the image if necessary (e.g., normalize pixel values)

    # Make predictions
    predictions = model.predict(resized_image)
    predicted_class = np.argmax(predictions)


    print(predicted_class)
    return predicted_class


# print(EURUSD.head(15))
# print(SMA)
# print('')


def get_fibonacci_levels(df, trend):
    trend = trend.lower()
    Low = df['Close'].min()
    High = df['High'].max()

    Diff = High - Low
    if trend == 'downtrend':
        Fib100 = High
        Fib618= Low + (Diff * 0.618)
        Fib50 = Low + (Diff * 0.5)
        Fib382 = Low + (Diff * 0.382)
        Fib236 = Low + (Diff * 0.236)
        Fib0 = Low
    else:
        Fib100 = Low
        Fib618= High + (Diff * 0.618)
        Fib50 = High + (Diff * 0.5)
        Fib382 = High + (Diff * 0.382)
        Fib236 = High + (Diff * 0.236)
        Fib0 = High

    return Fib0, Fib236, Fib382, Fib50, Fib618, Fib100


def optim_func(series):

    if series['# Trades'] < 10:
        return -1
    return series['Equity Final [$]'] / series['Exposure Time [%]']


model = tf.keras.models.load_model('/Users/motin/Downloads/traffic/traffic/trained_model')
new_size = (30, 30)
# candlestick_chart_png = 'C:\Users/motin/Downloads/traffic/traffic/candlestick_chart.png'

class Strat(TrailingStrategy):

    n0 = 18 # Exponential Moving Average
    n1 = 50 # Exponential Moving Average
    n2 = 200 # Simple Moving Average
    current_day = 0
    equity = 1000
    risk_percentage = 0.05
    reward_percentage = 0.15
    # current_price = 0
    reward_ratio = 2
    position_size = 0.01


    def init(self):

        super().init()
        super().set_trailing_sl(3)
    
        close = self.data.Close
        self.daily_sma0 = self.I(SMA, close, self.n0)
        self.daily_sma1 = self.I(SMA, close, self.n1)
        self.daily_sma2 = self.I(SMA, close, self.n2)

        self.hourly_sma0 = resample_apply(
            '1H', SMA, self.data.Close, self.n0
        )

        self.hourly_sma1 = resample_apply(
            '1H', SMA, self.data.Close, self.n1
        )
        self.hourly_sma2 = resample_apply(
            '1H', SMA, self.data.Close, self.n2
            
        )

    
    def support_and_resistance(self, df):
        peaks_range = [2, 3]
        num_peaks = -999

        sample_df = df.tail(120)
        sample = sample_df['Close'].to_numpy().flatten()
        sample_original = sample.copy()

        maxima = argrelextrema(sample, np.greater)
        minima = argrelextrema(sample, np.less)

        extrema = np.concatenate((maxima, minima), axis=1)[0]
        extrema_prices = np.concatenate((sample[maxima], sample[minima]))
        interval = extrema_prices[0]/10000

        bandwidth = interval

        while num_peaks < peaks_range[0] or num_peaks > peaks_range[1]:
            initial_price = extrema_prices[0]
            kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(extrema_prices.reshape(-1, 1))

            a, b = min(extrema_prices), max(extrema_prices)
            price_range = np.linspace(a, b, 1000).reshape(-1, 1)


            pdf = np.exp(kde.score_samples(price_range))
            peaks = find_peaks(pdf)[0]
            num_peaks = len(peaks)
            bandwidth += interval

            if bandwidth > 100*interval:
                print('Failed to converge, stopping...')
                break
        # print(price_range[peaks])
        new_price_range = price_range[peaks]
        new_price_range = np.delete(new_price_range, 1, axis=0)

        # Set the style of the plot
        sample_df.index = pd.to_datetime(sample_df.index)
        style = mpf.make_mpf_style(base_mpf_style='classic')

        # Create the figure object without plotting
        fig, axes = mpf.plot(sample_df, type='candle', volume=True, returnfig=True, style=style)

        # Save the figure to a file
        fig.savefig('sar_chart.png')
        price = self.data.Close[-1]
        
        if process_image('/Users/motin/Downloads/traffic/traffic/sar_chart.png') == 1:

            if price <= new_price_range[0][0]:
                if self.position:
                    self.position.close()
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount

                tp_level = price + (reward_amount/self.equity)
                sl_level = price - (risk_amount/self.equity)
                self.buy(tp=tp_level, sl=sl_level)
            elif price >= new_price_range[1][0]:
                if self.position:
                    self.position.close()
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount
                tp_level = price - (reward_amount/self.equity)
                sl_level = price + (risk_amount/self.equity)
                self.sell(tp=tp_level, sl=sl_level)
               

    def bullish_engulfing(self, df):
        df_test = df.tail(5)
        df_test = df_test.drop_duplicates()
        test_size = len(df)    
        num_engulfing = 0

        for i in range(test_size-1):
            first_candle = df_test.iloc[i-1]
            second_candle = df_test.iloc[i-2]
            third_candle  = df_test.iloc[i-3]
            fourth_candle = df_test.iloc[i-4]
            fifth_candle = df_test.iloc[i-5]
            second_test = first_candle.Close > second_candle.Open

            if is_bearish_candle(second_candle) and is_bullish_candle(first_candle) and second_test == True and is_bearish_run(fifth_candle, fourth_candle, third_candle, second_candle):
                num_engulfing += 1
                price = self.data.Close[-1]
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount

                tp_level = price + (reward_amount/self.equity)
                sl_level = price - (risk_amount/self.equity)

                # print('Bullish Engulfing')
                if self.hourly_sma1[-1] > self.hourly_sma2[-1] and not self.position:

                    # Set the style of the plot
                    df.index = pd.to_datetime(df.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(df.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')
                    if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2:
                        levels = get_fibonacci_levels(df=df.tail(75), trend='uptrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= price <= sixty_one8_retracement:
                            self.buy(tp=tp_level, sl=sl_level)
            break


    def bearish_engulfing(self, df):
        df_test = df.tail(5)
        df_test = df_test.drop_duplicates()
        test_size = len(df)    
        num_engulfing = 0

        for i in range(test_size-1):
            first_candle = df_test.iloc[i-1]
            second_candle = df_test.iloc[i-2]
            third_candle  = df_test.iloc[i-3]
            fourth_candle = df_test.iloc[i-4]
            fifth_candle = df_test.iloc[i-5]
            # first_test = first_candle.Open < second_candle.Close
            second_test = first_candle.Close < second_candle.Open

            if is_bullish_candle(second_candle) and is_bearish_candle(first_candle) and second_test == True and is_bullish_run(fifth_candle, fourth_candle, third_candle, second_candle):
                num_engulfing += 1
                # print('Bearish Engulfing')
                price = self.data.Close[-1]
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount

                tp_level = price - (reward_amount/self.equity)
                sl_level = price + (risk_amount/self.equity)

                if self.hourly_sma1[-1] < self.hourly_sma2[-1] and not self.position:
                    df.index = pd.to_datetime(df.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(df.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')
                    if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0: 
                        levels = get_fibonacci_levels(df=df.tail(75), trend='downtrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= price <= sixty_one8_retracement: 
                            self.sell(tp=tp_level, sl=sl_level)
            break
    

    def shooting_star(self, df):
        # print('')
        df = df.tail(5)
        df = df.drop_duplicates()
        test_size = len(df) 
        num_shooting_stars = 0
        bullish_shooting_stars = 0

        for i in range((test_size-1)-3):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            price = self.data.Close[-1]
            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test= abs(testing_candle.High - testing_candle.Close)
                if 2 < test < 2.1:
                    num_shooting_stars += 1
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)
                    
                    # print('bearish shooting star')
                    if not self.position and self.hourly_sma1[-1] < self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') != 2:    
                        self.sell(tp=tp_level, sl=sl_level)
            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test = abs(testing_candle.High - testing_candle.Close)
                if test > 2 and test < 2.1:
                    bullish_shooting_stars += 1
                    # print('bullish shooting star')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)

                    if not self.position and self.hourly_sma1[-1] > self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') != 0:
                        self.buy(size=0.05, tp=tp_level, sl=sl_level)
    

    def bullish_pinbar(self, df):
        df = df.tail(1)
        df = df.drop_duplicates()
        test_size = len(df)    
        num_pin_bars = 0
        price = self.data.Close[-1]

        for i in range(test_size-1):
            candle = df.iloc[i]
            is_pin_bar = (candle.Close - candle.Low) > 0.0004
            if is_pin_bar:
                num_pin_bars += 1
                # print('Bullish Pin Bar')
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount

                tp_level = price + (reward_amount/self.equity)
                sl_level = price - (risk_amount/self.equity)
                if not self.position and self.hourly_sma1[-1] > self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0:
                    self.buy(size=0.05, tp=tp_level, sl=sl_level)


    def bearish_pinbar(self, df):
        df = df.tail(1)
        df = df.drop_duplicates()
        test_size = len(df)    
        num_pin_bars = 0

        for i in range(test_size-1):
            candle = df.iloc[i]
            is_pin_bar = abs(candle.Close - candle.High) <  0.0004
            if is_pin_bar:
                num_pin_bars += 1
                # print('Bearish Pin Bar')
                price = self.data.Close[-1]
                risk_amount = self.equity * self.risk_percentage
                reward_amount = self.equity * self.reward_percentage
                r2r_ratio = reward_amount / risk_amount

                tp_level = price - (reward_amount/self.equity)
                sl_level = price + (risk_amount/self.equity)

                if not self.position and self.hourly_sma1[-1] < self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2:
                    self.sell(size=0.05, tp=tp_level, sl=sl_level)


    def doji_star(self, df):
        # print('')
        df = df.drop_duplicates()
        df = df.tail(5)
        test_size = len(df) 
        bullish_doji = 0
        bearish_doji = 0
        
        for i in range(test_size-4):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            price = self.data.Close[-1]

            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test1 = testing_candle.High - testing_candle.Close
                test2 = testing_candle.Close - testing_candle.Low
                if test1 == test2:
                    bullish_doji += 1
                    # print('bullish doji star')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)
                    
                    if not self.position and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2:
                        self.sell(size=0.05, tp=tp_level, sl=sl_level)
            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test1 = testing_candle.Open - testing_candle.Close
                test2 = testing_candle.Close - testing_candle.Low
                if test1 == test2:
                    bearish_doji += 1
                    # print('bearish doji star')
                    # price = self.data.Close[-1]
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)
                   
                    if not self.position and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0:
                        self.buy(size=0.05, tp=tp_level, sl=sl_level)


    def three_white_soldier(self, df):
        # print('')
        dataframe = df.drop_duplicates()
        df = df.drop_duplicates()
        df = df.tail(6)
        test_size = len(df) 
        three_white_soldiers = 0
        three_black_crows = 0

        for i in range(test_size-5):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            testing_candle_2 = df.iloc[i+5]
            price = self.data.Close[-1]

            if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if testing_candle_2.Close > testing_candle.Close and testing_candle.Close > prev_candle.Close:
                    three_white_soldiers += 1
                    # print('bullish three white soldiers')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)
                    if self.hourly_sma1[-1] > self.hourly_sma2[-1]:
                        dataframe.index = pd.to_datetime(dataframe.index)
                        style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)

                        # Save the figure to a file
                        fig.savefig('candlestick_chart.png')
                        if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2 and not self.position:
                            levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                            thirty_eight_retracement = levels[2]
                            sixty_one8_retracement = levels[4]
                            if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                                self.buy(tp=tp_level, sl=sl_level)
            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if testing_candle_2.Close < testing_candle.Close and testing_candle.Close < prev_candle.Close:
                    three_black_crows += 1
                    # print('bearish three black crows')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)
                    if self.hourly_sma1[-1] < self.hourly_sma2[-1]:
                        df.index = pd.to_datetime(df.index)
                        style = mpf.make_mpf_style(base_mpf_style='classic')

                        # Create the figure object without plotting
                        fig, axes = mpf.plot(df.tail(75), type='candle', volume=True, returnfig=True, style=style)

                        # Save the figure to a file
                        fig.savefig('candlestick_chart.png')
                        if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0 and not self.position:
                            levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                            thirty_eight_retracement = levels[2]
                            sixty_one8_retracement = levels[4]
                            if thirty_eight_retracement <= prev_candle.Close <= sixty_one8_retracement:
                                self.sell(tp=tp_level, sl=sl_level)


    def morning_star(self, df):
        # print('')
        dataframe = df.drop_duplicates()
        df = df.drop_duplicates()
        df = df.tail(6)
        test_size = len(df) 
        morning_stars = 0
        evening_stars = 0
        price = self.data.Close[-1]

        for i in range(test_size-5):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            testing_candle_2 = df.iloc[i+5]

            if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test = testing_candle.Open - testing_candle.Close
                if testing_candle_2.Close > testing_candle.Close and 0 < test < 2:
                    morning_stars += 1
                    # print('bullish morning star')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)

                    dataframe.index = pd.to_datetime(dataframe.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')
                    if self.hourly_sma1[-1] > self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2 and not self.position:
                        levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                            self.buy(tp=tp_level, sl=sl_level)

            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                test = testing_candle.Open - testing_candle.Close
                if testing_candle_2.Close < testing_candle.Close and 0 < test < 2 and testing_candle.Close < prev_candle.Close:
                    evening_stars += 1
                    # print('bearish morning star')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)

                    dataframe.index = pd.to_datetime(dataframe.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')

                    if self.hourly_sma1[-1] < self.hourly_sma2[-1] and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0 and not self.position:
                        levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= testing_candle.Close <= sixty_one8_retracement:
                            self.sell(tp=tp_level, sl=sl_level)


    def matching(self, df):
        # print('')
        df = df.drop_duplicates()
        df = df.tail(5)
        test_size = len(df) 
        matching_lows = 0
        matching_highs = 0
        price = self.data.Close[-1]

        for i in range(test_size-4):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]

            if is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if prev_candle.Low == testing_candle.Low and prev_candle.Close == testing_candle.Close:
                    matching_lows += 1
                    # print('matching low')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)
            
                    if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') != 0 and not self.position:
                        self.buy(size=0.05, tp=tp_level, sl=sl_level)
            elif is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle):
                if prev_candle.High == testing_candle.High and prev_candle.High == testing_candle.High:
                    matching_highs += 1 
                    # print('matching high')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)
                    if process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') != 2 and not self.position:
                        self.sell(size=0.05, tp=tp_level, sl=sl_level)


    def methods(self, df):
        dataframe = df.drop_duplicates()
        df = df.drop_duplicates()
        df = df.tail(8)
        test_size = len(df) 
        rising_methods = 0
        falling_methods = 0
        price = self.data.Close[-1]

        for i in range(test_size-7):
            first_prev_candle = df.iloc[i]
            second_prev_candle = df.iloc[i+1]
            third_prev_candle = df.iloc[i+2]
            prev_candle = df.iloc[i+3]
            testing_candle = df.iloc[i+4]
            testing_candle_2 = df.iloc[i+5]
            testing_candle_3 = df.iloc[i+6]
            final_candle = df.iloc[7]

            if is_bullish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.Close < prev_candle.Close and is_bearish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                if final_candle.Close > prev_candle.Close:
                    rising_methods += 1
                    # print('rising three methods')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price + (reward_amount/self.equity)
                    sl_level = price - (risk_amount/self.equity)

                    dataframe.index = pd.to_datetime(dataframe.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')

                    if self.hourly_sma1[-1] > self.hourly_sma2[-1]  and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 2 and not self.position:
                        levels = get_fibonacci_levels(df=dataframe.tail(75), trend='uptrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                            self.buy(tp=tp_level, sl=sl_level)

            elif is_bearish_run(first_prev_candle, second_prev_candle, third_prev_candle, prev_candle) and testing_candle.Close > prev_candle.Close and is_bullish_run_3(testing_candle, testing_candle_2, testing_candle_3):
                if final_candle.Close < prev_candle.Close:
                    falling_methods += 1
                    # print('falling three methods')
                    risk_amount = self.equity * self.risk_percentage
                    reward_amount = self.equity * self.reward_percentage
                    r2r_ratio = reward_amount / risk_amount

                    tp_level = price - (reward_amount/self.equity)
                    sl_level = price + (risk_amount/self.equity)

                    dataframe.index = pd.to_datetime(dataframe.index)
                    style = mpf.make_mpf_style(base_mpf_style='classic')

                    # Create the figure object without plotting
                    fig, axes = mpf.plot(dataframe.tail(75), type='candle', volume=True, returnfig=True, style=style)

                    # Save the figure to a file
                    fig.savefig('candlestick_chart.png')

                    if self.hourly_sma1[-1] < self.hourly_sma2[-1]  and process_image('/Users/motin/Downloads/traffic/traffic/candlestick_chart.png') == 0 and not self.position:
                        levels = get_fibonacci_levels(df=dataframe.tail(75), trend='downtrend')
                        thirty_eight_retracement = levels[2]
                        sixty_one8_retracement = levels[4]
                        if thirty_eight_retracement <= testing_candle_3.Close <= sixty_one8_retracement:
                            self.sell(tp=tp_level, sl=sl_level)


    def analyze_candlesticks(self, df):
        # self.support_and_resistance(df=df)
        self.bullish_engulfing(df=df)
        # self.bearish_engulfing(df=df)
        # self.bullish_pinbar(df=df)
        # self.bearish_pinbar(df=df)
        # self.shooting_star(df=df)
        # self.doji_star(df=df)
        # self.three_white_soldier(df=df)
        # self.morning_star(df=df)
        # self.matching(df=df)
        # self.methods(df=df)


    def next(self):
        super().next()
        # Creating a Pandas DataFrame
        # print(self.data)
        df = pd.DataFrame({'Open': self.data.Open, 'High': self.data.High, 'Low': self.data.Low, 'Close': self.data.Close, 'Volume': self.data.Volume})
        df.dropna(inplace=True)
        new_day = self.data.index[-1].day
        if self.current_day < new_day and self.position:
            self.position.close()
        self.current_day = new_day
        self.analyze_candlesticks(df=df)
        # print(self._broker.equity)


bt = Backtest(
    EURUSD, Strat,
    cash=100000, 
    # trade_on_close=True
)

# output = bt.optimize(
#     n1 = range(6, 50, 4),
#     n2 = range(50, 200, 5),
#     maximize = optim_func,
#     # constraint = lambda param: param.n2 > param.n1
# )
output = bt.run()
# bt.plot()

print(output)
print(output._strategy)

