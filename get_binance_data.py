import pandas as pd
from binance.client import Client
from decouple import config

BINANCE_API_KEY = config("BINANCE_API_KEY")
BINANCE_API_SECRET = config("BINANCE_API_SECRET")

client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
time_freq = Client.KLINE_INTERVAL_1MINUTE


def get_binance_data(ticker, start, end):
    columns = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'qav', 'num_trades',
               'taker_base_vol', 'taker_quote_vol', 'ignore']
    df = client.get_historical_klines(ticker, time_freq, start_str=start, end_str=end)
    df = pd.DataFrame(df,
                      columns=columns)
    df.index = [pd.to_datetime(x, unit='ms').strftime('%Y-%m-%d %H:%M:%S') for x in df.open_time]
    df = df.astype(float)
    usecols = ['open', 'high', 'low', 'close', 'volume', 'qav', 'num_trades', 'taker_base_vol', 'taker_quote_vol']
    df = df[usecols]
    return df


data = get_binance_data('BTCUSDT', "1 Oct, 2021", "31 Dec, 2023")
data.to_csv('BTCUSDT_1min_2.csv')
