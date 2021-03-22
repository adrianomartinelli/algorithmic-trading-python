# %% imports
import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import os
# %% import stock list
stocks = pd.read_csv('data/sp_500_stocks.csv')

# %% set up API call
# add vars to conda env: conda env config vars set IEX_CLOUD_API_TOKEN=YOUR_SECRET_TOKEN
IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']

base_url = 'https://sandbox.iexapis.com/stable'
auth = f'token={IEX_CLOUD_API_TOKEN}'

# %% fetch data, slow sequential calls


def get_data(sym):
    endpoint = f'/stock/{sym}/quote'
    req = f'{base_url}/{endpoint}/?{auth}'

    # %% fetch data
    resp = requests.get(req).json()
    ser = pd.Series({'Ticker': sym,
                     'Market Capitalisation': resp['marketCap'],
                     'Stock Price': resp['latestPrice'],
                     'Number of Shares to Buy': pd.NA
                     })
    return ser


stocks = stocks.iloc[:4, ].apply(lambda x: get_data(x.Ticker), axis=1)

# %% fast batch call


def get_data_batch(syms, endpoints=['marketCap', 'latestPrice'], lim=100):
    dat = None

    for i in range(0, len(syms), lim):
        print(i)
        symbols = ','.join(syms[i:(i+lim)])
        req = f'{base_url}/stock/market/batch?symbols={symbols}&types=quote&{auth}'
        resp = requests.get(req).json()

        # reshape data
        df = pd.DataFrame.from_dict(resp).T
        df = df.apply(lambda x: pd.Series(x.quote), axis=1)[endpoints]

        dat = pd.concat((dat, df))
    return dat


stocks = get_data_batch(stocks.Ticker)
# %%
stocks = stocks.reset_index()
stocks.columns = ['Ticker', 'Market Capitalisation', 'Stock Price']
stocks['Number of Shares to Buy'] = pd.NA

# %%
