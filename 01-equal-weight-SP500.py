# %% imports
import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import os
from tqdm import tqdm
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

    for i in tqdm(range(0, len(syms), lim)):
        symbols = ','.join(syms[i:(i+lim)])
        req = f'{base_url}/stock/market/batch?symbols={symbols}&types=quote&{auth}'
        resp = requests.get(req).json()

        # reshape data
        df = pd.DataFrame.from_dict(resp).T
        df = df.apply(lambda x: pd.Series(x.quote), axis=1)[endpoints]

        dat = pd.concat((dat, df))
    return dat


dat = get_data_batch(stocks.Ticker)
# %%
dat = dat.reset_index()
dat.columns = ['Ticker', 'Market Capitalisation', 'Stock Price']
dat['Number of Shares to Buy'] = pd.NA

# %%
# NOTE: We implement an approach to use as much as possible from the available assests by not simply rounding down
# like in the tutorial but sequentially buy shares and recompute the remaining amount available

assets = 1e7
N = len(dat)


def get_n_shares(price, assets, N):
    invest = assets / N
    n_shares = np.round(invest / price)

    assets -= n_shares * price
    N -= 1

    return int(n_shares)


dat['n_shares'] = dat.apply(lambda x: get_n_shares(
    x['Stock Price'], assets, N), axis=1)

# %% approach in tutorial
assets = 1e7
N = len(dat)
val = assets / N

dat['Number of Shares to Buy'] = dat.apply(
    lambda x: int(val / x['Stock Price']), axis=1)

# %%
print('Spent tutorial approach:')
print(dat.apply(lambda x: x['Stock Price'] *
      x['Number of Shares to Buy'], axis=1).sum())
print('Spent own approach:')
print(dat.apply(lambda x: x['Stock Price'] * x['n_shares'], axis=1).sum())

# %%
