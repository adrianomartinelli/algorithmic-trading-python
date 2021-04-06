# %% imports
from scipy.stats import percentileofscore
import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import os
from tqdm import tqdm

# %% load stocks
stocks = pd.read_csv('data/sp_500_stocks.csv')

# %% config API
IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']
base_url = 'https://sandbox.iexapis.com/stable'
auth = f'token={IEX_CLOUD_API_TOKEN}'

# %% fetch data, slow sequential calls


def get_data_batch(syms, time_range='3m', lim=100):
    IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']
    base_url = 'https://sandbox.iexapis.com/stable'
    auth = f'token={IEX_CLOUD_API_TOKEN}'

    dat = None
    for i in tqdm(range(0, len(syms), lim)):
        cur_syms = ','.join(syms[i:(i+lim)])
        endpoint = f'stock/market/batch?symbols={cur_syms}&types=quote,advanced-stats'
        req = f'{base_url}/{endpoint}&{auth}'
        resp = requests.get(req).json()

        # reshape data
        resp = pd.DataFrame.from_dict(resp, orient='index')
        df1 = resp.apply(lambda x: pd.Series(x['advanced-stats']), axis=1)
        df2 = resp.apply(lambda x: pd.Series(x['quote']), axis=1)
        df = pd.concat((df1, df2), axis=1)
        dat = pd.concat((dat, df))
    return dat


resp = get_data_batch(stocks.Ticker)

# %% select & compute features
# NOTE: EValue := enterprise value
dat = resp.copy()
dat['EValue/GP'] = dat['enterpriseValue']/dat['grossProfit']
dat['EValue/EBITDA'] = dat['enterpriseValue']/dat['EBITDA']

cols = ['latestPrice', 'peRatio', 'priceToBook',
        'priceToSales', 'EBITDA', 'enterpriseValue', 'grossProfit',
        'EValue/EBITDA', 'EValue/GP']
dat = dat[cols]

# %% fillna
print(dat.isna().sum())

# NOTE: somehow passing a dict does not work, A value is trying to be set on a copy of a slice from a DataFrame warning
# means = dat.mean(0).to_dict()
# dat = dat.fillna(means)

dat = dat.apply(lambda x: x.fillna(x.mean()))
dat = dat.iloc[:, [0, *list(range(2, dat.shape[1]))]]  # drop correct peRatio

# %% compute percentiles

cols = ['peRatio', 'priceToBook', 'priceToSales', 'EValue/EBITDA', 'EValue/GP']

for col in cols:
    def score(x): return percentileofscore(dat[col], x[col])
    dat[f'{col}_percentil'] = dat.apply(score, 1)

# %% compute rank
for col in cols:
    dat[f'{col}_rank'] = dat[col].rank(ascending=False)

# %% compute Robust Value (RV) score

percentil_cols = [f'{col}_percentil' for col in cols]
dat['percentil_RV'] = dat.apply(lambda x: x[percentil_cols].mean(), axis=1)

rank_cols = [f'{col}_rank' for col in cols]
dat['rank_RV'] = dat.apply(lambda x: x[rank_cols].mean(), axis=1)

# %%
dat.sort_values('percentil_RV', ascending=False, inplace=True)
dat
# %%
# NOTE: rest of the tutorial very repetitive.
