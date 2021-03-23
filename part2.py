# %% imports
import numpy as np
import pandas as pd
import requests
from scipy import stats
from tqdm import tqdm
# %% load stocks
stocks = pd.read_csv('data/sp_500_stocks.csv')

# %% config API
IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']

symbol = 'aapl'
base_url = 'https://sandbox.iexapis.com/stable'
auth = f'token={IEX_CLOUD_API_TOKEN}'

rng = '5d'
endpoint = f'/stock/{symbol}/chart/{rng}'

req = f'{base_url}/{endpoint}?{auth}'
# %% example request
requests.get(req).json()

# %% batch request
syms = ','.join(stocks.Ticker[:2].values)
rng = '5d'
endpoint = f'stock/market/batch?symbols={syms}&types=chart&range={rng}'

req = f'{base_url}/{endpoint}&{auth}'

# %% fetch data

# NOTE: we need to request the data in chunks since API is limited to 100 symbols


def get_data_batch(syms, time_range='5d', lim=100):
    IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']
    base_url = 'https://sandbox.iexapis.com/stable'
    auth = f'token={IEX_CLOUD_API_TOKEN}'

    dat = None
    for i in tqdm(range(0, len(syms), lim)):
        cur_syms = ','.join(syms[i:(i+lim)])
        endpoint = f'stock/market/batch?symbols={cur_syms}&types=chart&range={time_range}'
        req = f'{base_url}/{endpoint}&{auth}'
        resp = requests.get(req).json()

        # reshape data
        df = pd.DataFrame.from_dict(resp, orient='index')
        df = df.reset_index()\
            .explode('chart')\
            .apply(lambda x: pd.Series(x.chart), axis=1)\
            .set_index('symbol')
        dat = pd.concat((dat, df))
    return dat


dat = get_data_batch(stocks.Ticker)

# %% augment & tidy data
print(dat.info())

# drop column 0, introduced by empty response for certain stocks
dat.drop(0, 1, inplace=True)

# for some stocks we do not have chart data, drop
print('No chart data for the following stocks:')
print(list(filter(lambda x: x not in dat.index.unique(), stocks.Ticker)))
dat = dat.dropna(0)

# convert dates to pandas datetime, access to pandas versatile time API
dat.date = pd.to_datetime(dat.date)

# sort such that the most current date is first
dat.sort_values(['symbol', 'date'], ascending=[True, False])

# compute time deltas
dat['delta_t'] = dat.groupby(['symbol']).date.diff()
dat.delta_t = dat.delta_t.fillna(pd.Timedelta(0, unit="d"))
# %%
dat[['date', 'open', 'close', 'changePercent', 'change']]


# %%
