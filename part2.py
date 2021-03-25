"""
We do not follow the approach in the tutorial of using the One-Year-Return for the momentum.
Rather we use a weighted average of the daily returns, weighted with weight decay.
"""

# %% imports
import matplotlib.pyplot as plt
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
resp = requests.get(req).json()

# %% batch request
syms = ','.join(stocks.Ticker[:2].values)
rng = '5d'
endpoint = f'stock/market/batch?symbols={syms}&types=chart&range={rng}'

req = f'{base_url}/{endpoint}&{auth}'

# %% fetch data

# NOTE: we need to request the data in chunks since API is limited to 100 symbols


def get_data_batch(syms, time_range='3m', lim=100):
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


resp = get_data_batch(stocks.Ticker)

# %% augment & tidy data
dat = resp.copy()
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
dat = dat.sort_values(['symbol', 'date'], ascending=[True, False])

# compute time deltas
dat['delta_t'] = dat.groupby(['symbol']).date.transform(
    lambda x: np.abs(x - x.max()))
# dat.delta_t = dat.delta_t.fillna(pd.Timedelta(0, unit="d"))

# compute daily returns
# NOTE: for now we use the difference between opening and closing
dat['return'] = dat.apply(lambda x: (x.close - x.open)/x.open, axis=1)
# dat[['date', 'open', 'close', 'changePercent', 'change']]

# %% compute weighted average return

# NOTE: use exponential weight decay over time to weight current return more than past


def weighted_average(x, decay_fct=10):
    weights = 1 / np.exp(np.arange(len(x))/10)
    return np.average(x, weights=weights)


momentum = dat.groupby('symbol')['return'].agg(
    weighted_average).rename('momentum')
momentum = momentum.sort_values(ascending=False)

# %% plot high/low momentum stocks
# NOTE: We observe that the high momentum stocks actually have a lower return over the
# observed period of time. However, as expected the high momentum stocks recently show
# high returns and the low momentum negative returns.
top_n = 5

fig, axes = plt.subplots(2, 1, sharex=True)

for sym in momentum.index[:top_n]:
    plot_dat = dat[dat.index == sym].sort_values('date')
    y = plot_dat.close / plot_dat.open[0]  # normalise to 1
    axes[0].plot(plot_dat.date, y, markersize=5, marker='o', label=sym)

for sym in momentum.index[-top_n:]:
    plot_dat = dat[dat.index == sym].sort_values('date')
    y = plot_dat.close / plot_dat.open[0]  # normalise to 1
    axes[1].plot(plot_dat.date, y, markersize=5, marker='o', label=sym)

for ax, title in zip(axes, ['high', 'low']):
    ax.set_title(f'{title} momentum')
    ax.legend(loc='upper left')

fig.tight_layout()
fig.show()

# %% refine strategy
# As a simple refinement of the strategy we including mean return over 3 months

mean_3m_return = dat.groupby('symbol')['return'].agg(
    'mean').rename('mean_3m_return')
mean_3m_return.sort_values(ascending=False, inplace=True)

# %% compute score
# NOTE: compute a score as a linear combination of mean_3_month return and momentum
strat = pd.concat((momentum, mean_3m_return), axis=1)

weights = np.array([.3, .7])
strat['score'] = strat.apply(lambda x: np.average(
    x, weights=weights), axis=1, raw=True)
strat.sort_values('score', ascending=False, inplace=True)
strat
# %%
strat = pd.concat((momentum, mean_3m_return), axis=1)
top_n = 5
weights = [(1, 0), (.25, .75), (.5, .5), (.75, .25), (0, 1)]
fig, axes = plt.subplots(len(weights), 2, sharex=True,
                         figsize=(8, 2*len(weights)))

for cur_weight, cur_axes in zip(weights, axes):
    # compute score with current weights
    strat['score'] = strat.apply(lambda x: np.average(
        x, weights=cur_weight), axis=1, raw=True)
    strat.sort_values('score', ascending=False, inplace=True)

    syms = strat.index[:top_n]
    for sym in syms:
        plot_dat = dat[dat.index == sym].sort_values('date')
        y = plot_dat.close / plot_dat.open[0]  # normalise to 1
        cur_axes[0].plot(plot_dat.date, y, markersize=5, marker='o', label=sym)
        cur_axes[0].legend(loc='upper left')

    syms = strat.index[-top_n:]
    for sym in syms:
        plot_dat = dat[dat.index == sym].sort_values('date')
        y = plot_dat.close / plot_dat.open[0]  # normalise to 1
        cur_axes[1].plot(plot_dat.date, y, markersize=5, marker='o', label=sym)
        cur_axes[1].legend(loc='upper left')

    cur_axes[0].set_ylabel(f'weights: {cur_weight}')
    strat.drop('score', 1, inplace=True)


for ax, title in zip(axes[0, :], ['high', 'low']):
    ax.set_title(f'{title} momentum')

fig.tight_layout()
fig.show()


# %%
