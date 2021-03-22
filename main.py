# %% imports
import numpy as np
import pandas as pd
import requests
import xlsxwriter
import math
import os
# %% import stock list
stocks = pd.read_csv('data/sp_500_stocks.csv')

# %% API, free-token
# add vars to conda env: conda env config vars set IEX_CLOUD_API_TOKEN=YOUR_SECRET_TOKEN
IEX_CLOUD_API_TOKEN = os.environ['IEX_CLOUD_API_TOKEN']

# %%
sym = 'AAPL'
base_url = 'https://sandbox.iexapis.com/stable'
endpoint = f'/stock/{sym}/quote'
auth = f'?token={IEX_CLOUD_API_TOKEN}'
req = f'{base_url}/{endpoint}/{auth}'

# %%
resp = requests.get(req)
# %%
