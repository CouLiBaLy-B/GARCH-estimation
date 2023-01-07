from pandas_datareader import data as pdr
import numpy as np
import pandas as pd
import networkx as nx
from networkx.generators.random_graphs import erdos_renyi_graph as nxe
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf
import datetime
import requests
import warnings

warnings.filterwarnings("ignore")
import plotly.express as px
import streamlit as st
from networkx.generators.random_graphs import fast_gnp_random_graph as nxf

import yfinance as yf

# <== that's all it takes :-)
yf.pdr_override()

# download dataframe
''' 
# Projet de série temporelle

### COULIBALY Bourahima
### OSMAN Ahmed
'''

'''
## Importation des données'''
"Les données sont les données d'apple"
def date():
    i = 1
    while i == 1:
        start = st.date_input(
    "Start time",
    datetime.date(2019, 7, 6),key=1)

        end = st.date_input(
    "End time",
    datetime.date(2021, 7, 6),key=2)
        if start <= end:
            i = 0
    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]

d = date()
data = pdr.get_data_yahoo("AAPL", start=d[0], end=d[1])
data


''' Plotting
'''
fig, ax = plt.subplots()
fig = px.line(data,x = data.index, y = data.columns,
              title = "La variation des differentes sorties"
              )
#fig.select_yaxes

fig
plt.show()


def g():
    option = st.selectbox(
        'Lequel voulez vous afficher ?',
        ('Open','Close','High','Low','Adj Close','Volume'))
    return option


''' 
# Autocorrelation 
'''
x = data[g()]
fig, ax = plt.subplots()
fig = plot_acf(x, lags=50)
fig
plt.show()


'''
## Calibrer un modèle GARCH(1, 1) avec EMV
Une fois que nous avons décidé que les données pourraient avoir un modèle GARCH (1, 1) sous-jacent, nous aimerions calibrer le modèle GARCH (1, 1) aux données en estimant ses paramètres.

Pour ce faire, nous avons besoin de la fonction log-vraisemblance

$${\mathcal{L}( \theta) = \sum_{t=1}^T - \ln \sqrt{2\pi} - \frac x_t^2 2\sigma_t^2 - \frac 1 2 \ln(\sigma_t^2)}$$


Pour évaluer cette fonction, nous avons besoin de $x_t$ et ${\sigma}_t$ pour $1\le t \le T$. Nous avons 
$ x_t $, mais nous devons calculer ${\sigma}_t$. Pour faire cela, nous devons trouver une valeur pour 
${\sigma}_1$. Notre hypothèse sera $\sigma_1^2 = \hat E [x_t^2]$. Une fois que nous avons notre condition initiale, nous calculons le reste des $\sigma$ en utilisant l'équation


$$\sigma_t^2 = a_0 + a_1 x_{t-1}^2 + b_1\sigma_{t-1}^2$$
'''


def simulate_GARCH(T, a0, a1, b1, sigma1):
    # Initialize our values
    X = np.ndarray(T)
    sigma = np.ndarray(T)
    sigma[0] = sigma1

    for t in range(1, T):
        X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)

        sigma[t] = math.sqrt(a0 + b1 * sigma[t - 1] ** 2 + a1 * X[t - 1] ** 2)

    X[T - 1] = sigma[T - 1] * np.random.normal(0, 1)

    return X, sigma