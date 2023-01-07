import streamlit as st
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import datetime
from arch import arch_model
from sklearn.model_selection import train_test_split
from arch.__future__ import reindexing
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import plotly.express as px
import yfinance as yf
import plotly.graph_objects as go
from PIL import Image
from plotly.tools import mpl_to_plotly
yf.pdr_override()
np.random.seed(42)

"#  Projet Garch"


"# Modèle GARCH(1,1)"
st.write(r"""
$$
a_t = \varepsilon_t \sqrt
{\omega + \alpha_1
a_
{t - 1} ^ 2 +
\beta_1 \sigma_{t - 1} ^ 2}
$$

$$
a_0, a_1 \sim \mathcal
{N}(0, 1)
$$

$$
\sigma_0 = 1, \sigma_1 = 1
$$

$$
\varepsilon_t \sim \mathcal
{N}(0, 1)
$$
""")

"# Simumlated Garch data"
# create dataset

" Paramètres du modéle"
def parametre():
    col1, col2, col3 = st.columns(3)
    with col1:
        omega = st.number_input("Omega", min_value= 0.0, max_value=100.0, value=0.5)
    with col2:
        alpha_1 = st.number_input("alpha_1", min_value=0.0, max_value=100.0, value=0.5)
    with col3:
        beta_1 = st.number_input("beta_A", min_value=0.0, max_value=100.0, value=0.5)
    return omega, alpha_1,beta_1

omega, alpha_1, beta_1 = parametre()

n = 1000
@st.experimental_memo
def serie(n):
    series = [gauss(0, 1), gauss(0, 1)]
    vols = [1, 1]
    for _ in range(n):
        new_vol = np.sqrt(
            omega +
            alpha_1 * series[-1] ** 2 +
            beta_1 * vols[-1] ** 2
        )
        new_val = gauss(0, 1) * new_vol

        vols.append(new_vol)
        series.append(new_val)
    return vols, series

vols, series = serie(n)

"a0 = ", series[0] ,"$ \sim \mathcal{N}(0, 1)$"
"a1 = ",series[1] ,"$\sim \mathcal{N}(0, 1)$"


fig, ax = plt.subplots()
ax.plot(series)
plt.title("Simulation des données d'un modèle GARCH(1,1)", fontsize=20)
fig2 = mpl_to_plotly(fig)
fig2
plt.show()

"## Données et volatilité"
# plt.figure(figsize=(10,4))
fig, ax = plt.subplots()
ax.plot(series)
ax.plot(vols, color='red')
plt.title('Données et volatilité', fontsize=20)
fig2 = mpl_to_plotly(fig)
fig2
plt.show()


"##  PACF"
plot_pacf(np.array(series)**2)
plt.show()
st.pyplot(plt)

"##  Calibrage et apprentissage du modèle"
def p_q_choices():
    col1, col2 = st.columns(2)
    with col1:
        p = st.number_input("Paramètre p", min_value=0, max_value=30, value=5)
    with col2:
        q = st.number_input("Paramètre q", min_value=0, max_value=30, value=5)

    return p, q

test_size = int(n * 0.1)
train, test = series[:-test_size], series[-test_size:]
p, q = p_q_choices()
model = arch_model(train, p=p, q=q)
model_fit = model.fit()
# st.write("**Model Fit**")
# st.write(model_fit)
st.write("**Summary**")
st.write(model_fit.summary())


"## Prédictions"
predictions = model_fit.forecast(horizon=test_size)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
plt.title('Prédiction de la volatilité', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

predictions_long_term = model_fit.forecast(horizon=1000)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
plt.title('Long Term Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

"## Prédictions roulantes"

rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=p, q=q)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

#####
"# Application aux données réelles de la finance"


'''## Importation des données'''
"Les données sont les données de yahoo finance"

def date():
    i = 1
    while i == 1:
        col1, col2 = st.columns(2)
        with col1:
            start = st.date_input("Date de début", datetime.date(2019, 7, 6),key=1)
        with col2:
            end = st.date_input("Date de fin",datetime.date(2021, 7, 6),key=2)
        if start <= end:
            i = 0

    return [start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]

def compagny():
    ticker = st.selectbox(
             'Lequel voulez vous selectionner ?', ('CAC40','Apple','Accor',
             'Airbus SE',
             'Air Liquide S.A',
             'ArcelorMittal',
             'Atos SE',
             'Axa',
             'BNP Paribas',
             'Bouygues',
             'Capgemini',
             'Carrefour',
             'Compagnie de Saint-Gobain S.A.',
             'Credit Agricole S.A.',
             'Danone',
             'Dassault Systemes SA',
             'Engie',
             'EssilorLuxottica',
             'Hermes International',
             'Kering',
             'Legrand SA',
             "L'Oreal",
             'Lvmh Moet Hennessy Vuitton SE',
             'Michelin (CGDE)-B',
             'Orange.',
             'Pernod Ricard',
             'Publicis Groupe SA',
             'Renault S.A.',
             'Safran SA',
             'Sanofi',
             'Schneider Electric SE',
             'Societe Generale S.A.'))
    return ticker

d = date()
comp = compagny()

def tickers(a):
    d = ["^FCHI","AAPL",'AC.PA',
         'AIR.PA',
         'AI.PA',
         'MT.PA',
         'ATO.PA',
         'CS.PA',
         'BNP.PA',
         'EN.PA',
         'CAP.PA',
         'CA.PA',
         'SGO.PA',
         'ACA.PA',
         'BN.PA',
         'DSY.PA',
         'ENGI.PA',
         'EL.PA',
         'RMS.PA',
         'KER.PA',
         'LR.PA',
         'OR.PA',
         'MC.PA',
         'ML.PA',
         'ORA.PA',
         'RI.PA',
         'PUB.PA',
         'RNO.PA',
         'SAF.PA',
         'SAN.PA',
         'SU.PA',
         'GLE.PA']
    comp = ["CAC40",'Apple','Accor',
             'Airbus SE',
             'Air Liquide S.A',
             'ArcelorMittal',
             'Atos SE',
             'Axa',
             'BNP Paribas',
             'Bouygues',
             'Capgemini',
             'Carrefour',
             'Compagnie de Saint-Gobain S.A.',
             'Credit Agricole S.A.',
             'Danone',
             'Dassault Systemes SA',
             'Engie',
             'EssilorLuxottica',
             'Hermes International',
             'Kering',
             'Legrand SA',
             "L'Oreal",
             'Lvmh Moet Hennessy Vuitton SE',
             'Michelin (CGDE)-B',
             'Orange.',
             'Pernod Ricard',
             'Publicis Groupe SA',
             'Renault S.A.',
             'Safran SA',
             'Sanofi',
             'Schneider Electric SE',
             'Societe Generale S.A.']
    ind = comp.index(a)
    return d[ind]

Ticker = tickers(comp)

"le ticher de la compagnie selectionner est : ", Ticker
@st.experimental_memo
def data():
    data = pdr.get_data_yahoo("{}".format(Ticker), start=d[0], end=d[1])
    return data
data = data()
"La taille de nos données est : ", data.shape
data
data.reset_index(inplace = True)


'''### Graphe de l'actif financier
'''

fig = go.Figure(data=[go.Candlestick(x=data["Date"],
                open=data["Open"],
                high=data['High'],
                low=data['Low'],
                close=data['Close'])])


fig
plt.show()

data.set_index("Date", inplace=True)

Date_data = data.reset_index()
Date = Date_data["Date"]
Date



def g():
    option = st.selectbox(
        'Lequel voulez vous utiliser pour la prédiction ?',
        ('Adj Close','Open','Close','High','Low','Volume'))
    return option

g = g()

data = data.filter([g])

series = Date_data[["Date", g]].set_index("Date")
series

f"###  PACF des données {comp}: {g}"
plot_pacf(np.array(series[g].values)**2)
plt.show()
st.pyplot(plt)


"##  Calibrage et apprentissage du modèle"
def pv_qv_choices():
    col1, col2 = st.columns(2)
    with col1:
        p = st.number_input("Paramètre p", min_value=0, max_value=30, value=5, key = "pv")
    with col2:
        q = st.number_input("Paramètre q", min_value=0, max_value=30, value=5, key = "qv")

    return p, q

test_size = int(data.shape[0] * 0.1)

X_train, X_test = train_test_split(data, test_size = 0.1, shuffle= False)
X_test
X_train


#series = data.values
#train, test = series[:-test_size], series[-test_size:]
train, test = X_train.values, X_test.values
p, q = pv_qv_choices()
model = arch_model(train, p=p, q=q)
model_fit = model.fit()
# st.write("**Model Fit**")
# st.write(model_fit)
st.write("**Summary**")
st.write(model_fit.summary())

'## Données'
fig, ax= plt.subplots()
ax.plot(series)
# fig
fig2 = mpl_to_plotly(fig)
fig2
plt.show()

'## Vraie volatilité sur la periode'

Vol = data[g].pct_change().dropna()

fig, ax= plt.subplots()
ax.plot(Vol, color="red")
# fig
fig2 = mpl_to_plotly(fig)
fig2
plt.show()


# "## Données et volatilité"
# # plt.figure(figsize=(10,4))
# fig, ax = plt.subplots()
# ax.plot(series)
# ax.plot(Vol, color='red')
# plt.title('Données et volatilité', fontsize=20)
# fig2 = mpl_to_plotly(fig)
# fig2
# plt.show()


##

vols = Vol




"## Prédictions roulantes"

rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=p, q=q)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

plt.figure(figsize=(10,4))
true, = plt.plot(X_test.index(),vols[-test_size:])
preds, = plt.plot(X_test.index(),rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)


