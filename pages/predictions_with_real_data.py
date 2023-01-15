import streamlit as st
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
def data(Ticker , d ):
    data = pdr.get_data_yahoo("{}".format(Ticker), start=d[0], end=d[1])
    return data

data = data(Ticker = Ticker, d = d)
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




def g():
    option = st.selectbox(
        'Lequel voulez vous utiliser pour la prédiction ?',
        ('Adj Close','Open','Close','High','Low','Volume'))
    return option

g = g()

data = data.filter([g])

f"## Données {comp}: {g}"

"###  Calibrage et apprentissage du modèle - GARCH(1, 1)"

series = data.values
X_train, X_test = train_test_split(data, test_size = 0.2, shuffle= False)

series = data.values
#train, test = series[:-test_size], series[-test_size:]
train, test = X_train.values, X_test.values
p, q = 1, 1

returns =  np.log(data).diff().dropna()

model = arch_model(returns, p=p, q=q)
model_fit = model.fit()
#
st.write("**Summary**")
st.write(model_fit.summary())

'## Le log returns '

fig, ax= plt.subplots()
ax.plot(returns, color="red")
fig2 = mpl_to_plotly(fig)
fig2
plt.show()

##


"## Prédictions roulantes"

def horizon():
    ho = st.number_input("L'horizon de la prédiction : ", min_value=1, max_value= 100, value= 1, key ='ho')
    return ho

horizon = horizon()

test_size = int(data.shape[0] * 0.2)
rolling_predictions = []
for i in range(test_size):
    train = returns[:-(test_size-i)]
    model = arch_model(train, p=p, q=q, vol = "Garch", dist="Normal")
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=horizon)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


rolling  = pd.DataFrame(rolling_predictions, columns= ["pred"], index= returns[-test_size:].index)
rolling["test"] = returns[-test_size:]

fig, ax = plt.subplots()
fig = px.line(rolling,x = rolling.index, y = ["pred","test"],
              title = "La variation des differentes sorties"
              )


fig
plt.show()
