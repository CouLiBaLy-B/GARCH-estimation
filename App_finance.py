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
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import yfinance as yf
import math

from pandas.plotting import register_matplotlib_converters
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

register_matplotlib_converters()
sns.set(style='whitegrid', palette='muted', font_scale=1.5)

rcParams['figure.figsize'] = 22, 10

# <== that's all it takes :-)
yf.pdr_override()
np.random.seed(123)
# download dataframe
''' 
# Deep learning et série temporelle : Application a la finance

### Ibra le Sultan
'''

'''
## Importation des données'''
"Les données sont les données de yahoo finance"
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

def compagny():
    ticker = st.selectbox(
             'Lequel voulez vous afficher ?', ('Apple','Accor',
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
    d = ["AAPL",'AC.PA',
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
    comp = ['Apple','Accor',
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

data = pdr.get_data_yahoo("{}".format(Ticker), start=d[0], end=d[1])

"La taille de nos données est : ", data.shape
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

'''# Forecasting with garch model
'''
"Le utilisé ici permet de fait des predictions de la volatilité (la variation) de l'instant suivant (jour, mois, année)"

from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
g = g()
# Selection de nos données d'entrainements
data1 = data.filter([g])
log_returns = np.log(data1[g]) - np.log(data1[g].shift(1))

returns = 100*data1[g].pct_change().dropna()
"Log returns"
fig, ax = plt.subplots()
ax.plot(log_returns)
plt.ylabel('Pct return')
plt.title("{} returns".format(g))
fig
plt.show()

#"Returns"
#fig, ax = plt.subplots()
#ax.plot(returns)
#plt.ylabel('Pct return')
#plt.title("{} returns".format(g))
#fig
#plt.show()
'''#
PACF '''
"L'autocorrelation permet de connaitre les correlations entre deux valeurs de la série à deux instants differents" \
"La significativité de cette autocorrelation permet de trouver les parametres de notre modele GARCH(p,q)"
plt.figure(figsize=(12,6))
fig, ax = plt.subplots()
fig = plot_pacf(log_returns**2)
plt.legend()
fig
plt.show()


def garch():
    p = st.slider('Paramètre P', 1, 10, 4)
    q = st.slider('Paramètre Q', 1, 10, 1)
    test_size = st.slider('Test size', 0.0, 1.0, 0.2,0.01)
    return[p,q, test_size]
garch = garch()

act = st.button("Voulez vous lancer l'apprentissage ?", key=12, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
if act :
        # fit garch
        model = arch_model(log_returns, p =garch[0],q=garch[1])
        model_fit = model.fit()

        #model_fit

        rolling_predictions = []
        test_size = int(garch[2]*data1.shape[0])

        for i in range(test_size):
            train = log_returns[:-(test_size - i)]
            model = arch_model(train, p =garch[0],q=garch[1])
            model_fit = model.fit(disp ="off")
            pred = model_fit.forecast(horizon=1)
            rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

        rolling_predictions = pd.Series(rolling_predictions, index = log_returns.index[-test_size:])

        '''## Prédiction de la volatilité'''
        fig, ax = plt.subplots()
        ax.plot(log_returns[-test_size:])
        ax.plot(rolling_predictions)
        plt.title('Prediction de la volatilité et rolling forecast')
        plt.legend(["Vraie returns","prediction de volatilité"])
        fig
        plt.show()



'''# Forecastiong with LSTM'''

df = data1.values
#la taille de nos d'entrainement

def tail():
    taille = st.slider("La taille du train", max_value=1.0, min_value=0.6, value=0.8, step=0.01)
    return taille
def batch():
    option = st.selectbox(
        'Batch size :',
        (16,32,64))
    return option
batch = batch()
def epoch():
    option = st.slider('Epoch', 0, 100, 1)
    return option
epoch = epoch()
def dense():
    K = st.slider('Densité 1 ', 20, 100, 50)
    d = st.slider('Densité 2 ', 0, 100, 1)
    return[K,d]
dense = dense()
df_train_len = math.ceil(len(df) * tail())
"La taille de notre train est : ", df_train_len
active = st.button("Voulez vous lancer l'apprentissage ?", key=None, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
if active:
        # Centrage des données
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled_df = scaler.fit_transform(df)

        train_df = scaled_df[:df_train_len,:]
        x_train = []
        y_train = []

        for i in range(60, len(train_df)):
            x_train.append(train_df[i-60:i,0])
            y_train.append(train_df[i , 0])

        # convertir en tableau numpy
        x_train = np.array(x_train)
        y_train = np.array(y_train)

        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))
        #y_train

        # Construction de notre model
        model = Sequential()
        model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)))
        model.add(LSTM(50, return_sequences=False))
        model.add(Dense(dense[0]))
        model.add(Dense(dense[1]))

        model.compile(optimizer='adam', loss= 'mean_squared_error')

        # Entrainement de notre model
        model.fit(x_train, y_train, batch_size=batch, epochs=epoch)

        # Création de data_test
        test_df = scaled_df[df_train_len -60:,:]
        x_test = []
        y_test = df[df_train_len:,:]
        for i in range(60, len(test_df)):
            x_test.append(test_df[i-60:i,0])

        # Convertion en tableau numpy
        x_test = np.array(x_test)
        x_test = np.reshape(x_test,(x_test.shape[0],x_test.shape[1],1))

        # Prediction de valeurs
        predictions = model.predict(x_test)
        predictions = scaler.inverse_transform(predictions)

        # Element de qualité : RMSE

        rmse = np.sqrt(np.mean(predictions - y_test)**2)
        "Le RMSE est :", rmse

        #Representation graphique

        train = data[:df_train_len]
        valid = data[df_train_len:]
        valid['Predictions'] = predictions
        #plt.figure(figsize=(16,8))
        fig,ax = plt.subplots()
        ax.plot(train[g])
        ax.plot(valid[[g,'Predictions']])
        plt.title('Model LSTM')
        plt.xlabel("Date")
        plt.ylabel(g+" Prices")
        plt.legend(['Train ','Validation','Predictions'])
        fig
        plt.show()
        # les données de validation et la prediction
        valid

        # recuperation
        #apple_quote = pdr.get_data_yahoo(Ticker, start=d[0], end=d[1])
        # creation de nouveau



'''## Forecasting avec fbprophet
'''
from fbprophet import Prophet

acti = st.button("Voulez vous lancer l'apprentissage ?", key=113, help=None, on_click=None, args=None, kwargs=None, type="secondary", disabled=False)
if acti :
            Data2 = data[g]
            Data2 = Data2.reset_index()

            Data2.columns = ['ds','y']
            Data2['ds'] = Data2['ds'].apply(lambda x: x.strftime("%Y-%m-%d"))
            #Data2

            prophet = Prophet(daily_seasonality=False)
            prophet.fit(Data2)


            future_date = prophet.make_future_dataframe(periods = 30)

            prediction = prophet.predict(future_date)

            from fbprophet.plot import plot_plotly

            fig, ax = plt.subplots()
            fig = plot_plotly(prophet,prediction)
            fig
            plt.show()

            incounnu = Data2.iloc[-30:]
            df = Data2.iloc[:-30]

            prophet = Prophet(daily_seasonality=True)
            prophet.fit(df)

            future_dates = prophet.make_future_dataframe(periods=365)
            predictions = prophet.predict(future_dates)

            fig, ax = plt.subplots()
            fig = plot_plotly(prophet, predictions)
            fig
            plt.show()

            plt.figure(figsize=(10,6))

            pred = predictions[predictions['ds'].isin(incounnu['ds'])]
            pred.shape
            incounnu.shape

            fig, ax = plt.subplots()
            ax.plot(pd.to_datetime(incounnu['ds']), incounnu['y'], label= "Actuelle")
            ax.plot(pd.to_datetime(incounnu['ds']), pred['yhat'], label= "Predictions")

            plt.legend()
            fig
            plt.show()


'''# Detection d'anomalie dans les données
'''
data1 = data.filter([g])

def train_size():
    option = st.slider('Train size', 0.5, 1.0, 0.95,0.01)
    return option
train_siz = train_size()
train_size = int(len(data1) * train_siz)
test_size = len(data1) - train_size
train, test = data1.iloc[0:train_size], data1.iloc[train_size:len(data1)]
"La taille de notre est : ",train.shape, test.shape

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler = scaler.fit(train[[g]])

train[g] = scaler.transform(train[[g]])
test[g] = scaler.transform(test[[g]])


def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)


TIME_STEPS = 30

# reshape to [samples, time_steps, n_features]

X_train, y_train = create_dataset(train[[g]], train[g], TIME_STEPS)
X_test, y_test = create_dataset(test[[g]], test[g], TIME_STEPS)

X_train.shape

model = keras.Sequential()
model.add(keras.layers.LSTM(
    units=64,
    input_shape=(X_train.shape[1], X_train.shape[2])
))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.RepeatVector(n=X_train.shape[1]))
model.add(keras.layers.LSTM(units=64, return_sequences=True))
model.add(keras.layers.Dropout(rate=0.2))
model.add(keras.layers.TimeDistributed(keras.layers.Dense(units=X_train.shape[2])))
model.compile(loss='mae', optimizer='adam')

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,
    shuffle=False
)
fig, ax = plt.subplots()
ax.plot(history.history['loss'], label='train')
ax.plot(history.history['val_loss'], label='test')
plt.legend()
fig
plt.show()


X_train_pred = model.predict(X_train)

train_mae_loss = np.mean(np.abs(X_train_pred - X_train), axis=1)
fig, ax = plt.subplots()
ax = sns.distplot(train_mae_loss, bins=50, kde=True)

fig
plt.show()
"Ibra"
X_test_pred= model.predict(X_test)
#X_test_pred = model.predict(X_test)
"ibra"
test_mae_loss = np.mean(np.abs(X_test_pred - X_test), axis=1)
test_rmae_loss = np.mean((X_test_pred - X_test)**2, axis=1)
THRESHOLD = 0.05

test_score_df = pd.DataFrame(index=test[TIME_STEPS:].index)
test_score_df['loss_norme_1'] = test_mae_loss
test_score_df['loss_norme_2'] = test_rmae_loss
test_score_df['threshold'] = THRESHOLD
test_score_df['anomaly'] = test_score_df.loss_norme_2 > test_score_df.threshold
#test_score_df['anomaly']  = test_score_df['anomaly'].astype(bool)
test_score_df["True_value"] = test[TIME_STEPS:][g]

fig, ax =  plt.subplots()
ax.plot(test_score_df.index, test_score_df.loss_norme_1, label='loss_norme_1')
ax.plot(test_score_df.index, test_score_df.loss_norme_2, label='loss_norme_2')
ax.plot(test_score_df.index, test_score_df.threshold, label='threshold')
plt.xticks(rotation=25)
plt.legend()
fig
plt.show()

anomalies = test_score_df[test_score_df.anomaly == True]
anomalies

if anomalies.shape[0] > 0 :
        fig, axx = plt.subplots()
        plt.plot(
          test[TIME_STEPS:].index,
          scaler.inverse_transform(test[TIME_STEPS:][[g]]),
          label='close price'
        )

        fig1, ax = plt.subplots()

        ax = sns.scatterplot(
          anomalies.index,
          scaler.inverse_transform(anomalies.True_value),
          color=sns.color_palette()[3],
          s=52,
          label='anomaly'
        )
        plt.xticks(rotation=25)
        plt.legend()
        fig
        fig1
        plt.show()