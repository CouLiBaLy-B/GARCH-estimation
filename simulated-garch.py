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
        alpha_1 = st.number_input("alpha 1", min_value=0.0, max_value=100.0, value=0.5)
    with col3:
        beta_1 = st.number_input("beta 1", min_value=0.0, max_value=100.0, value=0.5)
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

"$a_0$ = ", series[0] ,"$ \sim \mathcal{N}(0, 1)$"
"$a_1$ = ",series[1] ,"$\sim \mathcal{N}(0, 1)$"


fig, ax = plt.subplots()
ax.plot(series)
plt.title("Simulation des données d'un modèle GARCH(1,1)", fontsize=20)
fig2 = mpl_to_plotly(fig)
fig2
plt.show()

"## Données et volatilité"

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
plt.legend(['Vraie volatilité', 'Volatilité prédite'], fontsize=16)
st.pyplot(plt)

predictions_long_term = model_fit.forecast(horizon=1000)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
plt.title('Long Term Volatility Prediction', fontsize=20)
plt.legend(['Vraie volatilité', 'Volatilité prédite'], fontsize=16)
st.pyplot(plt)

"## Prédictions roulantes"
####


rolling_predictions = []
#Rolling = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=p, q=q)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    print(pred)
    #Rolling.append(pred.variance.values)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))


#Rolling
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Rolling Forecast', fontsize=20)
plt.legend(['Vraie volatilité', 'Volatilité prédite'], fontsize=16)
st.pyplot(plt)

#####
