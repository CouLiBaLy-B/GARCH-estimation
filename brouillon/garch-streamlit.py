import streamlit as st
from random import gauss
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import datetime
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# ------- Data
st.header("Data")
data = pd.read_csv("data.csv")
st.write(data.head(8))
data["date"] = data.index
data["year"] = data["date"].astype(str).str.split("-", expand=True)[0]
data = data["Adj Close"].copy()


# ------- GARCH(1,1) Model
st.header("GARCH(1,1) Model")
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

# ------ Simumlated Garch data
# create dataset
n = data.shape[0]

# params
omega = 0.5
alpha_1 = 0.1
beta_1 = 0.3

test_size = int(n * 0.1)

# a0, a1 : initialisation
series = [gauss(0, 1), gauss(0, 1)]

# sigma
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

plt.figure(figsize=(10,4))
plt.plot(series)
plt.title('Simulated GARCH(1,1) Data', fontsize=20)
st.pyplot(plt)

# ------ Data and Volatility
plt.figure(figsize=(10,4))
plt.plot(series)
plt.plot(vols, color='red')
plt.title('Data and Volatility', fontsize=20)
st.pyplot(plt)

# ------ PACF
plot_pacf(np.array(series)**2)
plt.show()
st.pyplot(plt)

# ------ Fit the Garch Model
train, test = series[:-test_size], series[-test_size:]
model = arch_model(train, p=2, q=2)
model_fit = model.fit()
# st.write("**Model Fit**")
# st.write(model_fit)
st.write("**Summary**")
st.write(model_fit.summary())


# ------ Predictions
predictions = model_fit.forecast(horizon=test_size)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions.variance.values[-1, :]))
plt.title('Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

predictions_long_term = model_fit.forecast(horizon=1000)
plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(np.sqrt(predictions_long_term.variance.values[-1, :]))
plt.title('Long Term Volatility Prediction', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

# ------ Rolling Forecast Origin
st.header("Rolling Forecast Origin")
rolling_predictions = []
for i in range(test_size):
    train = series[:-(test_size-i)]
    model = arch_model(train, p=2, q=2)
    model_fit = model.fit(disp='off')
    pred = model_fit.forecast(horizon=1)
    rolling_predictions.append(np.sqrt(pred.variance.values[-1,:][0]))

plt.figure(figsize=(10,4))
true, = plt.plot(vols[-test_size:])
preds, = plt.plot(rolling_predictions)
plt.title('Volatility Prediction - Rolling Forecast', fontsize=20)
plt.legend(['True Volatility', 'Predicted Volatility'], fontsize=16)
st.pyplot(plt)

