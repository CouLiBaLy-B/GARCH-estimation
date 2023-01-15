import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas_datareader import data as pdr
import datetime
from dateutil.relativedelta import relativedelta
from functools import partial
import math
import numpy as np
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats.stattools import jarque_bera
from sklearn.model_selection import train_test_split
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from arch import arch_model
from arch.__future__ import reindexing
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
        SixAgo = start + relativedelta(months=-6)
    return [SixAgo.strftime("%Y-%m-%d"), start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")]

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
             'Societe Generale S.A.'), key = f'{np.random.rand()}')
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
    data = pdr.get_data_yahoo("{}".format(Ticker), start=d[1], end=d[2])
    dfAgo = pdr.get_data_yahoo("{}".format(Ticker), start=d[0], end=d[1])
    return data, dfAgo

data, dfAgo = data(Ticker = Ticker, d = d)
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
data, dfAgo = data.filter([g]), dfAgo.filter([g])

f"## Données {comp}: {g}"


Data = 100* np.log(data/data.shift(1)).dropna(axis = 0)
series = Data.values

"Log returns"
fig, ax= plt.subplots()
ax.plot(Data, color="red")
fig2 = mpl_to_plotly(fig)
fig2
plt.show()


r"""
## Calibrer un modèle GARCH(1, 1) avec EMV

Pour ce faire le calibrage de notre modèle dont les paramètres vérifies :
$$
\left(\hat{\omega}_n, \hat{\alpha}_n, \hat{\beta}_n\right)=\operatorname{argmax}_{(\omega, \alpha, \beta)} L_n(\omega, \alpha, \beta ; \text { data })
$$
Nous avons besoin de la fonction log-vraisemblance définie par : 

$$
\mathcal{L}(\theta) = \sum_{t=1}^T - \ln \sqrt{2\pi} - \frac{s_t^2}{2\sigma_t^2} - \frac{1}{2}\ln(\sigma_t^2)
$$

Pour évaluer cette fonction, nous avons besoin de $s_t$ et ${\sigma}_t$ pour $1\le t \le T$. Nous avons 
$ s_t $, mais nous devons calculer ${\sigma}_t$. Pour faire cela, nous devons trouver une valeur pour 
${\sigma}_1$. 

Notre hypothèse sera $\sigma_1^2 = \hat E [s_t^2]$ la moyenne des 6 dernier mois avant. Une fois que nous avons notre condition initiale, 
nous calculons le reste des $\sigma$ en utilisant l'équation


$$\sigma_t^2 = \omega + \alpha s_{t-1}^2 + \beta\sigma_{t-1}^2$$
"""
#dfAgo = np.log(dfAgo/dfAgo.shift(1)).dropna(axis = 0)
sigma21 = np.sqrt(np.mean( data.values ** 2))

# calcul de sigma 2
def compute_squared_sigmas(X, initial_sigma, theta):
    a0 = theta[0]
    a1 = theta[1]
    b1 = theta[2]
    T = len(X)
    sigma2 = np.ndarray(T)
    sigma2[0] = initial_sigma ** 2
    for t in range(1, T):
        sigma2[t] = a0 + a1 * X[t - 1] ** 2 + b1 * sigma2[t - 1]
    return sigma2


#"Regardons les sigmas que nous venons de simuler."

sigma2 = np.sqrt(compute_squared_sigmas(series, sigma21,[.5,0.5, 0.5]))
#sigma2

r"""
Maintenant que nous pouvons calculer les $\sigma_t$, nous allons définir la fonction de vraisemblance. 
Cette fonction prendra en entrée nos observations $s$ et $\theta$ et retournera $-\mathcal {L}(\theta)$.
Notez que nous re-calculons constamment les $\sigma_t$ dans cette fonction.
"""

def negative_log_likelihood(X, theta):
    T = len(X)
    # Estimate initial sigma squared
    initial_sigma = np.sqrt(np.mean(X ** 2))
    # Generate the squared sigma values
    sigma2 = compute_squared_sigmas(X, initial_sigma, theta)
    # Now actually compute
    return -sum(
        [-np.log(np.sqrt(2.0 * np.pi)) -
         (X[t] ** 2) / (2.0 * sigma2[t]) -
         0.5 * np.log(sigma2[t])
         for t in range(T)]
    )

r"""
Maintenant on optimise numériquement 
$$\hat\theta = \arg \max_{(\omega, \alpha, \beta)}\mathcal{L}(\theta) = \arg \min_{(\omega, \alpha, \beta)}-\mathcal{L}(\theta)$$

Sous les contraintes  suivantes : $$ \omega \geq 0, \beta < 1, \alpha + \beta < 1 $$ (c'est contraintes permets d'assurées le caractère stationnaire du modèle)
"""

# Make our objective function by plugging X into our log likelihood function
objective = partial(negative_log_likelihood, series)

# Define the constraints for our minimizer
def constraint1(theta):
    return 1 - (theta[1] + theta[2])

def constraint2(theta):
    return theta[0]

def constraint3(theta):
    return 1 - theta[2]

def constraint4(theta):
    return theta[1]

def constraint5(theta):
    return theta[2]


cons = ({'type': 'ineq', 'fun': constraint1},
        {'type': 'ineq', 'fun': constraint2},
        {'type': 'ineq', 'fun': constraint3},
        {'type': 'ineq', 'fun': constraint4},
        {'type': 'ineq', 'fun': constraint5}
        )

# contrainte >=0

# Actually do the minimization
result = scipy.optimize.minimize(objective, (1.0, .1, 0.1),
                        method='SLSQP',
                        constraints = cons)

"#### Les estimateurs du maximum de vraisemblance"
theta_mle = result.x

estimateurMLE = pd.DataFrame(theta_mle, index= ["omega", "alpha", "beta"])
estimateurMLE


"""
## Prédire le future
Maintenant que nous avons calibré le modèle à nos observations, nous aimerions pouvoir prédire à quoi ressemblera 
la volatilité future. Pour ce faire, nous pouvons simplement simuler plus de valeurs en utilisant la définition du modèle et les paramètres estimés par MV.
"""

## Fonction de simulation GARCH

def GARCH(T, omega, alpha, beta, sigma1):
    X = np.ndarray(T)
    sigma = np.ndarray(T)
    sigma[0] = sigma1
    for t in range(1,T):
        X[t - 1] = sigma[t - 1] * np.random.normal(0, 1)
        sigma[t] = np.sqrt(omega + alpha * sigma[t - 1] ** 2 + beta * X[t - 1] ** 2)
    X[T-1] = sigma[T-1] * np.random.normal(0,1)
    return X, sigma
##

sigma_2 = compute_squared_sigmas(series, np.sqrt(np.mean(series ** 2)), [theta_mle[0],theta_mle[1],theta_mle[2]] )
col1, col2, col3, col4 = st.columns(4)
with col1 :
    initial_sigma = sigma_2[-1]
    " Sigma initial"
    initial_sigma
with col2 :
    "Omega"
    omega_estimate = theta_mle[0]
    omega_estimate
with col3 :
    "Alpha "
    alpha_estimate = theta_mle[1]
    alpha_estimate
with col4 :
    "beta "
    beta_estimate = theta_mle[2]
    beta_estimate

def horizon():
    hor = st.number_input("Horizon de prédiction " , min_value= 1, max_value= 1000, value=100)
    return hor

horizon  = horizon()
horizon = int(horizon)
X_forecast, sigma_forecast = GARCH(horizon, omega_estimate, alpha_estimate, beta_estimate, initial_sigma)
#X_forecast

fig, ax = plt.subplots()
ax.plot(Data.index[-100:], series[-100:], 'b-')
ax.plot(Data.index[-100:], sigma_2[-100:], 'r-')
ax.plot([Data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], X_forecast, 'b--')
ax.plot([Data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], sigma_forecast, 'r--')
plt.xlabel('Time')
plt.legend(['Log-return', 'sigma', "pred log-return", "pred-sigma"])
fig2 = mpl_to_plotly(fig)
fig2
plt.show()






"Pour s'assuré que notre modèle fonction bien, nous allons estimé les paramètres avec la libraire arch de python avant de continuer notre étude"

"### GARCH(1, 1) - avec la librarie arch"

series = data.values
X_train, X_test = train_test_split(data, test_size = 0.2, shuffle= False)

series = data.values
#train, test = series[:-test_size], series[-test_size:]
train, test = X_train.values, X_test.values
p, q = 1, 1

returns =  100*np.log(data).diff().dropna()

model = arch_model(returns, p=p, q=q)
model_fit = model.fit()
#
if st.checkbox("Afficher le summary", value = False):
       st.write("**Summary**")
       st.write(model_fit.summary())
"Vu les paramètres estimés avec arch et notre estimation, nous pouvons dire que que le notre modèle estime bien les paramètres."

"### Prédictions roulantes (pas nécessaire)"
horizon = 1
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



"""# Comparaison de la volatilité avec  pour une autre composante du CAC40"""

@st.experimental_memo
def Data(Ticker , d ):
    data = pdr.get_data_yahoo("{}".format(Ticker), start=d[1], end=d[2])
    return data

comp = compagny()
Ticke = tickers(comp)
dataC = Data(Ticker = Ticke, d = d)
dataC = dataC.filter([g])
DataC = 100* np.log(dataC/dataC.shift(1)).dropna(axis = 0)
seriesC = DataC.values
objective = partial(negative_log_likelihood, seriesC)

resultC = scipy.optimize.minimize(objective, (1.0, .1, 0.1),
                        method='SLSQP',
                        constraints = cons)

"#### Les estimateurs du maximum de vraisemblance"
theta_mleC = resultC.x

sigma_2C = compute_squared_sigmas(seriesC, np.sqrt(np.mean(seriesC ** 2)), theta_mleC )
col1, col2, col3, col4 = st.columns(4)
with col1 :
    initial_sigmaC = sigma_2C[-1]
    " Sigma initial"
    initial_sigmaC
with col2 :
    "Omega"
    omega_estimateC = theta_mleC[0]
    omega_estimateC
with col3 :
    "Alpha "
    alpha_estimateC = theta_mleC[1]
    alpha_estimateC
with col4 :
    "beta "
    beta_estimateC = theta_mleC[2]
    beta_estimateC

# Prediction et representation graphique

X_forecastC, sigma_forecastC = GARCH(horizon, omega_estimateC, alpha_estimateC, beta_estimateC, initial_sigmaC)


fig, ax = plt.subplots()
ax.plot(data.index[-100:], series[-100:], 'b-')
ax.plot(data.index[-100:], sigma_2[-100:], 'r-')
ax.plot([data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], X_forecast, 'b--')
ax.plot([data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], sigma_forecast, 'r--')
ax.plot(data.index[-100:], seriesC[-100:], 'b-')
ax.plot(data.index[-100:], sigma_2C[-100:], 'r-')
ax.plot([data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], X_forecastC, 'g--')
ax.plot([data.index[-1] + relativedelta(days=i) for i in range(0, horizon)], sigma_forecastC, 'y--')
plt.xlabel('Time')
plt.legend([f'Log-return {Ticker} ', f'sigma {Ticker}', f"pred log-return {Ticker}", f"pred-sigma {Ticker}", f'Log-return {Ticke}', f'sigma {Ticke}', f"pred log-return {Ticke}", f"pred-sigma {Ticke}"])
fig2 = mpl_to_plotly(fig)
fig2
plt.show()



