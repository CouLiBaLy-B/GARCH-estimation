"""## Estimation du modèle GARCH & prévision de la volatilité

La première étape consiste à estimer le modèle GARCH (1,1) suivant, en utilisant la méthode de la (pseudo) log-vraisemblance,

$$
\begin{aligned}
\epsilon_t & =\sigma_t Z_t, \quad\left(Z_t\right)_t \text { i.i.d. } \mathcal{N}(0,1) \\
\sigma_t^2 & =\omega+\alpha \epsilon_{t-1}^2+\beta \sigma_{t-1}^2 .
\end{aligned}
$$

Pour le jeu de données, on peut choisir la série temporelle des log-retours quotidiens du CAC 40, sur une période de plusieurs années. 
En utilisant ce modèle, prédire la volatilité du CAC 40 pour les jours à venir (arrêter l'horizon temporel de la prédiction lorsqu'il est clair que les données passées n'ont pas de pouvoir prédictif au-delà de cet horizon).

En utilisant (presque) le même code, tracez la volatilité quotidienne estimée pour le CAC au cours de l'année 2020. 
Comparez avec la volatilité, calculée de la même manière, d'un actif du secteur de la santé (il est possible de prendre "CAC Santé Indice")."""
