import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml

st.header("EL MEHDI - IAENG")
st.title("TP LDA : Projection 1D (Version MNIST 28x28)")
st.write("Chargement du vrai dataset MNIST et calcul de la projection 1D *From Scratch*.")

# 1. Chargement des données avec Cache pour Streamlit (indispensable pour 28x28)
@st.cache_data
def load_mnist_784():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

X, y = load_mnist_784()

# Filtrage strict pour ne garder que les chiffres 0 et 1
masque = (y == 0) | (y == 1)
X_filtre = X[masque] / 255.0  # Normalisation pour 28x28
y_filtre = y[masque]

# 2. Séparation des classes et probabilités a priori
X_0 = X_filtre[y_filtre == 0]
X_1 = X_filtre[y_filtre == 1]

P_0 = len(X_0) / len(X_filtre)
P_1 = len(X_1) / len(X_filtre)

# 3. Calcul des Moyennes et de la Covariance Partagée (784 dimensions)
mu_0 = np.mean(X_0, axis=0)
mu_1 = np.mean(X_1, axis=0)

cov_0 = np.cov(X_0, rowvar=False)
cov_1 = np.cov(X_1, rowvar=False)

# Régularisation à 1e-4 adaptée aux 784 dimensions
Sigma = (cov_0 + cov_1) / 2 + 1e-4 * np.eye(784)

# 4. Calcul du vecteur de Fisher (w) et du seuil (w0)
w = np.linalg.solve(Sigma, mu_1 - mu_0)
w_0 = 0.5 * np.dot(w.T, (mu_1 + mu_0)) - np.log(P_1 / P_0)

st.success("Calculs LDA terminés avec succès !")

# 5. Projection 1D et Visualisation spécifique à Streamlit
projections = np.dot(X_filtre, w)

fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(projections[y_filtre == 0], bins=40, alpha=0.6, color='blue', label='Chiffre 0 (28x28)')
ax.hist(projections[y_filtre == 1], bins=40, alpha=0.6, color='red', label='Chiffre 1 (28x28)')
ax.axvline(x=w_0, color='black', linestyle='--', linewidth=2.5, label=f'Frontière (w_0 = {w_0:.2f})')

ax.set_title("Projection LDA 1D From Scratch (MNIST 28x28)")
ax.set_xlabel("Valeur de la projection 1D")
ax.set_ylabel("Nombre d'images")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# Commande obligatoire pour Streamlit à la place de plt.show()
st.pyplot(fig)
