import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

st.title("TP LDA : Frontière de Décision 2D (0 vs 1)")
st.write("Implémentation LDA *From Scratch* avec visualisation de la frontière de décision.")

# 1. Chargement des données
@st.cache_data
def load_data():
    digits = load_digits()
    return digits.data / 16.0, digits.target

X, y = load_data()

# Filtrage strict pour les chiffres 0 et 1
masque = (y == 0) | (y == 1)
X_filtre = X[masque]
y_filtre = y[masque]

X_0 = X_filtre[y_filtre == 0]
X_1 = X_filtre[y_filtre == 1]

# 2. Sélection des 2 meilleurs pixels selon le critère de Fisher basique
diff_moyennes = np.abs(np.mean(X_0, axis=0) - np.mean(X_1, axis=0))
les_2_pixels = np.argsort(diff_moyennes)[-2:]

X_2D = X_filtre[:, les_2_pixels]
X_0_2D = X_0[:, les_2_pixels]
X_1_2D = X_1[:, les_2_pixels]

# 3. Calculs LDA Mathématiques (Moyennes, Covariance partagée, Règle de Bayes)
mu_0 = np.mean(X_0_2D, axis=0)
mu_1 = np.mean(X_1_2D, axis=0)

cov_0 = np.cov(X_0_2D, rowvar=False)
cov_1 = np.cov(X_1_2D, rowvar=False)
Sigma = (cov_0 + cov_1) / 2 + 1e-5 * np.eye(2) # Régularisation

P_0 = len(X_0_2D) / len(X_2D)
P_1 = len(X_1_2D) / len(X_2D)

# Vecteur de Fisher (w) et Seuil (w0)
w = np.linalg.solve(Sigma, mu_1 - mu_0)
w_0 = 0.5 * np.dot(w.T, (mu_1 + mu_0)) - np.log(P_1 / P_0)

st.success("Calculs LDA terminés avec succès !")

# 4. Affichage des résultats mathématiques en bref
st.subheader("Paramètres du modèle calculés :")
col1, col2 = st.columns(2)
col1.metric("Seuil de décision (w₀)", f"{w_0:.4f}")
col2.metric("Pixels sélectionnés", f"{les_2_pixels[0]} et {les_2_pixels[1]}")

# 5. Graphique de la Frontière de Décision 2D
st.subheader("Graphique : Frontière de décision (Boundary)")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(X_0_2D[:, 0], X_0_2D[:, 1], color='blue', alpha=0.7, label='Chiffre 0', edgecolors='k')
ax.scatter(X_1_2D[:, 0], X_1_2D[:, 1], color='red', alpha=0.7, label='Chiffre 1', edgecolors='k')

# Tracer la ligne de décision
x_vals = np.linspace(np.min(X_2D[:, 0]), np.max(X_2D[:, 0]), 100)
y_vals = -(w[0] / w[1]) * x_vals + (w_0 / w[1])
ax.plot(x_vals, y_vals, color='black', linestyle='--', linewidth=2.5, label='Frontière LDA')

ax.set_xlabel(f"Intensité Pixel {les_2_pixels[0]}")
ax.set_ylabel(f"Intensité Pixel {les_2_pixels[1]}")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# Afficher le graphique Matplotlib dans Streamlit
st.pyplot(fig)
