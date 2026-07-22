import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

st.header("EL MEHDI - IAENG")
st.title("TP LDA : Projection 1D (Version MNIST 28x28)")
st.write("Chargement du vrai dataset MNIST et calcul de la projection 1D *From Scratch*.")

# 1. Chargement des données avec Cache pour Streamlit
@st.cache_data
def load_mnist_784():
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    return X, y

with st.spinner("Chargement du dataset MNIST en cours..."):
    X, y = load_mnist_784()

# Filtrage strict pour ne garder que les chiffres 0 et 1, 
# et limitation rigoureuse à 1500 images pour une rapidité maximale
masque = (y == 0) | (y == 1)
X_filtre = X[masque][:1500] / 255.0  
y_filtre = y[masque][:1500]

# 2. Fonction lourde mise en cache pour isoler le calcul mathématique LDA (784 dimensions)
@st.cache_resource
def compute_lda_weights(X_f, y_f):
    X_0 = X_f[y_f == 0]
    X_1 = X_f[y_f == 1]

    P_0 = len(X_0) / len(X_f)
    P_1 = len(X_1) / len(X_f)

    mu_0 = np.mean(X_0, axis=0)
    mu_1 = np.mean(X_1, axis=0)

    cov_0 = np.cov(X_0, rowvar=False)
    cov_1 = np.cov(X_1, rowvar=False)

    # Régularisation à 1e-4 adaptée aux 784 dimensions
    Sigma = (cov_0 + cov_1) / 2 + 1e-4 * np.eye(784)

    w = np.linalg.solve(Sigma, mu_1 - mu_0)
    w_0 = 0.5 * np.dot(w.T, (mu_1 + mu_0)) - np.log(P_1 / P_0)
    
    return w, w_0

with st.spinner("Calculs mathématiques LDA en cours..."):
    w, w_0 = compute_lda_weights(X_filtre, y_filtre)

st.success("Calculs LDA terminés avec succès !")

# 3. Projection 1D et Prédictions
projections = np.dot(X_filtre, w)

# Règle de décision : si projection > w_0 on prédit 1, sinon 0 (ou inversement selon le sens des moyennes)
# On détermine dynamiquement le sens en comparant les moyennes des projections de chaque classe
mean_proj_0 = np.mean(projections[y_filtre == 0])
mean_proj_1 = np.mean(projections[y_filtre == 1])

if mean_proj_0 < mean_proj_1:
    y_pred = np.where(projections >= w_0, 1, 0)
else:
    y_pred = np.where(projections >= w_0, 0, 1)

# Calcul de l'accuracy
acc = accuracy_score(y_filtre, y_pred)

# Affichage de l'Accuracy dans l'interface
st.metric(label="📊 Précision (Accuracy) du modèle LDA From Scratch", value=f"{acc * 100:.2f}%")

# 4. Visualisation sécurisée
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(projections[y_filtre == 0], bins=40, alpha=0.6, color='blue', label='Chiffre 0 (28x28)')
ax.hist(projections[y_filtre == 1], bins=40, alpha=0.6, color='red', label='Chiffre 1 (28x28)')
ax.axvline(x=w_0, color='black', linestyle='--', linewidth=2.5, label=f'Frontière (w_0 = {w_0:.2f})')

ax.set_title(f"Projection LDA 1D From Scratch (Accuracy : {acc * 100:.2f}%)")
ax.set_xlabel("Valeur de la projection 1D")
ax.set_ylabel("Nombre d'images")
ax.legend()
ax.grid(True, linestyle=':', alpha=0.6)

# Affichage et libération propre de la mémoire
st.pyplot(fig)
plt.close(fig)
