import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger le modèle
with open("models/model.pkl", "rb") as file:
    model = pickle.load(file)

# Simuler la standardisation (remplacez par le scaler utilisé lors de l'entraînement)
scaler = StandardScaler()

# Interface utilisateur avec Streamlit
st.title("Prédiction du Churn Client")

st.write("Remplissez les informations suivantes pour prédire si un client va churner.")

# Exemple de colonnes (à adapter selon les features du dataset)
feature_names = ['REGION', 'TENURE', 'MONTANT', 'FREQUENCE_RECH', 'REVENUE',
                 'ARPU_SEGMENT', 'FREQUENCE', 'DATA_VOLUME', 'ON_NET', 'ORANGE',
                 'TIGO', 'ZONE1', 'ZONE2', 'MRG', 'REGULARITY', 'TOP_PACK', 'FREQ_TOP_PACK']

user_input = []
for feature in feature_names:
    user_input.append(st.number_input(f"{feature}", value=0.0))

# Convertir l'entrée utilisateur en tableau numpy
input_data = np.array(user_input).reshape(1, -1)
input_data = scaler.fit_transform(input_data)  # Remplacez par scaler.transform si sauvegardé

if st.button("Prédire"):
    prediction = model.predict(input_data)
    st.write("### Résultat de la prédiction:")
    st.write("🔴 Churn" if prediction[0] == 1 else "🟢 Non Churn")

# Pour exécuter l'application : 'streamlit run app.py' dans le terminal

