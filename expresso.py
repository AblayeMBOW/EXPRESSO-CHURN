import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Charger les données
data = pd.read_csv("Expresso_churn_dataset.csv")

# Séparer les caractéristiques (X)
X = data.drop(columns=["CHURN"])

# Standardiser les données
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Sauvegarder le scaler
with open("scaler.pkl", "wb") as file:
    pickle.dump(scaler, file)
