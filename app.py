import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

# Cargar los modelos guardados
rf_model = joblib.load('rf_best_model.pkl')
knn_model = joblib.load('knn_best_model.pkl')

# Título de la aplicación
st.title("Predicción de Supervivencia Titanic")
st.write("Utilice este modelo para predecir la supervivencia de un pasajero en el Titanic utilizando los modelos entrenados de Random Forest y K-Nearest Neighbors.")

# Inputs para el usuario
st.sidebar.header("Ingrese los detalles del pasajero")
Pclass = st.sidebar.selectbox("Pclass", [1, 2, 3])
Sex = st.sidebar.selectbox("Sex", ["Mujer", "Hombre"])
Age = st.sidebar.number_input("Age", min_value=0, value=30)
SibSp = st.sidebar.number_input("SibSp", min_value=0, value=0)
Parch = st.sidebar.number_input("Parch", min_value=0, value=0)
Fare = st.sidebar.number_input("Fare", min_value=0.0, value=7.25)
Embarked = st.sidebar.selectbox("Embarked", ['C', 'Q', 'S'])

# Convertir entradas a formato numérico
Sex = 1 if Sex == "Hombre" else 0
Embarked = {'C': 0, 'Q': 1, 'S': 2}[Embarked]

# Crear un DataFrame con las entradas
input_data = pd.DataFrame([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]], columns=['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked'])

# Escalar los datos
scaler = StandardScaler()
input_data_scaled = scaler.fit_transform(input_data)

# Botón para hacer predicción
if st.button("Predecir Supervivencia"):
    # Hacer predicciones con ambos modelos
    rf_prediction = rf_model.predict(input_data_scaled)[0]
    knn_prediction = knn_model.predict(input_data_scaled)[0]

    # Mostrar los resultados
    st.write(f"**Predicción de Random Forest:** {'Superviviente' if rf_prediction == 1 else 'No sobreviviente'}")
    st.write(f"**Predicción de K-Nearest Neighbors:** {'Superviviente' if knn_prediction == 1 else 'No sobreviviente'}")


