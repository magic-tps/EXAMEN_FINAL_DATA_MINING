import joblib
import streamlit as st
import numpy as np

# Cargar todos los modelos desde el archivo único
modelos = joblib.load('modelos_clasificacion.pkl')
lin_reg = modelos["regresion_lineal"]
svc = modelos["svc"]
log_reg = modelos["regresion_logistica"]


clase_map = {
    0: 'Anabas testudineus',
    1: 'Coilia dussumieri',
    3: 'Otolithoides biauritus',
    4: 'Otolithoides pama',
    5: 'Pethia conchonius',
    6: 'Polynemus paradiseus',
    7: 'Puntius lateristriga',
    8: 'Setipinna taty',
    9: 'Sillaginopsis panijus',
}
  
# Crear la interfaz de Streamlit
st.title("Aplicación de Clasificación de Especies")

# Entrada del usuario
st.header("Ingrese los valores para clasificar la especie")
length = st.number_input("Longitud", min_value=0.0, step=0.1)
weight = st.number_input("Peso", min_value=0.0, step=0.1)
w_l_ratio = st.number_input("Relación Peso-Longitud", min_value=0.0, step=0.01)

# Botón de predicción
if st.button("Predecir"):
    input_data = np.array([[length, weight, w_l_ratio]])

    # Predicciones con los modelos y mapeo al nombre de la clase
    pred_lin_reg = clase_map[lin_reg.predict(input_data).round().astype(int)[0]]
    pred_svc = clase_map[svc.predict(input_data)[0]]
    pred_log_reg = clase_map[log_reg.predict(input_data)[0]]

    # Mostrar resultados
    st.subheader("Resultados de las Predicciones")
    st.write(f"**Regresión Lineal Predice:** {pred_lin_reg}")
    st.write(f"**SVC Predice:** {pred_svc}")
    st.write(f"**Regresión Logística Predice:** {pred_log_reg}")