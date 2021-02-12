from pycaret.classification import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
#from pycaret.regression import *

def predict_quality(model, df):
    
    predictions_data = predict_model(estimator = model, data = df)
    return predictions_data['Label'][0]
    
model = load_model('Final_rf')
############################################
# Title - The title and introductory text and images are all written in Markdown format here, using st.write()

st.write("""

# Predicción de victimización de empresas

Esta aplicación predice la victimizacin  de una empresa en Perú mediante un modelo de aprendizaje automático impulsado por[Pycaret](https://pycaret.org/).

Los datos del modelo son btenids de INEI [victimizacin de empresas](https://www.inei.gob.pe) Dataset.

Juega con los valores a través de los controles deslizantes del panel izquierdo para generar nuevas predicciones.
""")
st.write("---")



full_df = pd.read_csv('data/example1.csv')



# Sidebar - this sidebar allows the user to set the parameters that will be used by the model to create the prediction.
st.sidebar.header('Especifique los parámetros para determinar la predicción.')

Cdig = st.sidebar.slider('Código de Actividad Económica', int(full_df.Cdig.min()), int(full_df.Cdig.max()), int(full_df.Cdig.min()))
Departament = st.sidebar.selectbox('Nombre del Departamento', ['AMAZONAS','ÁNCASH','APURÍMAC','AREQUIPA','AYACUCHO','CAJAMARCA','CALLAO','CUSCO','LIMA'])
Tama = st.sidebar.selectbox('Tamaño de Empresa', ['MICRO', 'PEQUEÑA','GRANDE'])
P1 = st.sidebar.selectbox('Infraestructura física (alambrado, muros, etc.)?', ['Si', 'NO'])
P2 = st.sidebar.selectbox('Sistema de video y captura de imágenes?', ['Si', 'NO'])
P3 = st.sidebar.selectbox('Sistema de control de acceso de personal?', ['Si', 'NO'])
P4 = st.sidebar.selectbox('Sistema de alarma de seguridad electrónica?',['Si', 'NO'])
P5 = st.sidebar.selectbox('Seguridad para el traslado de valores?', ['Si', 'NO'])
P6 = st.sidebar.selectbox('Seguridad para el traslado de bienes?', ['Si', 'NO'])
P7 = st.sidebar.selectbox('Personal para resguardo (guardaespaldas)?',['Si', 'NO'])
P8 = st.sidebar.selectbox('Personal de seguridad de bienes e inmuebles?', ['Si', 'NO'])
   
features  = {'Cdig': Cdig,
            'Departament': Departament,
            'Tama': Tama,
            'P1': P1,
            'P2': P2,
            'P3': P3,
            'P4': P4,
            'P5': P5,
            'P6': P6,
            'P7': P7,
            'P8': P8}

features_df  = pd.DataFrame([features])

st.table(features_df)  

if st.button('Predicción'):
    
    prediction = predict_quality(model, features_df)
    
    st.write('Según sus selecciones, el modelo predice un valor de '+ str(prediction))




   