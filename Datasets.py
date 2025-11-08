from allDataframes import df_og
from allDataframes import df_pinguinos
from allDataframes import df_preprocesado
import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler

st.title("Clustering de pingüinos")
st.write("Se busca agrupar pingüinos en distintas especies según información física (masa, tamaño de aleta, etc.).")
#DATASET
st.markdown("## **Datasets de pingüinos**")
    #ORIGINAL
st.markdown("### Dataset original:")
st.write(df_og)
st.write("El dataset presenta datos nulos e incoherentes los cuales se deben limpiar")
    #LIMPIADO DATOS NULOS E INCOHERENTES
st.markdown("### Dataset limpio:")
st.write(df_pinguinos)
st.write("Ahora se convierte la variable categórica (get_dummies) y se estandarizan los datos (StandardScaler)")
    #DATOS ESCALADOS
st.markdown("### Dataset con datos escalados por StandardScaler:")
st.write(df_preprocesado)