from allDataframes import df_preprocesado
from allDataframes import X
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.title("Clustering de pingüinos - Clustering Jerárquico")

#CLUSTERING JERÁRQUICO
st.markdown("### Dendograma")
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_title("Dendograma de pingüinos")
dend = shc.dendrogram(shc.linkage(df_preprocesado, method='ward'))
st.pyplot(fig)
    #CLUSTERING
st.markdown("## Parámetros")
num_clusters = st.number_input("Cantidad de clusters", 2, 10)
link = st.selectbox("Linkage", ["ward", "complete", "average", "single"])
clustering = AgglomerativeClustering(n_clusters=num_clusters, linkage=link)
cj_res = clustering.fit_predict(df_preprocesado)
df_cj = df_preprocesado.copy()
df_cj.insert(0, "cluster", cj_res)
#MÉTRICAS DE EVALUACIÓN
st.markdown("## Métricas de evaluación")
silhouette = silhouette_score(df_preprocesado, cj_res)
db_index = davies_bouldin_score(df_preprocesado, cj_res)
ch_index = calinski_harabasz_score(df_preprocesado, cj_res)
st.write("Silhouette Score: ", silhouette)
st.write("Davies-Bouldin Index: ", db_index)
st.write("Calinski-Harabasz Index: ", ch_index)
    #RESULTADOS
st.markdown("## Resultados")
st.write("Dataset con columna 'cluster' indicando el grupo del pingüino")
st.write(df_cj)
st.write("Cantidad de pingüinos por cluster")
st.write(df_cj.cluster.value_counts())
st.markdown("### Scatter Plot Interactivo")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("Selecciona feature para eje X:", 
                            df_preprocesado.columns, 
                            index=3)
with col2:
    y_feature = st.selectbox("Selecciona feature para eje Y:", 
                            df_preprocesado.columns, 
                            index=1)
fig1, ax = plt.subplots(figsize=(10, 6))
scatter2 = ax.scatter(df_cj[x_feature], 
                      df_cj[y_feature], 
                      c=df_cj['cluster'], 
                      cmap='viridis', 
                      alpha=0.7,
                      s=50)
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(f'Clustering Jerárquico: {x_feature} vs {y_feature}')
ax.legend()
plt.colorbar(scatter2, ax=ax, label='Cluster')
st.pyplot(fig1)
    #VISUALIZACIÓN DE ESPECIES CON GRÁFICO ARAÑA
st.markdown("### **Visualización de especies con gráfico**")
means = df_cj.groupby("cluster").mean().round(2)
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]
fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
for i, (index, row) in enumerate(means.iterrows()):
    values = row[features].tolist()
    values += values[:1]
    ax2.plot(angles, values, label=f"Cluster {index}")
    ax2.fill(angles, values, alpha=0.1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(features, fontsize=10)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig2)