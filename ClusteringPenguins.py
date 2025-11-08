from allDataframes import df_og
from allDataframes import df_pinguinos
from allDataframes import df_clean
from allDataframes import df_preprocesado
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


st.title("Clustering de pingüinos")

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X = scaler.fit_transform(df_clean)

#CLUSTERING KMEANS
from sklearn.cluster import KMeans
st.markdown("## **Clustering Kmeans**")
    #GRÁFICO DEL CODO
st.markdown("### Gráfico del codo")
num_clusters = range(1, 11)
kmeans = [KMeans(n_clusters=i) for i in num_clusters]
score = [kmeans[i].fit(X).score(X) for i in range(len(kmeans))]
fig, ax = plt.subplots()
ax.plot(num_clusters, score)
ax.grid()
ax.set_xlabel('Número de Clusters')
ax.set_ylabel('Score')
ax.set_title('Gráfico del codo')
st.pyplot(fig)
st.write("Lo más optimo sería usar 4 clusters en el agrupamiento.")
    #CLUSTERING
st.markdown("## Parámetros")
num_clusters = st.number_input("Cantidad de clusters", 2, 10)
r_state = st.number_input("Random state", 1, 50)
num_init = st.number_input("n_init", 1, 50)
kmeans = KMeans(n_clusters=num_clusters, random_state=r_state, n_init=num_init)
kmeans_res = kmeans.fit_predict(df_preprocesado)
df_kmeans = df_preprocesado.copy()
df_kmeans.insert(0, "cluster", kmeans_res)
    #MÉTRICAS DE EVALUACIÓN
st.markdown("## Métricas de evaluación")
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
silhouette = silhouette_score(df_preprocesado, kmeans_res)
db_index = davies_bouldin_score(df_preprocesado, kmeans_res)
ch_index = calinski_harabasz_score(df_preprocesado, kmeans_res)
st.write("Silhouette Score: ", silhouette)
st.write("Davies-Bouldin Index: ", db_index)
st.write("Calinski-Harabasz Index: ", ch_index)
    #RESULTADOS
st.markdown("## Resultados")
st.write("Dataset con columna 'cluster' indicando el grupo del pingüino")
st.write(df_kmeans)
st.write("Cantidad de pingüinos por cluster")
st.write(df_kmeans.cluster.value_counts())
st.markdown("### Scatter Plot Interactivo")
col1, col2 = st.columns(2)
with col1:
    x_feature = st.selectbox("Selecciona feature para eje X:", 
                            df_preprocesado.columns, 
                            index=0)
with col2:
    y_feature = st.selectbox("Selecciona feature para eje Y:", 
                            df_preprocesado.columns, 
                            index=1)
fig1, ax = plt.subplots(figsize=(10, 6))
scatter2 = ax.scatter(df_kmeans[x_feature], 
                      df_kmeans[y_feature], 
                      c=df_kmeans['cluster'], 
                      cmap='viridis', 
                      alpha=0.7,
                      s=50)
ax.set_xlabel(x_feature)
ax.set_ylabel(y_feature)
ax.set_title(f'KMeans Clustering: {x_feature} vs {y_feature}')
ax.legend()
plt.colorbar(scatter2, ax=ax, label='Cluster')
st.pyplot(fig1)
    #VISUALIZACIÓN DE ESPECIES CON GRÁFICO ARAÑA
st.markdown("### **Visualización de especies con gráfico**")
means = df_kmeans.groupby("cluster").mean().round(2)
features = ["culmen_length_mm", "culmen_depth_mm", "flipper_length_mm", "body_mass_g"]  #SOLO NUMERICAS
angles = np.linspace(0, 2 * np.pi, len(features), endpoint=False).tolist()
angles += angles[:1]
fig2, ax2 = plt.subplots(figsize=(6,6), subplot_kw=dict(polar=True))
for i, (index, row) in enumerate(means.iterrows()):
    values = row[features].tolist()
    values += values[:1]  # cerrar el círculo
    ax2.plot(angles, values, label=f"Cluster {index}")
    ax2.fill(angles, values, alpha=0.1)
ax2.set_xticks(angles[:-1])
ax2.set_xticklabels(features, fontsize=10)
ax2.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
st.pyplot(fig2)