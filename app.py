import streamlit as st

datasets = st.Page("Datasets.py", title="Datasets", icon=":material/dataset:")
kmeans = st.Page("kmeans.py", title="KMeans", icon=":material/linked_services:")
jerarquico = st.Page("jerarquico.py", title="Clustering Jerárquico", icon=":material/tenancy:")

pg = st.navigation([datasets, kmeans, jerarquico])
pg.run()