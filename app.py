import streamlit as st

datasets = st.Page("Datasets.py", title="Datasets", icon=":material/dataset:")
clustering = st.Page("ClusteringPenguins.py", title="Clustering", icon=":material/linked_services:")

pg = st.navigation([datasets, clustering])
pg.run()