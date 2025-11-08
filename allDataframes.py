import pandas as pd
from sklearn.preprocessing import StandardScaler

df_og = pd.read_csv("penguins.csv")
df_pinguinos = df_og.dropna(axis='rows')
df_pinguinos = df_pinguinos.drop([9, 14])

df_clean = pd.get_dummies(df_pinguinos).drop("sex_.", axis=1)
scaler = StandardScaler()
X = scaler.fit_transform(df_clean)
df_preprocesado = pd.DataFrame(data=X, columns=df_clean.columns)