import pandas as pd
from sklearn.preprocessing import StandardScaler

def carregar_e_normalizar_dados(caminho_csv, variaveis_relevantes, alvo='Desgaste'):
    """
    Carrega o dataset, separa as variáveis independentes e dependente,
    e normaliza os dados de entrada.
    """
    data = pd.read_csv(caminho_csv)
    X = data[variaveis_relevantes]
    y = data[alvo]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler
