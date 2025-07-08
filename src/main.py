import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

from preprocessing import variaveis_relevantes
from model_utils import criar_modelo_nn, get_modelos_base, empilhar_modelos

# Carregar os dados (adicione seu caminho ou exemplo)
data = pd.read_csv('seu_arquivo.csv')  # <- Altere este nome conforme necessário

# Separar variáveis
X = data[variaveis_relevantes]
y = data['Desgaste']

# Normalização
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Validação cruzada
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Previsões out-of-fold
preds_nn = np.zeros(len(y))
preds_svr = np.zeros(len(y))
preds_et = np.zeros(len(y))
preds_xgb = np.zeros(len(y))
preds_lgbm = np.zeros(len(y))
preds_gbr = np.zeros(len(y))

modelos_base = get_modelos_base()

for train_idx, val_idx in kf.split(X_scaled):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Rede Neural
    modelo_nn = criar_modelo_nn(X_scaled.shape[1])
    modelo_nn.fit(X_train, y_train, epochs=200, batch_size=20, verbose=0)
    preds_nn[val_idx] = modelo_nn.predict(X_val).flatten()

    # Demais modelos
    modelos_treinados = {}
    for nome, modelo in modelos_base.items():
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_val)
        if nome == 'SVR': preds_svr[val_idx] = y_pred
        elif nome == 'ExtraTrees': preds_et[val_idx] = y_pred
        elif nome == 'XGB': preds_xgb[val_idx] = y_pred
        elif nome == 'LGBM': preds_lgbm[val_idx] = y_pred
        elif nome == 'GBR': preds_gbr[val_idx] = y_pred

# Empilhamento
meta_model = empilhar_modelos(
    [preds_nn, preds_svr, preds_et, preds_xgb, preds_lgbm, preds_gbr],
    y
)

# Avaliação
X_meta = np.column_stack((preds_nn, preds_svr, preds_et, preds_xgb, preds_lgbm, preds_gbr))
y_pred_final = meta_model.predict(X_meta)
r2_final = r2_score(y, y_pred_final)
print(f'R² do Ensemble (Stacking): {r2_final:.4f}')

# Visualizações comparativas
def plot_result(y_true, y_pred, title):
    plt.figure(figsize=(10, 6))
    plt.plot(y_true.values, label='Valores Reais', marker='o')
    plt.plot(y_pred, label=title, linestyle='--', marker='x')
    plt.xlabel('Amostras')
    plt.ylabel('Vida Útil da Ferramenta')
    plt.title(f'Comparação: {title}')
    plt.legend()
    plt.grid(True)
    plt.show()

plot_result(y, preds_svr, 'SVR')
plot_result(y, preds_nn, 'Redes Neurais')
plot_result(y, y_pred_final, 'Ensemble (Stacking)')
