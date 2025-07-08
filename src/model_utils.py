import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

def criar_modelo_nn(input_dim):
    modelo = Sequential()
    modelo.add(Dense(64, input_dim=input_dim, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(32, activation='relu'))
    modelo.add(Dropout(0.2))
    modelo.add(Dense(16, activation='relu'))
    modelo.add(Dense(1))
    modelo.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=0.001))
    return modelo

def get_modelos_base():
    modelos = {
        'SVR': SVR(C=1, epsilon=0.01, kernel='linear'),
        'ExtraTrees': ExtraTreesRegressor(n_estimators=100, random_state=42),
        'XGB': XGBRegressor(colsample_bytree=1.0, learning_rate=0.1, max_depth=3,
                            min_child_weight=1, n_estimators=200, subsample=0.8),
        'LGBM': LGBMRegressor(colsample_bytree=1.0, learning_rate=0.1, max_depth=3,
                              n_estimators=100, reg_alpha=0.1, reg_lambda=0.1, subsample=0.8),
        'GBR': GradientBoostingRegressor(learning_rate=0.1, max_depth=7,
                                         min_samples_leaf=1, min_samples_split=10,
                                         n_estimators=200),
    }
    return modelos

def empilhar_modelos(predicoes_base, y_true):
    """
    Recebe as predições dos modelos base (out-of-fold) e treina um meta-modelo (regressão linear).
    """
    X_meta = np.column_stack(predicoes_base)
    meta_model = LinearRegression()
    meta_model.fit(X_meta, y_true)
    return meta_model
