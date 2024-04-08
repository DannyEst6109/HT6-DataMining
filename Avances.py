# Librerias
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report
import time

# Cargar datos
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

train = train.iloc[:1460, :]

# Porcentaje de datos para entrenamiento
porcentaje = 0.7
np.random.seed(123)
train, test = train_test_split(train, test_size=(1-porcentaje), random_state=123)

# Visualización de los primeros registros de train y test
print(train.head())
print(test.head())

# Crear variables dicotómicas
train['grupo'] = pd.cut(train['SalePrice'], bins=[0, 178000, 301000, np.inf], labels=[1, 2, 3])
train['grupo2'] = pd.cut(train['SalePrice'], bins=[0, 178000, 301000, np.inf], labels=[1, 2, np.nan])
train['grupo3'] = pd.cut(train['SalePrice'], bins=[0, 178000, 301000, np.inf], labels=[1, np.nan, np.nan])

# Convertir a variables categóricas
train['grupo'] = train['grupo'].astype('category')
train['grupo2'] = train['grupo2'].astype('category')
train['grupo3'] = train['grupo3'].astype('category')

# Crear variables dummy
train = pd.concat([train, pd.get_dummies(train['grupo'], prefix='datos'),
                   pd.get_dummies(train['grupo2'], prefix='datos'),
                   pd.get_dummies(train['grupo3'], prefix='datos')], axis=1)

# Seleccionar variables para el modelo
variables_modelo = ['LotFrontage', 'LotArea', 'GrLivArea', 'YearBuilt', 'BsmtUnfSF',
                    'TotalBsmtSF', 'X1stFlrSF', 'GarageYrBlt', 'GarageArea', 
                    'YearRemodAdd', 'SalePrice', 'datos_1', 'datos_2', 'datos_3']

train = train[variables_modelo].dropna()

# Dividir datos en entrenamiento y prueba
train, test = train_test_split(train, test_size=(1-porcentaje), random_state=123)

# Entrenar modelos
for i in range(1, 4):
    tic = time.time()
    print(f"Entrenamiento modelo casas {'caras' if i == 1 else 'intermedias' if i == 2 else 'baratas'}")
    X_train = train.drop(columns=['SalePrice', f'datos_{i}'])
    y_train = train[f'datos_{i}']
    modelo = sm.Logit(y_train, sm.add_constant(X_train)).fit(maxiter=100)
    print(modelo.summary())
    toc = time.time()
    print(f"Tiempo transcurrido: {toc - tic:.2f} segundos\n")
