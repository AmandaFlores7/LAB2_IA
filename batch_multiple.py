import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('winequality-red.csv', sep=';')

# División de datos en entrenamiento y prueba (80% entrenamiento, 20% prueba)
train_size = int(0.8 * len(data))
train_data = data[:train_size]
test_data = data[train_size:]

# Extrae las características y las etiquetas de entrenamiento y prueba
train_X = train_data[['alcohol', 'volatile acidity', 'fixed acidity']].values
train_y = train_data['quality'].values
test_X = test_data[['alcohol', 'volatile acidity', 'fixed acidity']].values
test_y = test_data['quality'].values

# Normalización de los datos de entrenamiento
scaler = MinMaxScaler()
train_X = scaler.fit_transform(train_X)

# Normalización de los datos de prueba
test_X = scaler.transform(test_X)

# Función de entrenamiento utilizando descenso de gradiente por lotes
def train_linear_regression(X, y, learning_rate, num_iterations):
    # Inicializa los parámetros del modelo
    theta = np.zeros(X.shape[1])
    m = len(X)

    # Descenso de gradiente
    for iteration in range(num_iterations):
        # Calcula las predicciones y el error
        y_pred = np.dot(X, theta)
        error = y_pred - y

        # Actualiza los parámetros utilizando el gradiente
        theta -= (learning_rate / m) * np.dot(X.T, error)

    return theta

# Hiperparámetros del modelo
learning_rate = 0.01
num_iterations = 1000

# Entrena el modelo utilizando los datos de entrenamiento
theta = train_linear_regression(train_X, train_y, learning_rate, num_iterations)

# Función para hacer predicciones
def predict(X, theta):
    return np.dot(X, theta)

# Realiza predicciones en los conjuntos de entrenamiento y prueba
train_predictions = predict(train_X, theta)
test_predictions = predict(test_X, theta)

# Cálculo de las métricas de evaluación
def mse(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def rmse(y_true, y_pred):
    return np.sqrt(mse(y_true, y_pred))

def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def r_squared(y_true, y_pred):
    ssr = np.sum((y_true - y_pred) ** 2)
    sst = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ssr / sst)

def adjusted_r_squared(y_true, y_pred, num_features):
    r2 = r_squared(y_true, y_pred)
    n = len(y_true)
    return 1 - ((1 - r2) * (n - 1) / (n - num_features - 1))

# Calcular métricas de evaluación en los conjuntos de entrenamiento y prueba
train_mse = mse(train_y, train_predictions)
train_rmse = rmse(train_y, train_predictions)
train_mae = mae(train_y, train_predictions)
train_r2 = r_squared(train_y, train_predictions)
train_adjusted_r2 = adjusted_r_squared(train_y, train_predictions, train_X.shape[1])

test_mse = mse(test_y, test_predictions)
test_rmse = rmse(test_y, test_predictions)
test_mae = mae(test_y, test_predictions)
test_r2 = r_squared(test_y, test_predictions)
test_adjusted_r2 = adjusted_r_squared(test_y, test_predictions, test_X.shape[1])

# Imprimir las métricas de evaluación
print("Métricas de evaluación (conjunto de entrenamiento):")
print("MSE:", train_mse)
print("RMSE:", train_rmse)
print("MAE:", train_mae)
print("R^2:", train_r2)
print("R^2 ajustado:", train_adjusted_r2)
print("--------------------------------------------")
print("Métricas de evaluación (conjunto de prueba):")
print("MSE:", test_mse)
print("RMSE:", test_rmse)
print("MAE:", test_mae)
print("R^2:", test_r2)
print("R^2 ajustado:", test_adjusted_r2)

# Gráfico de la regresión
plt.scatter(train_X[:, 0], train_y, color='blue', label='Training Data')
#plt.scatter(test_X[:, 0], test_y, color='red', label='Test Data')
plt.plot(train_X[:, 0], train_predictions, color='green', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Regresión lineal con múltiples variables')
plt.legend()
plt.show()