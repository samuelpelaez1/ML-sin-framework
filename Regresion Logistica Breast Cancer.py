# Importarmos las librerias necesarias
import pandas as pd
import numpy as np

# Definimos la funciones necesarias

# Funcion de activación
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Funcion de costo (negative log loss)
def compute_cost(X, y, theta):
    m = len(y) # Numero de datos 
    h = sigmoid(np.dot(X, theta)) 
    cost = (-1/m) * (np.dot(y, np.log(h + 1e-10)) + np.dot((1-y), np.log(1-h + 1e-10)))
    return cost

# Gradiente de la función de costo, esta para minimizar el costo
def compute_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(np.dot(X, theta))
    gradient = (1/m) * np.dot(X.T, (h - y))
    return gradient

# Optimización de los parámetros para minimizar el costo 
def gradient_descent(X, y, theta, learning_rate, iterations):
    cost_history = []
    for i in range(iterations):
        gradient = compute_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        cost = compute_cost(X, y, theta)
        cost_history.append(cost)
        print(f"Iteration {i+1}: Cost {cost}")
    
    return theta, cost_history

# Generar las predicciones 
def predict(X, theta):
    probabilities = sigmoid(np.dot(X, theta))
    return [1 if p >= 0.5 else 0 for p in probabilities]

# Convertir el 'target' a 0 y 1 B (benign) = 0, M(malignant) = 1
df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1})

#Separamos los features y el target
X = df[['radius_mean', 'texture_mean', 'perimeter_mean',
       'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean',
       'concave points_mean', 'symmetry_mean', 'fractal_dimension_mean',
       'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se',
       'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se',
       'fractal_dimension_se', 'radius_worst', 'texture_worst',
       'perimeter_worst', 'area_worst', 'smoothness_worst',
       'compactness_worst', 'concavity_worst', 'concave points_worst',
       'symmetry_worst', 'fractal_dimension_worst']].values  
y = df['diagnosis'].values 

# Setear una semilla para poder repetir el mismo experimento
np.random.seed(12)

# Mezclar aleatoriamente el orden de los features para evitar sesgos
indices = np.random.permutation(X.shape[0])

# Separar el 80% de los datos para entrenamiento y el 20% para pruebas
train_size = int(0.8 * X.shape[0])

# Separar los datos en entrenamiento y pruebas
X_train = X[indices[:train_size]]
y_train = y[indices[:train_size]]
X_test = X[indices[train_size:]]
y_test = y[indices[train_size:]]

# Agregar un término de sesgo a los features
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]

# Inicializar los parámetros theta
theta = np.zeros(X_train.shape[1])

# Entrenar el modelo
theta_final, cost_history = gradient_descent(X_train, y_train, theta, learning_rate=0.001, iterations=1000)

# Hacer predicciones en el conjunto de prueba
predictions = predict(X_test, theta_final)

# Evaluar el modelo en el conjunto de prueba e imprimir 
accuracy = np.mean(predictions == y_test) * 100
print(f"Accuracy: {accuracy:.2f}%")

# Convertir las predicciones de 0,1 a 'M','B'
predictions_mapped = ['M' if p == 1 else 'B' for p in predictions]

# Crear un DataFrame con los resultados para comparación
results_df = pd.DataFrame({
    'Index': df.index[indices[train_size:]],  # Índice original
    'True Diagnosis': df['diagnosis'].map({0: 'B', 1: 'M'}).iloc[indices[train_size:]].values,  # Valores reales
    'Predicted Diagnosis': predictions_mapped  # Predicciones
})

# Mostrar los resultados
print(results_df)