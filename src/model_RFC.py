import pandas as pd
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config_ruta import ruta_datos 
from load import cargar_datos




#Dividir el set en 'x' e 'Y'
def dividir_datos(ruta):
    """
    Divide los datos en variables independientes (X) y dependiente (Y).

    Parámetros:
    ruta (str): Ruta del archivo CSV.

    Retorna:
    tuple: Un par (X, Y) donde:
        - X es un DataFrame con las variables independientes.
        - Y es un Series con la variable dependiente.
    """
    # Cargar los datos usando la función cargar_datos
    datos = pd.DataFrame(cargar_datos(ruta))
    
    if datos is not None:
        print("Columnas encontradas en el archivo:")
        print(datos.columns)  # Verifica las columnas presentes en el DataFrame
        
        if 'custcat' in datos.columns:
        
            try:
                # Separar la variable dependiente (Y) y las independientes (X)
                X = datos.drop(columns=['custcat'])  # Eliminar la columna objetivo
                Y = datos['custcat']  # Seleccionar la columna objetivo
                print("Datos divididos exitosamente.")
                return X, Y
            
            except Exception as e:
                print(f"Error inesperado al dividir los datos: {e}")
        else:
            print("Error: La columna 'custcut' no existe en los datos.")
    else:
        print("No se pudieron cargar los datos. Verifica la ruta y formato del archivo.")
    
    # Si ocurre un error, devolver None
    return None, None


# Dividir los datos
X, Y = dividir_datos(ruta_datos)

if X is not None and Y is not None:
    print("Variables independientes (X):")
    print(X.head())
    print("Variable dependiente (Y):")
    print(Y.head())
else:
    print("No se pudieron dividir los datos.")
    


# 1. Dividir los datos en train y test (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# 2. Inicializar el modelo de RandomForestClassifier
modelo = RandomForestClassifier(random_state=42)

# 3. Entrenar el modelo con el conjunto de entrenamiento
modelo.fit(X_train, Y_train)

# 4. Predecir los valores en el conjunto de prueba
Y_pred = modelo.predict(X_test)

# 5. Calcular la precisión del modelo en el conjunto de prueba
precision = accuracy_score(Y_test, Y_pred)
print(f"Precisión en el conjunto de prueba: {precision:.4f}")

# 6. Realizar validación cruzada con 5 particiones
# Esto realiza validación cruzada sobre el conjunto de entrenamiento para obtener una medida más robusta del rendimiento del modelo
cv_scores = cross_val_score(modelo, X_train, Y_train, cv=5)

# Mostrar los resultados de la validación cruzada
print(f"Precisión promedio en validación cruzada: {cv_scores.mean():.4f}")
print(f"Desviación estándar de la validación cruzada: {cv_scores.std():.4f}")