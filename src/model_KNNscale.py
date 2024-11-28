from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

import pandas as pd
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, cross_val_score

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
    






# Paso 1: Identificar las variables continuas
# Asumimos que todas las variables numéricas son continuas (esto puede cambiar según el contexto)

#X_categoricas=['region', 'marital','address','retire', 'gender','reside' ]
var_ord=['ed']

# Paso 1: Variables categóricas
variables_categoricas = ['region', 'marital', 'address', 'retire', 'gender', 'reside']
X['region'] = X['region'].astype('category')
X['marital'] = X['marital'].astype('category')
#X['address'] = X['address'].astype('category') #Y si calculo frecuencia para este
X['retire'] = X['retire'].astype('category')
X['gender'] = X['gender'].astype('category')
X['reside'] = X['reside'].astype('category')

print(X[variables_categoricas])
# Paso 2: Aplicar get_dummies() a las variables categóricas
X_categoricas = pd.get_dummies(X[variables_categoricas], drop_first=True)  # drop_first=True evita la multicolinealidad
print(f'Con get_dummies:\n{X_categoricas}')




# Paso 3: Concatenar las variables categóricas codificadas con el resto de las variables
X_restantes = X.drop(columns=variables_categoricas)  # Quitamos las columnas categóricas originales
X_final = pd.concat([X_restantes, X_categoricas], axis=1)  # Concatenamos las columnas transformadas

# Paso 4: Escalar las variables continuas
variables_continuas = ['tenure', 'age', 'income','employ'] # Asegúrate de identificar correctamente las variables continuas
scaler = StandardScaler()
X_continuas = X_final[variables_continuas]  # Seleccionamos las variables continuas
X_continuas_scaled = scaler.fit_transform(X_continuas)  # Escalamos

# Reemplazamos las variables continuas escaladas
X_final[variables_continuas] = X_continuas_scaled
print(X_final)


# Paso 5: Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X_final, Y, test_size=0.2, random_state=42)

# Paso 6: Entrenar el modelo con Random Forest Classifier
# Inicializa y entrena el modelo de KNN
knn = KNeighborsClassifier(n_neighbors=5)  # Ajusta el valor de k
knn.fit(X_train, Y_train)

# Realiza predicciones en el conjunto de prueba
Y_pred = knn.predict(X_test)

# Calcula la precisión del modelo
precision_knn = accuracy_score(Y_test, Y_pred)
print(f"Precisión con KNN en el conjunto de prueba: {precision_knn:.4f}")

# Realizar validación cruzada para estimar el rendimiento
cv_scores_knn = cross_val_score(knn, X_train, Y_train, cv=5)
print(f"Precisión promedio en validación cruzada con KNN: {cv_scores_knn.mean():.4f}")
print(f"Desviación estándar de la validación cruzada con KNN: {cv_scores_knn.std():.4f}")