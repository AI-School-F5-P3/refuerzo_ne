import pandas as pd
import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import SelectKBest, f_classif
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from config_ruta import ruta_datos 
from load import cargar_datos
datos = pd.DataFrame(cargar_datos(ruta_datos))


# Inicializar el transformador KBinsDiscretizer
# n_bins: número de intervalos deseados
# encode: tipo de codificación ('ordinal' para convertir a categorías ordinales)
# strategy: método para dividir los datos (uniforme, cuantiles, kmeans)
kbd = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')

# Ajustar y transformar la columna 'tenure'
datos['tenure_cat'] = kbd.fit_transform(datos[['tenure']]).astype('int')
#df['tenure_cat'] = pd.Categorical(df['tenure_cat'], categories=['Bajo', 'Medio', 'Alto'], ordered=True) #para CatBoost


# Ajustar y transformar la columna 'age'
datos['age_cat'] = kbd.fit_transform(datos[['age']]).astype('int')

# Ajustar y transformar la columna 'age'
datos['income_cat'] = kbd.fit_transform(datos[['income']]).astype('int')


#
# Ajustar y transformar la columna 'employ'
datos['employ_cat'] = kbd.fit_transform(datos[['employ']]).astype('int')

# Ajustar y transformar la columna 'ed'
datos['ed_cat']=pd.Categorical(datos['ed'], ordered=True) #para CatBoost


# Ajustar y transformar la columna 'reside'
datos['reside_cat']=pd.Categorical(datos['reside'], ordered=True) #para CatBoost

# Mostrar el DataFrame transformado
print(datos)
print(datos.info())

datos=datos[['age_cat','ed_cat','income_cat','reside_cat','tenure_cat','employ_cat','custcat','retire']]

#Dividir el set en 'x' e 'Y'
def dividir_datos(datos):


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
X, Y = dividir_datos(datos)

if X is not None and Y is not None:
    print("Variables independientes (X):")
    print(X.head())
    print("Variable dependiente (Y):")
    print(Y.head())
else:
    print("No se pudieron dividir los datos.")
    
    
# Seleccionar las mejores características
selector = SelectKBest(score_func=f_classif, k=5)
X_new = selector.fit_transform(X, Y)
selected_features = X.columns[selector.get_support()]
print(selected_features)




print(X_new) 
 # 1. Dividir los datos en train y test (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)




#----------------

# Crear y entrenar el modelo
model = LogisticRegression(max_iter=100)  # max_iter controla el número máximo de iteraciones

model.fit(X_train, Y_train)

# Hacer predicciones
y_pred = model.predict(X_test)

# 5. Calcular la precisión del modelo en el conjunto de prueba
precision = accuracy_score(Y_test, y_pred)
print(f"Precisión en el conjunto de prueba: {precision:.4f}")

# 6. Realizar validación cruzada con 5 particiones
# Esto realiza validación cruzada sobre el conjunto de entrenamiento para obtener una medida más robusta del rendimiento del modelo
cv_scores = cross_val_score(model, X_train, Y_train, cv=5)

# Mostrar los resultados de la validación cruzada
print(f"Precisión promedio en validación cruzada: {cv_scores.mean():.4f}")
print(f"Desviación estándar de la validación cruzada: {cv_scores.std():.4f}")


from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression

# Definir los hiperparámetros a optimizar
param_grid = {
    'penalty': ['l1', 'l2'],           # Tipos de regularización
    'C': [0.01, 0.1, 1, 10, 100],     # Inverso de la fuerza de regularización
    'solver': ['liblinear', 'saga'],  # Algoritmos para optimizar
    'max_iter': [100, 200, 500]       # Número de iteraciones máximas
}

# Crear el modelo base
logistic_model = LogisticRegression()

# Crear el objeto GridSearch
grid_search = GridSearchCV(estimator=logistic_model, 
                           param_grid=param_grid, 
                           cv=5,                  # Número de particiones para validación cruzada
                           scoring='accuracy',   # Métrica para evaluar
                           verbose=1, 
                           n_jobs=-1)            # Paralelismo para acelerar la búsqueda

# Realizar la búsqueda en el conjunto de entrenamiento
grid_search.fit(X_train, Y_train)

# Mostrar los mejores hiperparámetros
print(f"Mejores hiperparámetros: {grid_search.best_params_}")
print(f"Mejor precisión: {grid_search.best_score_:.4f}")

# Entrenar el modelo final con los mejores hiperparámetros
best_model = grid_search.best_estimator_
best_model.fit(X_train, Y_train)

# Evaluar en el conjunto de prueba
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(Y_test, y_pred)
print(f"Precisión en el conjunto de prueba con gridSearch: {accuracy:.4f}")

