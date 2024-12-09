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

    # Cargar los datos usando la función cargar_datos

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
selector = SelectKBest(score_func=f_classif, k=6)
X_new = selector.fit_transform(X, Y)
selected_features = X.columns[selector.get_support()]
print(selected_features)

# One-Hot Encoding para variables categóricas relevantes


print(X_new) 
 # 1. Dividir los datos en train y test (80% entrenamiento, 20% prueba)
X_train, X_test, Y_train, Y_test = train_test_split(X_new, Y, test_size=0.2, random_state=42)




#----------------CATBOOST

# Crear y entrenar el modelo CatBoost
model = CatBoostClassifier(iterations=100,         # Número de iteraciones (árboles)
                           learning_rate=0.1,      # Tasa de aprendizaje
                           depth=6,                # Profundidad de los árboles
                            # Columnas categóricas
                           verbose=10)             # Mostrar progreso

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

