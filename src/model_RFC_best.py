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